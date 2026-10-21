#!/usr/bin/env python3
"""FedDx closed-set diagnosis evaluation by teacher-forced candidate scoring.

For each clinical summary, score the same complete vocabulary of diagnosis strings
under the base model or a LoRA adapter. No free-text decoding, no invalid responses.
Each completed patient prediction is appended, so interrupted runs can resume.

Examples:
  uv run python src/evaluate_gemma4_closed_set.py --model-id vanilla
  uv run python src/evaluate_gemma4_closed_set.py --model-id 1000
  uv run python src/evaluate_gemma4_closed_set.py --model-id all
  uv run python src/evaluate_gemma4_closed_set.py --model-id all --reports-only

This is an independent evaluation protocol: results live under
runs/gemma4_evaluation_closed_set and do not touch free-generation caches.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from evaluate_gemma4_diagnoses import (
    PROMPT, adapter_signature, existing_predictions, read_rows, report, sha,
    write_json_atomic,
)


PROTOCOL_VERSION = 1


def build_question(summary: str, labels: list[str]) -> str:
    # Keep the original training prompt and explicitly state the candidate set.
    return (PROMPT.format(summary=summary.strip()) +
            "\n\nChoose exactly one diagnosis from the following candidates:\n" +
            "\n".join(f"- {label}" for label in labels) +
            "\nRespond with only the chosen diagnosis name.")


def score_labels(model, processor, question: str, labels: list[str], chunk_size: int, max_length: int):
    """Return per-label mean token log-probability under the same prompt.

    Prefix is tokenized with Gemma's real chat template. The candidate diagnosis
    token IDs are appended *after* the assistant-prefix tokens, without EOS.
    Candidate scoring includes every diagnosis token and excludes the prompt.
    """
    import torch
    from torch.nn import functional as F

    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    encoded = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    prefix = encoded["input_ids"][0].tolist()
    if len(prefix) >= max_length:
        raise ValueError(f"Prompt is {len(prefix)} tokens, >= max-length={max_length}; refusing truncation")
    tokenizer = processor.tokenizer
    # The assistant prefix ends with a special role boundary. Candidate strings
    # can be tokenized as standalone text and concatenated after that boundary.
    candidates = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(not ids for ids in candidates):
        raise ValueError("At least one candidate label tokenized to zero tokens")
    if len(prefix) + max(map(len, candidates)) > max_length:
        raise ValueError("Prompt + diagnosis exceeds max-length; increase --max-length")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad or EOS token")
    device = model.device
    scores = []
    for start in range(0, len(candidates), chunk_size):
        subset = candidates[start:start + chunk_size]
        longest = max(len(prefix) + len(ids) for ids in subset)
        input_ids, masks = [], []
        for ids in subset:
            actual = prefix + ids
            pad = longest - len(actual)
            input_ids.append(actual + [pad_id] * pad)
            masks.append([1] * len(actual) + [0] * pad)
        input_tensor = torch.tensor(input_ids, device=device, dtype=torch.long)
        mask_tensor = torch.tensor(masks, device=device, dtype=torch.long)
        with torch.inference_mode():
            output = model(input_ids=input_tensor, attention_mask=mask_tensor, use_cache=False)
            logits = output.logits
            # Position prefix_len - 1 predicts the first diagnosis token.
            start_logits = len(prefix) - 1
            end_logits = start_logits + max(len(ids) for ids in subset)
            selected = logits[:, start_logits:end_logits, :].float()
            log_probs = F.log_softmax(selected, dim=-1)
            for k, ids in enumerate(subset):
                n = len(ids)
                targets = torch.tensor(ids, device=device, dtype=torch.long)
                token_lp = log_probs[k, :n].gather(-1, targets[:, None]).squeeze(-1)
                scores.append(float(token_lp.mean().item()))
        del output, logits, input_tensor, mask_tensor
    return scores


def evaluate_one(args, rows, labels, model_id):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    dest = args.output_root / args.split / ("vanilla" if model_id == "vanilla" else f"stage-{model_id}")
    pred_path = dest / "predictions.jsonl"
    manifest_path = dest / "manifest.json"
    adapter = None if model_id == "vanilla" else args.adapter_root / f"stage-{model_id}" / "adapter_final"
    signature = None if adapter is None else adapter_signature(adapter)
    manifest = {
        "protocol": "teacher_forced_candidate_mean_logprob", "version": PROTOCOL_VERSION,
        "model": args.model, "revision": args.revision, "model_id": model_id,
        "data_path": str(args.data.resolve()),
        "dataset_fingerprint": sha([[r["patient_id"], r["clinical_summary"], r["target"]["diagnosis"]] for r in rows]),
        "vocabulary": labels, "prompt_template": PROMPT,
        "prompt_suffix": "candidate_list_all_labels_v1", "label_scoring": "mean_logprob_without_eos",
        "max_length": args.max_length, "precision": "4bit_nf4_bf16_or_fp16",
        "adapter_signature": signature,
    }
    if args.reset_cache:
        for p in (pred_path, manifest_path):
            if p.exists():
                p.unlink()
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"Cache mismatch at {manifest_path}; use --reset-cache to rebuild")
    elif pred_path.exists():
        raise RuntimeError(f"Predictions without manifest at {pred_path}; refusing unsafe reuse")
    else:
        write_json_atomic(manifest_path, manifest)

    expected = {r["patient_id"]: r["target"]["diagnosis"] for r in rows}
    predictions = existing_predictions(pred_path, expected)
    pending = [r for r in rows if r["patient_id"] not in predictions]
    print(f"{model_id}: {len(predictions)} cached / {len(rows)} total; {len(pending)} remaining", flush=True)
    if pending:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for candidate likelihood scoring")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForMultimodalLM.from_pretrained(
            args.model, revision=args.revision, dtype=dtype, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_storage=dtype,
            ),
        )
        if adapter is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("a", encoding="utf-8") as f:
            for i, r in enumerate(pending, 1):
                question = build_question(r["clinical_summary"], labels)
                scores = score_labels(model, processor, question, labels, args.candidate_batch, args.max_length)
                best = max(range(len(scores)), key=lambda k: scores[k])
                runner_up = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:3]
                record = {
                    "patient_id": r["patient_id"], "truth": r["target"]["diagnosis"],
                    "prediction": labels[best], "score": scores[best],
                    "top3": [{"diagnosis": labels[k], "mean_logprob": scores[k]} for k in runner_up],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                predictions[record["patient_id"]] = record
                if i % 10 == 0 or i == len(pending):
                    print(f"{model_id}: {i}/{len(pending)} new predictions", flush=True)
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    report(rows, predictions, labels, dest, f"{model_id} closed-set")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", choices=("vanilla", "1000", "3000", "5000", "all"), required=True)
    p.add_argument("--split", choices=("validate", "test"), default="validate")
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--train-data", type=Path, default=Path("data/generated_clinical_v2/train.jsonl"))
    p.add_argument("--model", default="google/gemma-4-12B-it")
    p.add_argument("--revision", default=None)
    p.add_argument("--adapter-root", type=Path, default=Path("runs/gemma4_qlora"))
    p.add_argument("--output-root", type=Path, default=Path("runs/gemma4_evaluation_closed_set"))
    p.add_argument("--candidate-batch", type=int, default=4,
                   help="Candidates evaluated together (reduce to 1 if OOM)")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--reports-only", action="store_true")
    p.add_argument("--reset-cache", action="store_true")
    args = p.parse_args()
    if args.data is None:
        args.data = Path(f"data/generated_clinical_v2/{args.split}.jsonl")
    if args.candidate_batch < 1 or args.max_length < 128:
        p.error("Invalid candidate-batch or max-length")
    if args.reports_only and args.reset_cache:
        p.error("Cannot combine --reports-only and --reset-cache")
    rows = read_rows(args.data)
    train_rows = read_rows(args.train_data)
    labels = sorted({r["target"]["diagnosis"] for r in train_rows} | {r["target"]["diagnosis"] for r in rows})
    wanted = ("vanilla", "1000", "3000", "5000") if args.model_id == "all" else (args.model_id,)
    for model_id in wanted:
        if args.reports_only:
            dest = args.output_root / args.split / ("vanilla" if model_id == "vanilla" else f"stage-{model_id}")
            predictions = existing_predictions(dest / "predictions.jsonl", {r["patient_id"]: r["target"]["diagnosis"] for r in rows})
            if len(predictions) != len(rows):
                raise RuntimeError(f"Missing {len(rows)-len(predictions)} predictions for {model_id}")
            report(rows, predictions, labels, dest, f"{model_id} closed-set")
        else:
            evaluate_one(args, rows, labels, model_id)


if __name__ == "__main__":
    main()
