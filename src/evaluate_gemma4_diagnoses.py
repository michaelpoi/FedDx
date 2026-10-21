#!/usr/bin/env python3
"""FedDx Gemma 4 diagnosis evaluation with cached/resumable predictions.

Run from project root:
  uv run python src/evaluate_gemma4_diagnoses.py --model-id vanilla
  uv run python src/evaluate_gemma4_diagnoses.py --model-id 1000
  uv run python src/evaluate_gemma4_diagnoses.py --model-id 3000
  uv run python src/evaluate_gemma4_diagnoses.py --model-id 5000
  uv run python src/evaluate_gemma4_diagnoses.py --model-id all

Predictions are appended to JSONL and reused on subsequent runs. A strict manifest
rejects caching across changed data, prompts, model, adapters, or decode settings.
Use --reset-cache to deliberately regenerate one model's predictions.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

PROMPT = (
    "You are a clinical diagnosis classifier. Given the clinical summary, "
    "output only the single most likely diagnosis name, without explanation.\n\n"
    "Clinical summary:\n{summary}"
)
INVALID = "[INVALID / OUT OF VOCABULARY]"


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_rows(path):
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            pid = r["patient_id"]
            if pid in seen:
                raise ValueError(f"Duplicate patient ID in {path}:{line_no}: {pid}")
            seen.add(pid)
            if not r["clinical_summary"].strip() or not r["target"]["diagnosis"].strip():
                raise ValueError(f"Missing clinical summary or diagnosis at {path}:{line_no}")
            rows.append(r)
    if not rows:
        raise ValueError(f"Empty validation dataset: {path}")
    return rows


def normalized(s):
    s = unicodedata.normalize("NFKC", s).casefold().strip()
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .\t\n\r\"'`")


def predicted_class(raw, labels):
    # Strict classification: don't count mentions buried in explanations or
    # clinical evidence; report out-of-vocabulary outputs explicitly.
    t = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    t = re.sub(r"^(?:diagnosis|answer)\s*:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^(?:the most likely diagnosis is|most likely diagnosis is)\s+", "", t, flags=re.IGNORECASE)
    lookup = {normalized(label): label for label in labels}
    if len(lookup) != len(labels):
        raise ValueError("Two labels normalize to the same string")
    return lookup.get(normalized(t), INVALID)


def adapter_signature(path):
    # Hash config + actual adapter weights to invalidate stale prediction cache.
    if not path.is_dir() or not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter at {path}")
    weights = [p for p in path.iterdir() if p.name in ("adapter_model.safetensors", "adapter_model.bin")]
    if len(weights) != 1:
        raise FileNotFoundError(f"Expected one adapter_model.safetensors/bin in {path}")
    dig = hashlib.sha256()
    for p in [path / "adapter_config.json", weights[0]]:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                dig.update(chunk)
    return dig.hexdigest()


def write_json_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def existing_predictions(path, expected):
    found = {}
    if not path.exists():
        return found
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A crash might have interrupted the final append. Repair only
                # by explicit human action to avoid dropping a valid record.
                raise ValueError(f"Corrupted prediction line {number} in {path}; repair it before resume")
            pid = row["patient_id"]
            if pid not in expected or row["truth"] != expected[pid] or pid in found:
                raise ValueError(f"Unexpected/duplicated/mismatched prediction at {path}:{number}")
            found[pid] = row
    return found


def report(rows, predictions, labels, dest, model_id):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, f1_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = [r["target"]["diagnosis"] for r in rows]
    y_pred = [predictions[r["patient_id"]]["prediction"] for r in rows]
    axes = labels + [INVALID]
    matrix = confusion_matrix(y_true, y_pred, labels=axes)
    used = set(y_true)
    metrics = {
        "model": model_id,
        "n": len(rows),
        "classes_in_validation": len(used),
        "classes_in_vocabulary": len(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_present_classes": float(f1_score(y_true, y_pred, labels=sorted(used), average="macro", zero_division=0)),
        "macro_f1_full_vocabulary": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy_present_classes": float(balanced_accuracy_score(y_true, y_pred)),
        "invalid_count": sum(p == INVALID for p in y_pred),
    }
    dest.mkdir(parents=True, exist_ok=True)
    write_json_atomic(dest / "metrics.json", metrics)
    with (dest / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted", *axes])
        for label, row in zip(axes, matrix):
            w.writerow([label, *row.tolist()])
    report_dict = classification_report(y_true, y_pred, labels=axes, output_dict=True, zero_division=0)
    write_json_atomic(dest / "classification_report.json", report_dict)
    with (dest / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, labels=axes, zero_division=0))

    # Hide INVALID's all-zero true row, but retain its predicted column.
    plot_matrix = matrix[:len(labels), :]
    fig, ax = plt.subplots(figsize=(22, 19))
    im = ax.imshow(plot_matrix, interpolation="nearest", cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(axes)), axes, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ax.set_xlabel("Predicted diagnosis (last column = invalid answer)")
    ax.set_ylabel("True diagnosis")
    ax.set_title(f"{model_id}: {len(rows)} cases | accuracy={metrics['accuracy']:.3f} | macro F1={metrics['macro_f1_present_classes']:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(dest / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    print(f"{model_id}: accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1_present_classes']:.4f}, invalid={metrics['invalid_count']}")


def eval_one(args, rows, labels, model_id):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    dest = args.output_root / args.split / ("vanilla" if model_id == "vanilla" else f"stage-{model_id}")
    pred_path = dest / "predictions.jsonl"
    manifest_path = dest / "manifest.json"
    adapter = None if model_id == "vanilla" else args.adapter_root / f"stage-{model_id}" / "adapter_final"
    signature = None if adapter is None else adapter_signature(adapter)
    manifest = {
        "version": 1, "model": args.model, "model_revision": args.revision,
        "model_id": model_id, "dataset": str(args.data.resolve()),
        "dataset_fingerprint": sha([[r["patient_id"], r["clinical_summary"], r["target"]["diagnosis"]] for r in rows]),
        "vocabulary": labels, "prompt": PROMPT, "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy", "precision": "4bit_nf4_bf16_or_fp16",
        "adapter_signature": signature,
    }
    if args.reset_cache:
        if pred_path.exists():
            pred_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError(f"Cache configuration changed for {model_id}: {manifest_path}. Use --reset-cache to recompute.")
    elif pred_path.exists():
        raise RuntimeError(f"Predictions exist without manifest at {pred_path}; refusing unsafe cache reuse")
    else:
        write_json_atomic(manifest_path, manifest)
    expected = {r["patient_id"]: r["target"]["diagnosis"] for r in rows}
    predictions = existing_predictions(pred_path, expected)
    pending = [r for r in rows if r["patient_id"] not in predictions]
    print(f"{model_id}: cached {len(predictions)}/{len(rows)}, pending {len(pending)}", flush=True)

    if pending:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for inference")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
        processor.tokenizer.padding_side = "left"
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
        pad = processor.tokenizer.pad_token_id
        if pad is None:
            pad = processor.tokenizer.eos_token_id
        # Flush a prediction after every case: interruption loses at most one item.
        with pred_path.open("a", encoding="utf-8") as f:
            for i, r in enumerate(pending, 1):
                question = PROMPT.format(summary=r["clinical_summary"].strip())
                messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
                # Match the assistant prefix used during fine-tuning.
                rendered = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                thought_suffix = "<|channel>thought\n<channel|>"

                if not rendered.endswith(thought_suffix):
                    raise RuntimeError(
                        f"Unexpected Gemma generation template: {rendered[-120:]!r}"
                    )

                rendered = rendered[:-len(thought_suffix)]

                if not rendered.endswith("<|turn>model\n"):
                    raise RuntimeError("Unexpected assistant prefix")

                inputs = processor(
                    text=rendered,
                    return_tensors="pt",
                ).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens,
                        do_sample=False, pad_token_id=pad,
                    )
                raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                prediction = predicted_class(raw, labels)
                value = {"patient_id": r["patient_id"], "truth": r["target"]["diagnosis"],
                         "prediction": prediction, "raw_response": raw}
                f.write(json.dumps(value, ensure_ascii=False) + "\n")
                f.flush()
                predictions[r["patient_id"]] = value
                if i % 25 == 0 or i == len(pending):
                    print(f"{model_id}: generated {i}/{len(pending)} (total {len(predictions)}/{len(rows)})", flush=True)
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    report(rows, predictions, labels, dest, model_id)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", choices=("vanilla", "1000", "3000", "5000", "all"), required=True)
    p.add_argument("--split", choices=("validate", "test"), default="validate")
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--model", default="google/gemma-4-12B-it")
    p.add_argument("--revision", default=None, help="Optional pinned Hugging Face revision")
    p.add_argument("--adapter-root", type=Path, default=Path("runs/gemma4_qlora"))
    p.add_argument("--output-root", type=Path, default=Path("runs/gemma4_evaluation"))
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--reset-cache", action="store_true", help="Discard existing cached predictions for selected model(s)")
    p.add_argument("--reports-only", action="store_true", help="Recompute plots/metrics from completed cached predictions, no GPU")
    args = p.parse_args()
    if args.data is None:
        args.data = Path(f"data/generated_clinical_v2/{args.split}.jsonl")
    rows = read_rows(args.data)
    # Use all training diagnosis labels, not just those found in validation:
    # otherwise a correct prediction for a rare absent validation class looks invalid.
    train_file = Path("data/generated_clinical_v2/train.jsonl")
    train_rows = read_rows(train_file)
    labels = sorted({r["target"]["diagnosis"] for r in train_rows} | {r["target"]["diagnosis"] for r in rows})
    wanted = ("vanilla", "1000", "3000", "5000") if args.model_id == "all" else (args.model_id,)
    for model_id in wanted:
        if args.reports_only:
            dest = args.output_root / args.split / ("vanilla" if model_id == "vanilla" else f"stage-{model_id}")
            expected = {r["patient_id"]: r["target"]["diagnosis"] for r in rows}
            predictions = existing_predictions(dest / "predictions.jsonl", expected)
            if len(predictions) != len(rows):
                raise RuntimeError(f"Missing {len(rows)-len(predictions)} predictions for {model_id}")
            report(rows, predictions, labels, dest, model_id)
        else:
            eval_one(args, rows, labels, model_id)


if __name__ == "__main__":
    main()
