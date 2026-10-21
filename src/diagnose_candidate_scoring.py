#!/usr/bin/env python3
"""Investigate mean-token vs total-logprob diagnosis ranking (FedDx Gemma 4).

No evaluation cache is modified. Uses teacher-forced token probabilities for
complete model assistant responses under the exact training conversation format.
"""
import argparse
import json
from pathlib import Path

PROMPT = (
    "You are a clinical diagnosis classifier. Given the clinical summary, "
    "output only the single most likely diagnosis name, without explanation.\n\n"
    "Clinical summary:\n{summary}"
)


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rendered(processor, question, answer):
    return processor.apply_chat_template(
        [
            {"role": "user", "content": [{"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ], tokenize=False, add_generation_prompt=False,
    )


def token_scoring_sequence(processor, question, diagnosis):
    """Return IDs and separate diagnosis and ending-token positions."""
    marker = "FEDDX_UNIQUE_ASSISTANT_SLOT_8D22AB"
    marked = rendered(processor, question, marker)
    full = rendered(processor, question, diagnosis)
    if marked.count(marker) != 1 or marked.replace(marker, diagnosis) != full:
        raise RuntimeError("Cannot reliably locate diagnosis in rendered Gemma chat")
    start = marked.index(marker)
    end = start + len(diagnosis)
    tok = processor.tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = tok["input_ids"], tok["offset_mapping"]
    # Exact copy of training's assistant-response region selection, with separate
    # buckets for diagnosis tokens and suffix (e.g. end-of-turn + newline).
    diag_pos = [i for i, (a, b) in enumerate(offsets) if a >= start and a < end and b > a]
    if not diag_pos or diag_pos[0] == 0:
        raise RuntimeError(f"No valid diagnosis token positions: {diagnosis}")
    if processor.tokenizer.decode([ids[i] for i in diag_pos], skip_special_tokens=True).strip() != diagnosis:
        raise RuntimeError(f"Diagnosis-token boundary mismatch: {diagnosis!r}")
    suffix_pos = [i for i in range(diag_pos[0] + 1, len(ids))
                  if (offsets[i][1] <= offsets[i][0] or offsets[i][0] >= end)
                  and i not in diag_pos]
    if not suffix_pos:
        raise RuntimeError("No assistant-ending tokens found")
    return ids, diag_pos, suffix_pos


def score(model, processor, question, diagnosis, torch):
    ids, diag_pos, suffix_pos = token_scoring_sequence(processor, question, diagnosis)
    tensor = torch.tensor([ids], device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=tensor, attention_mask=torch.ones_like(tensor)).logits
        lprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    def value(pos):
        return float(lprobs[pos - 1, ids[pos]].item())
    diag = [value(i) for i in diag_pos]
    ending = [value(i) for i in suffix_pos]
    def summarize(values):
        return {"total": sum(values), "mean": sum(values) / len(values), "tokens": len(values)}
    return {
        "diagnosis": summarize(diag),
        "complete_response": summarize(diag + ending),
        "diagnosis_tokens": processor.tokenizer.convert_ids_to_tokens([ids[i] for i in diag_pos]),
        "diagnosis_token_logprobs": [round(v, 4) for v in diag],
        "ending_tokens": processor.tokenizer.convert_ids_to_tokens([ids[i] for i in suffix_pos]),
    }


def winner(scores, key, metric):
    return max(scores, key=lambda label: scores[label][key][metric])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, default=Path("runs/gemma4_qlora_independent/stage-5000/adapter_final"))
    p.add_argument("--model", default="google/gemma-4-12B-it")
    p.add_argument("--validation", type=Path, default=Path("data/generated_clinical_v2/validate.jsonl"))
    p.add_argument("--open-predictions", type=Path,
                   default=Path("runs/gemma4_evaluation_independent/validate/stage-5000/predictions.jsonl"))
    p.add_argument("--closed-predictions", type=Path,
                   default=Path("runs/gemma4_evaluation_closed_set_independent/validate/stage-5000/predictions.jsonl"))
    p.add_argument("--pair", nargs=2, default=["Stable angina", "Unstable angina"], metavar=("LABEL_A", "LABEL_B"))
    p.add_argument("--n", type=int, default=3, help="Number of disagreements to inspect")
    args = p.parse_args()

    rows = {r["patient_id"]: r for r in load_jsonl(args.validation)}
    open_predictions = {r["patient_id"]: r for r in load_jsonl(args.open_predictions)}
    closed_predictions = {r["patient_id"]: r for r in load_jsonl(args.closed_predictions)}
    cases = [r for r in rows.values()
             if r["target"]["diagnosis"] == args.pair[0]
             and open_predictions.get(r["patient_id"], {}).get("prediction") == args.pair[0]
             and closed_predictions.get(r["patient_id"], {}).get("prediction") == args.pair[1]]
    cases = cases[:args.n]
    if not cases:
        raise SystemExit("No matching disagreements found. Check --pair and prediction file paths.")
    print(f"Selected {len(cases)} cases; adapter={args.adapter}", flush=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(args.model)
    base = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=dtype, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_storage=dtype,
        ),
    )
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    model.eval()

    for case in cases:
        question = PROMPT.format(summary=case["clinical_summary"].strip())
        results = {label: score(model, processor, question, label, torch) for label in args.pair}
        print(f"\n=== {case['patient_id']} truth={case['target']['diagnosis']} ===")
        print("Cached open prediction:", open_predictions[case["patient_id"]]["prediction"])
        print("Cached closed prediction:", closed_predictions[case["patient_id"]]["prediction"])
        for label, record in results.items():
            print(f"{label}: diagnosis tokens={record['diagnosis_tokens']}")
            print("  individual logprobs:", record["diagnosis_token_logprobs"])
            print("  ending tokens:", record["ending_tokens"])
            for key in ("diagnosis", "complete_response"):
                d = record[key]
                print(f"  {key}: total={d['total']:.4f} mean={d['mean']:.4f} tokens={d['tokens']}")
        for key in ("diagnosis", "complete_response"):
            print(f"WINNER {key}: total={winner(results, key, 'total')}; mean={winner(results, key, 'mean')}")
    print("\nNote: This uses the *training-matched prompt*. The earlier closed-set evaluator may also")
    print("use a candidate-list prompt, so differences cannot be attributed to length normalization alone.")


if __name__ == "__main__":
    main()