#!/usr/bin/env python3
"""Evaluate open-ended Gemma 4 diagnosis-token confidence on a test sample.

Each completed prediction is appended to JSONL. An input/model configuration
manifest prevents silently mixing incompatible runs. Reports can be regenerated
without a GPU using --reports-only.

The scores are UNCALIBRATED token likelihoods, not probabilities of a correct
medical diagnosis. The red/yellow/green bands use the same 0.50/0.80 heuristic
thresholds as the Gradio interface.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

from feddx_confidence import confidence_band, score_generated_diagnosis

PROMPT = (
    "You are a clinical diagnosis classifier. Given the clinical summary, "
    "output only the single most likely diagnosis name, without explanation.\n\n"
    "Clinical summary:\n{summary}"
)
INVALID = "[INVALID / OUT OF VOCABULARY]"


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows, ids = [], set()
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row["patient_id"]
            if pid in ids:
                raise ValueError(f"Duplicate patient_id {pid} in {path}:{number}")
            ids.add(pid)
            rows.append(row)
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = value.replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", value).strip(" .\t\n\r\"'`")


def predict_label(raw: str, lookup: dict[str, str]) -> str:
    # Remove exactly a leading decoded channel header (not arbitrary reasoning).
    raw = re.sub(r"^\s*thought\s*\n", "", raw, count=1, flags=re.IGNORECASE)
    first = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    first = re.sub(r"^(?:diagnosis|answer)\s*:\s*", "", first, flags=re.IGNORECASE)
    first = re.sub(r"^(?:the most likely diagnosis is|most likely diagnosis is)\s+", "", first, flags=re.IGNORECASE)
    return lookup.get(norm(first), INVALID)


def json_hash(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def make_manifest(args: argparse.Namespace, selected: list[dict], labels: list[str]) -> dict:
    adapter = None if args.model_id == "vanilla" else args.adapter_root / f"stage-{args.model_id}" / "adapter_final"
    if adapter is not None:
        # Reuse the existing evaluator's actual adapter weight fingerprint.
        from evaluate_gemma4_diagnoses import adapter_signature
        adapter_sig = adapter_signature(adapter)
    else:
        adapter_sig = None
    return {
        "format_version": 1,
        "model_id": args.model_id,
        "base_model": args.model,
        "revision": args.revision,
        "adapter_root": str(args.adapter_root) if adapter else None,
        "adapter_signature": adapter_sig,
        "data_file": str(args.data.resolve()),
        "selected_records_hash": json_hash([[r["patient_id"], r["clinical_summary"], r["target"]["diagnosis"]] for r in selected]),
        "vocabulary_hash": json_hash(labels),
        "seed": args.seed,
        "n": len(selected),
        "prompt": PROMPT,
        "max_new_tokens": args.max_new_tokens,
        "decode": "greedy",
        "confidence_method": "generated_diagnosis_token_logprobs",
    }


def select_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n < 1 or n > len(rows):
        raise ValueError(f"--n must be between 1 and {len(rows)}")
    # Deterministic sample; preserve randomized draw order for reproducibility.
    return random.Random(seed).sample(rows, n)


def load_previous(path: Path, selected: list[dict]) -> dict[str, dict]:
    allowed = {r["patient_id"]: r["target"]["diagnosis"] for r in selected}
    found = {}
    if not path.exists():
        return found
    for row in read_jsonl(path):
        pid = row["patient_id"]
        if pid not in allowed or row.get("truth") != allowed[pid] or pid in found:
            raise ValueError(f"Invalid/duplicated prediction in {path}: {pid}")
        found[pid] = row
    return found


def inference(args: argparse.Namespace, selected: list[dict], lookup: dict[str, str], dest: Path, previous: dict) -> None:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    pending = [r for r in selected if r["patient_id"] not in previous]
    if not pending:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to generate new confidence scores")
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
    if args.model_id != "vanilla":
        from peft import PeftModel
        adapter = args.adapter_root / f"stage-{args.model_id}" / "adapter_final"
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    pad = processor.tokenizer.pad_token_id
    if pad is None:
        pad = processor.tokenizer.eos_token_id

    with (dest / "predictions.jsonl").open("a", encoding="utf-8") as output:
        for index, record in enumerate(pending, 1):
            messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT.format(summary=record["clinical_summary"].strip())}]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=pad,
                    return_dict_in_generate=True, output_scores=True,
                )
            raw = processor.tokenizer.decode(
                generated.sequences[0, input_len:],
                skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            prediction = predict_label(raw, lookup)
            try:
                confidence = score_generated_diagnosis(
                    processor.tokenizer, generated.sequences, generated.scores, input_len,
                )
            except Exception as e:
                confidence = {"available": False, "reason": f"Scoring error: {type(e).__name__}: {e}"}
            score = confidence.get("geometric_mean_token_probability") if confidence.get("available") else None
            band = confidence_band(score)[0] if score is not None else "UNAVAILABLE"
            row = {
                "patient_id": record["patient_id"],
                "truth": record["target"]["diagnosis"],
                "prediction": prediction,
                "correct": prediction == record["target"]["diagnosis"],
                "raw_response": raw,
                "confidence_band": band,
                "confidence": confidence,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            previous[record["patient_id"]] = row
            if index % 10 == 0 or index == len(pending):
                print(f"New predictions: {index}/{len(pending)}; total: {len(previous)}/{len(selected)}", flush=True)
            del generated, inputs


def summarize(records: list[dict]) -> dict:
    scored = [r for r in records if r.get("confidence", {}).get("available")]
    def get(r, key):
        return float(r["confidence"][key])
    n = len(records)
    ordered = sorted(get(r, "geometric_mean_token_probability") for r in scored)
    def quantile(p: float):
        if not ordered:
            return None
        k = (len(ordered)-1)*p
        lo, hi = math.floor(k), math.ceil(k)
        return ordered[lo] + (ordered[hi]-ordered[lo])*(k-lo)
    groups = {}
    for band in ("LOW", "MODERATE", "HIGH", "UNAVAILABLE"):
        subset = [r for r in records if r["confidence_band"] == band]
        groups[band] = {
            "count": len(subset), "fraction": len(subset)/n if n else None,
            "accuracy": sum(r["correct"] for r in subset)/len(subset) if subset else None,
        }
    result = {
        "n": n, "scored_n": len(scored),
        "overall_accuracy": sum(r["correct"] for r in records)/n if n else None,
        "invalid_count": sum(r["prediction"] == INVALID for r in records),
        "confidence_bands": groups,
        "geometric_mean_token_probability": {
            "mean": statistics.mean(ordered) if ordered else None,
            "median": quantile(0.5), "min": min(ordered) if ordered else None,
            "max": max(ordered) if ordered else None,
            "p05": quantile(0.05), "p25": quantile(0.25),
            "p75": quantile(0.75), "p95": quantile(0.95),
            "below_0_50": sum(x < 0.5 for x in ordered),
            "between_0_50_and_0_80": sum(0.5 <= x < 0.8 for x in ordered),
            "at_least_0_80": sum(x >= 0.8 for x in ordered),
        },
        "weakest_token_mean": statistics.mean(get(r, "minimum_token_probability") for r in scored) if scored else None,
        "whole_sequence_mean": statistics.mean(get(r, "joint_diagnosis_token_probability") for r in scored) if scored else None,
        "correct_mean_confidence": statistics.mean(get(r, "geometric_mean_token_probability") for r in scored if r["correct"]) if any(r["correct"] for r in scored) else None,
        "incorrect_mean_confidence": statistics.mean(get(r, "geometric_mean_token_probability") for r in scored if not r["correct"]) if any(not r["correct"] for r in scored) else None,
    }
    # ROC AUC of model score for identifying correct predictions, when defined.
    try:
        from sklearn.metrics import roc_auc_score
        ys = [int(r["correct"]) for r in scored]
        result["auroc_correctness"] = float(roc_auc_score(ys, [get(r, "geometric_mean_token_probability") for r in scored])) if len(set(ys)) == 2 else None
    except ImportError:
        result["auroc_correctness"] = None
    return result


def create_reports(dest: Path, selected: list[dict], previous: dict) -> None:
    records = [previous[r["patient_id"]] for r in selected]
    summary = summarize(records)
    (dest / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (dest / "confidence_records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "truth", "prediction", "correct", "band", "mean_token_prob", "weakest_token_prob", "whole_sequence_prob", "n_tokens", "raw_response"])
        for r in records:
            c = r["confidence"]
            writer.writerow([r["patient_id"], r["truth"], r["prediction"], r["correct"], r["confidence_band"], c.get("geometric_mean_token_probability"), c.get("minimum_token_probability"), c.get("joint_diagnosis_token_probability"), c.get("number_of_diagnosis_tokens"), r["raw_response"]])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable: skipping figures")
    else:
        scored = [r for r in records if r["confidence"].get("available")]
        scores = [r["confidence"]["geometric_mean_token_probability"] for r in scored]
        if scores:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(scores, bins=20, range=(0, 1), edgecolor="black")
            ax.axvline(0.5, linestyle="--", label="Low / moderate threshold (0.50)")
            ax.axvline(0.8, linestyle="--", label="Moderate / high threshold (0.80)")
            ax.set(xlabel="Geometric mean diagnosis-token probability (uncalibrated)", ylabel="Number of patients", title="FedDx: generated-answer confidence distribution", xlim=(0, 1))
            ax.legend()
            fig.tight_layout()
            fig.savefig(dest / "confidence_histogram.png", dpi=160)
            plt.close(fig)

            # Risk-coverage curve: most confident cases accepted first.
            ranked = sorted(scored, key=lambda r: r["confidence"]["geometric_mean_token_probability"], reverse=True)
            coverage = [i/len(ranked) for i in range(1, len(ranked)+1)]
            risks = []
            mistakes = 0
            for i, r in enumerate(ranked, 1):
                mistakes += not r["correct"]
                risks.append(mistakes/i)
            with (dest / "risk_coverage.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["coverage", "error_rate", "accepted_cases"])
                writer.writerows((coverage[i], risks[i], i+1) for i in range(len(coverage)))
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(coverage, risks)
            ax.set(xlabel="Coverage (fraction automatically accepted)", ylabel="Error rate among accepted cases", title="Selective prediction: risk vs. coverage", xlim=(0, 1), ylim=(0, 1))
            fig.tight_layout()
            fig.savefig(dest / "risk_coverage.png", dpi=160)
            plt.close(fig)

    print(f"\nResults saved to: {dest}")
    print(f"Records: {summary['n']}; confidence scored: {summary['scored_n']}; accuracy: {summary['overall_accuracy']:.1%}")
    for band, group in summary["confidence_bands"].items():
        accuracy = f"{group['accuracy']:.1%}" if group["accuracy"] is not None else "n/a"
        print(f"  {band:11s} {group['count']:5d} ({group['fraction']:.1%}); accuracy={accuracy}")
    stats = summary["geometric_mean_token_probability"]
    if stats["median"] is not None:
        print(f"Mean confidence: {stats['mean']:.3f}; median: {stats['median']:.3f}; min: {stats['min']:.3f}; p05: {stats['p05']:.3f}")
    print("Note: scores are uncalibrated token likelihoods, not clinical correctness probabilities.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", choices=("vanilla", "1000", "5000"), required=True)
    p.add_argument("--n", type=int, default=100, help="Random test records to evaluate (default 100)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=("test", "validate"), default="test")
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--model", default="google/gemma-4-12B-it")
    p.add_argument("--revision", default=None)
    p.add_argument("--adapter-root", type=Path, default=Path("runs/gemma4_qlora"))
    p.add_argument("--output-root", type=Path, default=Path("runs/feddx_confidence"))
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--reports-only", action="store_true", help="Recompute reports from existing predictions, no GPU")
    args = p.parse_args()
    if args.data is None:
        args.data = Path(f"data/generated_clinical_v2/{args.split}.jsonl")
    rows = read_jsonl(args.data)
    selected = select_rows(rows, args.n, args.seed)
    # The full vocabulary matters; the selected sample may omit rare classes.
    train = read_jsonl(Path("data/generated_clinical_v2/train.jsonl"))
    labels = sorted({r["target"]["diagnosis"] for r in train + rows})
    lookup = {norm(label): label for label in labels}
    if len(lookup) != len(labels):
        raise ValueError("Duplicate label normalization in diagnosis vocabulary")
    dest = args.output_root / args.split / ("vanilla" if args.model_id == "vanilla" else f"stage-{args.model_id}") / f"n{args.n}_seed{args.seed}"
    dest.mkdir(parents=True, exist_ok=True)
    manifest_file = dest / "manifest.json"
    manifest = make_manifest(args, selected, labels)
    if manifest_file.exists():
        if json.loads(manifest_file.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"Run configuration changed: {manifest_file}. Use a new output-root, n, or seed.")
    elif (dest / "predictions.jsonl").exists():
        raise RuntimeError("Existing predictions lack a manifest. Refusing to mix experiments.")
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    previous = load_previous(dest / "predictions.jsonl", selected)
    print(f"{args.model_id}: {len(previous)}/{len(selected)} completed; {len(selected)-len(previous)} pending", flush=True)
    if not args.reports_only:
        inference(args, selected, lookup, dest, previous)
    if len(previous) != len(selected):
        raise RuntimeError(f"Need all {len(selected)} predictions to summarize; have {len(previous)}")
    create_reports(dest, selected, previous)


if __name__ == "__main__":
    main()
