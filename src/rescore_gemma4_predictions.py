#!/usr/bin/env python3
"""Conservatively normalize cached Gemma diagnosis responses; no inference needed.

Usage:
  uv run python src/rescore_gemma4_predictions.py --model-id all
  uv run python src/rescore_gemma4_predictions.py --model-id vanilla --split test

Reads cached raw_response and does NOT modify original predictions/metrics.
Writes reports in <model>/normalized/. Unrecognized diagnoses remain INVALID.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from evaluate_gemma4_diagnoses import INVALID, normalized, predicted_class, read_rows, report, existing_predictions, write_json_atomic

# Deliberately narrow equivalences; do not map related but different conditions.
# A key may be used only if its target is in the study's class vocabulary.
ALIASES = {
    "upper respiratory tract infection": "URTI",
    "upper respiratory infection": "URTI",
    "acute upper respiratory tract infection": "URTI",
    "gastroesophageal reflux disease": "GERD",
    "gastro-oesophageal reflux disease": "GERD",
    "systemic lupus erythematosus": "SLE",
    "supraventricular tachycardia (paroxysmal)": "PSVT",
    "paroxysmal supraventricular tachycardia": "PSVT",
    "paroxysmal svt": "PSVT",
    "guillain-barre syndrome": "Guillain-Barré syndrome",
    "guillain barré syndrome": "Guillain-Barré syndrome",
    "guillain barre syndrome": "Guillain-Barré syndrome",
    "acute otitis media (aom)": "Acute otitis media",
    "acute middle ear infection": "Acute otitis media",
    "atrial fib": "Atrial fibrillation",
    "afib": "Atrial fibrillation",
    "bronchial asthma exacerbation": "Bronchospasm / acute asthma exacerbation",
    "acute asthma exacerbation": "Bronchospasm / acute asthma exacerbation",
}


def conservative_prediction(raw: str, labels: list[str]) -> tuple[str, str]:
    exact = predicted_class(raw, labels)
    if exact != INVALID:
        return exact, "exact"
    # Match entire first response line only, with modest answer-prefix handling.
    # Reuse the scorer's extraction for any direct class matches; aliases are not
    # permitted within explanations or multi-diagnosis lists.
    import re
    t = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    t = re.sub(r"^(?:diagnosis|answer)\s*:\s*", "", t, flags=re.I)
    t = re.sub(r"^(?:the most likely diagnosis is|most likely diagnosis is)\s+", "", t, flags=re.I)
    key = normalized(t)
    lookup = {normalized(k): v for k, v in ALIASES.items()}
    label = lookup.get(key)
    if label in labels:
        return label, "alias"
    return INVALID, "invalid"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", choices=("vanilla", "1000", "3000", "5000", "all"), required=True)
    p.add_argument("--split", choices=("validate", "test"), default="validate")
    p.add_argument("--data", type=Path)
    p.add_argument("--output-root", type=Path, default=Path("runs/gemma4_evaluation"))
    p.add_argument("--train-data", type=Path, default=Path("data/generated_clinical_v2/train.jsonl"))
    args = p.parse_args()
    source = args.data or Path(f"data/generated_clinical_v2/{args.split}.jsonl")
    rows = read_rows(source)
    train_rows = read_rows(args.train_data)
    labels = sorted({r["target"]["diagnosis"] for r in rows + train_rows})
    expected = {r["patient_id"]: r["target"]["diagnosis"] for r in rows}
    models = ("vanilla", "1000", "3000", "5000") if args.model_id == "all" else (args.model_id,)

    for model_id in models:
        directory = args.output_root / args.split / ("vanilla" if model_id == "vanilla" else f"stage-{model_id}")
        path = directory / "predictions.jsonl"
        if not path.exists():
            print(f"Skipping {model_id}: no predictions at {path}")
            continue
        cache = existing_predictions(path, expected)
        if len(cache) != len(rows):
            print(f"Skipping {model_id}: incomplete cache ({len(cache)}/{len(rows)})")
            continue
        normalized_predictions = {}
        changes = []
        by_method = Counter()
        for row in rows:
            pid = row["patient_id"]
            original = cache[pid]
            prediction, method = conservative_prediction(original["raw_response"], labels)
            by_method[method] += 1
            normalized_predictions[pid] = {**original, "prediction": prediction}
            if prediction != original["prediction"]:
                changes.append({"patient_id": pid, "truth": original["truth"],
                                "raw_response": original["raw_response"],
                                "old_prediction": original["prediction"],
                                "new_prediction": prediction,
                                "changed_to_correct": prediction == original["truth"]})
        dest = directory / "normalized"
        report(rows, normalized_predictions, labels, dest, model_id + " (conservative normalization)")
        with (dest / "normalization_changes.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=("patient_id", "truth", "raw_response", "old_prediction", "new_prediction", "changed_to_correct"))
            writer.writeheader()
            writer.writerows(changes)
        write_json_atomic(dest / "normalization_summary.json", {
            "model_id": model_id, "method_counts": dict(by_method),
            "changed_predictions": len(changes),
            "changed_to_correct": sum(c["changed_to_correct"] for c in changes),
            "note": "Only explicit, conservative diagnosis-name equivalents; related diagnoses remain invalid. Original predictions untouched.",
            "aliases": ALIASES,
        })
        print(f"{model_id}: {len(changes)} changed; {sum(c['changed_to_correct'] for c in changes)} newly correct; see {dest}")


if __name__ == "__main__":
    main()
