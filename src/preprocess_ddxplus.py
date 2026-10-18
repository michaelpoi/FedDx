"""Decode DDXPlus patient CSVs into JSONL records for later LLM case generation.

Only pass each record's `generation_input` to a narrative-generating LLM.
The `target` must remain hidden from the generator.
"""

import argparse
import ast
import csv
import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

LOG = logging.getLogger("feddx.preprocess")
SPLITS = ("train", "validate", "test")
DELIMITER = "_@_"


def parse_literal(value, field, row_number):
    if value is None or not str(value).strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Row {row_number}: invalid {field}: {value[:120]!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Row {row_number}: {field} must contain a list")
    return parsed


def meaning(meta, code):
    item = meta.get("value_meaning", {}).get(code)
    if isinstance(item, dict):
        return item.get("en") or item.get("fr")
    return item


def decode_evidences(tokens, metadata, strict=True):
    """Decode binary/categorical/multi-choice data without inventing negatives.

    Group multichoice values under one question. Unknown codes raise by default.
    """
    grouped = {}
    for token in tokens:
        if not isinstance(token, str):
            raise ValueError(f"Non-string evidence: {token!r}")
        code, separator, value_code = token.partition(DELIMITER)
        meta = metadata.get(code)
        if meta is None:
            if strict:
                raise ValueError(f"Unknown evidence code: {code}")
            LOG.warning("Skipping unknown evidence %s", code)
            continue
        kind = meta.get("data_type")
        question = meta.get("question_en")
        if not question:
            raise ValueError(f"Missing English question for {code}")
        if kind not in ("B", "C", "M"):
            raise ValueError(f"Unknown data_type {kind!r} for {code}")

        if kind == "B":
            if separator:
                raise ValueError(f"Unexpected value for binary evidence: {token}")
            answer = "yes"
        else:
            if not separator:
                raise ValueError(f"Missing value for {kind} evidence: {token}")
            allowed = {str(v) for v in meta.get("possible-values", [])}
            if value_code == str(meta.get("default_value")):
                # A default code does not establish a clinical finding.
                continue
            if value_code in allowed:
                answer = meaning(meta, value_code)
                if answer is None:
                    # Some releases store readable numeric values directly.
                    answer = value_code
            elif re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value_code):
                # DDXPlus includes scalar answers like E_56_@_4 that are
                # not always enumerated as possible-values in metadata.
                if not math.isfinite(float(value_code)):
                    raise ValueError(f"Non-finite numeric value {value_code!r} for {code}")
                answer = value_code
            else:
                raise ValueError(f"Unexpected value {value_code!r} for {code}")
            if not str(answer).strip() or str(answer).strip().lower() in ("na", "n/a", "none", "unknown"):
                continue

        if code not in grouped:
            grouped[code] = {
                "evidence_code": code,
                "question": question,
                "answer": [] if kind == "M" else answer,
                "type": "medical_history" if meta.get("is_antecedent") else "symptom",
                "data_type": kind,
                "parent_evidence_code": meta.get("code_question"),
            }
        elif kind != "M":
            if grouped[code]["answer"] != answer:
                raise ValueError(f"Conflicting categorical answers for {code}")
        if kind == "M" and answer not in grouped[code]["answer"]:
            grouped[code]["answer"].append(answer)
    return list(grouped.values())


def decode_initial(token, metadata):
    if not token:
        return None
    items = decode_evidences([token], metadata)
    return items[0] if items else None


def normalize_differential(raw, row_number):
    candidates = parse_literal(raw, "DIFFERENTIAL_DIAGNOSIS", row_number)
    result = []
    for item in candidates:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Row {row_number}: malformed differential entry {item!r}")
        result.append({"diagnosis": str(item[0]), "score": float(item[1])})
    return result


def preprocess_row(row, split, index, metadata):
    row_number = index + 2  # 1-based CSV row including header
    evidence_tokens = parse_literal(row.get("EVIDENCES"), "EVIDENCES", row_number)
    evidence = decode_evidences(evidence_tokens, metadata)
    raw_initial = (row.get("INITIAL_EVIDENCE") or "").strip()
    if raw_initial and raw_initial not in evidence_tokens:
        raise ValueError(f"Row {row_number}: INITIAL_EVIDENCE is not in EVIDENCES")
    age_raw = (row.get("AGE") or "").strip()
    features = {
        "age": int(float(age_raw)) if age_raw else None,
        "sex": (row.get("SEX") or "").strip() or None,
    }
    record_id = f"{split}_{index:07d}"
    initial = decode_initial(raw_initial, metadata)
    # Deliberately avoid including PATHOLOGY or DIFFERENTIAL_DIAGNOSIS here.
    generation_input = {
        "patient_id": record_id,
        "patient_features": features,
        "clinical_evidence": evidence,
        "initial_evidence": initial,
    }
    target = {
        "diagnosis": (row.get("PATHOLOGY") or "").strip(),
        "differential_diagnoses": normalize_differential(
            row.get("DIFFERENTIAL_DIAGNOSIS"), row_number
        ),
    }
    if not target["diagnosis"]:
        raise ValueError(f"Row {row_number}: empty pathology label")
    return {"generation_input": generation_input, "target": target}


def process_split(split, input_dir, output_dir, metadata, max_rows):
    source = input_dir / f"{split}.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Missing {source}")
    destination = output_dir / f"{split}.jsonl"
    counts = Counter()
    processed = 0
    with source.open("r", encoding="utf-8-sig", newline="") as handle, \
            destination.open("w", encoding="utf-8") as out:
        reader = csv.DictReader(handle)
        required = {"AGE", "SEX", "EVIDENCES", "INITIAL_EVIDENCE", "PATHOLOGY"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source} missing columns: {sorted(missing)}")
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            try:
                record = preprocess_row(row, split, index, metadata)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Failed processing {source} CSV row {index + 2}: {exc}") from exc
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[record["target"]["diagnosis"]] += 1
            processed += 1
            if processed % 100_000 == 0:
                LOG.info("%s: processed %s records", split, processed)
    LOG.info("Saved %s records to %s", processed, destination)
    return {"rows": processed, "diagnoses": dict(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/ddxplus"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--max-rows", type=int, default=None, help="Rows per split (for testing)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    metadata_file = args.input_dir / "release_evidences.json"
    if not metadata_file.is_file():
        parser.error(f"Evidence metadata not found: {metadata_file}")
    with metadata_file.open(encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        parser.error("Expected evidence metadata as a dictionary keyed by evidence code")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in args.splits:
        report[split] = process_split(split, args.input_dir, args.output_dir, metadata, args.max_rows)
    (args.output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOG.info("Wrote preprocessing report to %s", args.output_dir / "preprocessing_report.json")


if __name__ == "__main__":
    main()
