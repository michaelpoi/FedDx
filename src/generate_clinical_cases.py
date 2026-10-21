"""Generate third-person clinician-style summaries from preprocessed DDXPlus.

Run from project root:
  uv run python src/generate_clinical_cases.py --split train --max-records 20
  uv run python src/generate_clinical_cases.py --split train --max-records 100 --resume

Requires torch, transformers, accelerate and access to google/gemma-4-12B-it.
The model NEVER receives the target diagnosis or differential diagnoses.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import math
import os
import random
from collections import Counter
import json
import logging
import re
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

LOG = logging.getLogger("feddx.clinical_cases")
DEFAULT_MODEL = "google/gemma-4-12B-it"
PROMPT_VERSION = "clinician-v2"

SYSTEM_PROMPT = """You write concise THIRD-PERSON clinical case summaries from decoded
questionnaire evidence for a synthetic medical research dataset.

Rules:
- Use ONLY supplied information. Do not infer a diagnosis, differential, prognosis,
  examination result, treatment, duration, or severity not explicitly supplied.
- Do not mention the diagnosis or guess the likely condition.
- A YES to a question with alternatives (A OR B) establishes AT LEAST ONE,
  NOT both and NOT a specific alternative. Never rewrite A OR B as A AND B,
  A WITH B, or as two confirmed findings. Preserve the original OR uncertainty.
  Example: 'asthma OR previous bronchodilator use' must NOT become
  'asthma (with previous bronchodilator use)'.
- A multi-select answer listing selected items may establish each selected item;
  do not mistake a list of selected answers for alternatives in a question.
- Do not invent additional history, time courses, or relationships.
- Preserve negative findings if explicitly given; missing facts are unknown.
- Clinical-history items describe history, not new or current symptoms.
- Use precise, readable clinician language. Avoid verbatim questionnaire questions,
  unnecessary anatomical lists, and unsupported assumptions.
- A numeric questionnaire response with no validated scale semantics is not supplied.
- If the evidence appears unusual or inconsistent, faithfully report it;
  do not silently repair or fabricate it.
- Output ONLY a JSON object with two STRING fields:
  {"presenting_findings": "...", "relevant_history": "..."}
- Use an empty string when no evidence supports a section. Do not include
  demographic details in these fields; those are added deterministically.
"""

NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
SCALE_LANGUAGE = re.compile(
    r"\b(?:location precision|precision of (?:the )?location|onset speed|speed of onset|"
    r"pain appeared at a speed|rate the precision)\b", re.IGNORECASE
)


def make_public_input(generation_input: dict) -> tuple[dict, list[str]]:
    """No label exposure. Remove only uninterpretable numeric questionnaire values.

    We preserve every original finding in the output, even if not passed to the LLM.
    Other categorical values (including Y/N) are not treated as numeric scales.
    """
    visible = []
    flags: list[str] = []
    for evidence in generation_input.get("clinical_evidence", []):
        question = evidence.get("question", "")
        answer = evidence.get("answer")
        code = evidence.get("evidence_code", "")
        if answer is None or answer == "" or answer == []:
            continue
        if evidence.get("data_type") == "C" and isinstance(answer, str) and NUMERIC.fullmatch(answer.strip()):
            flags.append(f"excluded_uninterpreted_numeric:{code}")
            continue
        visible.append({
            "evidence_code": code,
            "question": question,
            "answer": answer,
            "category": evidence.get("type", "symptom"),
        })
    return {
        "patient_features": generation_input.get("patient_features", {}),
        "evidence": visible,
    }, flags


def make_messages(public_input: dict) -> list[dict]:
    text = (
        "Write a clinician-style summary of the following patient evidence. "
        "Present current symptoms and relevant medical history separately. "
        "A positive OR question confirms at least one alternative, never both. "
        "Preserve uncertainty and avoid unsupported specificity; "
        "do not write a diagnosis. Keep each section compact and complete.\n\n"
        + json.dumps(public_input, ensure_ascii=False)
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]


def parse_output(raw: str) -> dict[str, str]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Model output is not valid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("Model output must be an object")
    sections = {}
    for key in ("presenting_findings", "relevant_history"):
        if not isinstance(result.get(key), str):
            raise ValueError(f"Missing string field: {key}")
        sections[key] = result[key].strip()
        if len(sections[key]) > 6000:
            raise ValueError(f"Field too long: {key}")
    if not sections["presenting_findings"] and not sections["relevant_history"]:
        raise ValueError("Model returned no clinical content")
    if SCALE_LANGUAGE.search(" ".join(sections.values())):
        raise ValueError("Uninterpreted questionnaire scale leaked into output")
    return sections


def demographic_phrase(features: dict) -> str:
    age = features.get("age")
    sex_code = str(features.get("sex", "")).strip().upper()
    sex = {"M": "male", "F": "female"}.get(sex_code, "patient")
    if age is not None and str(age).strip():
        return f"{age}-year-old {sex}" if sex != "patient" else f"{age}-year-old patient"
    return sex if sex != "patient" else "patient"


def format_summary(features: dict, sections: dict[str, str]) -> str:
    demo = demographic_phrase(features)
    if sections["presenting_findings"]:
        summary = f"Patient: {demo}.\nPresenting findings: {sections['presenting_findings']}"
    else:
        summary = f"Patient: {demo}.\nPresenting findings: Not documented."
    if sections["relevant_history"]:
        summary += f"\nRelevant history: {sections['relevant_history']}"
    return summary


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for n, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["patient_id"])
            except (KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid existing output line {n} in {path}; refusing unsafe resume") from exc
    return ids



def source_records(source: Path):
    """Iterate input in stable order with minimal memory overhead."""
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def source_signature(source: Path) -> dict:
    stat = source.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_selection_plan(source: Path, output: Path, args, done: set[str]) -> dict:
    """Select additional unique source IDs; never change a saved plan on resume.

    natural: uniformly sample records without replacement.
    hybrid: weight each record by 1/sqrt(source_class_frequency), increasing
            representation of rare diagnoses without enforcing equality.
    """
    plan_path = args.output_dir / f"{args.split}.selection.json"
    signature = source_signature(source)
    if plan_path.exists() and not args.overwrite:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        expected = {"strategy": args.sampling, "seed": args.seed,
                    "target_total": args.target_total, "source_signature": signature}
        for name, value in expected.items():
            if plan.get(name) != value:
                raise ValueError(f"Saved sampling plan disagrees on {name}; use the original settings or a new output directory")
        selected = plan["selected_ids"]
        if len(selected) != len(set(selected)):
            raise ValueError("Saved sampling plan contains duplicate patient IDs")
        return plan

    if args.resume and plan_path.exists():
        raise ValueError("Cannot safely replace an existing selection plan")
    needed = args.target_total - len(done)
    if needed <= 0:
        selected = []
        counts = Counter()
    else:
        counts = Counter()
        candidate_total = 0
        seen_ids = set(done)
        for item in source_records(source):
            pid = item["generation_input"]["patient_id"]
            if pid in seen_ids:
                if pid not in done:
                    raise ValueError(f"Duplicate source patient ID: {pid}")
                continue
            seen_ids.add(pid)
            label = item["target"]["diagnosis"]
            counts[label] += 1
            candidate_total += 1
        if needed > candidate_total:
            raise ValueError(f"Need {needed} more cases but only {candidate_total} source cases remain")
        rng = random.Random(args.seed)
        # Weighted reservoir sampling via random exponential keys.
        # Keep the smallest 'needed' keys in a bounded max heap.
        heap = []
        for item in source_records(source):
            pid = item["generation_input"]["patient_id"]
            if pid in done:
                continue
            label = item["target"]["diagnosis"]
            weight = counts[label] ** (-0.5 if args.sampling == "hybrid" else 0.0)
            key = -math.log(max(rng.random(), 1e-300)) / weight
            entry = (-key, pid)
            if len(heap) < needed:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
        selected = [pid for _, pid in heap]
    plan = {
        "version": 1, "strategy": args.sampling, "seed": args.seed,
        "target_total": args.target_total, "source_signature": signature,
        "initial_output_count": len(done), "selected_ids": sorted(selected),
        "candidate_diagnosis_counts": dict(counts),
    }
    temp = plan_path.with_suffix(plan_path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, plan_path)
    LOG.info("Saved %s sampling plan: %d additional patients, %d existing", args.sampling, len(selected), len(done))
    return plan


def generate_batch(model, processor, public_inputs: list[dict], args) -> list[dict[str, str] | Exception]:
    """Generate a padded batch. Parse each output independently for targeted retries."""
    conversations = [make_messages(public_input) for public_input in public_inputs]
    encoded = processor.apply_chat_template(
        conversations if len(conversations) > 1 else conversations[0],
        tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", processor_kwargs={"padding": True},
    ).to(model.device)
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": processor.tokenizer.pad_token_id
            if processor.tokenizer.pad_token_id is not None
            else processor.tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        kwargs["temperature"] = args.temperature
    with torch.inference_mode():
        generated = model.generate(**encoded, **kwargs)
    # The output includes the entire padded input, even for shorter prompts.
    prefix_length = encoded["input_ids"].shape[1]
    completions = processor.batch_decode(
        generated[:, prefix_length:], skip_special_tokens=True
    )
    del encoded, generated
    parsed = []
    for completion in completions:
        try:
            parsed.append(parse_output(completion))
        except ValueError as exc:
            parsed.append(exc)
    return parsed


def generate_with_retries(model, processor, public_input: dict, args, patient_id: str):
    for attempt in range(args.retries + 1):
        try:
            parsed = generate_batch(model, processor, [public_input], args)[0]
            if not isinstance(parsed, Exception):
                return parsed
            raise parsed
        except ValueError as exc:
            LOG.warning("%s attempt %d rejected: %s", patient_id, attempt + 1, exc)
    return None


def generate_pending(model, processor, pending: list[tuple], args):
    """Batch first, then retry only invalid cases individually.

    If the optional batch exhausts VRAM, retry each case singly and keep going.
    """
    if len(pending) == 1:
        item, generation_input, public_input, flags = pending[0]
        return [(pending[0], generate_with_retries(model, processor, public_input, args,
                                                   generation_input["patient_id"]))]
    try:
        outcomes = generate_batch(model, processor, [entry[2] for entry in pending], args)
    except torch.OutOfMemoryError:
        LOG.warning("Batch size %d exhausted GPU memory; falling back to single cases", len(pending))
        torch.cuda.empty_cache()
        outcomes = [ValueError("batch GPU memory exhausted") for _ in pending]
    results = []
    for entry, outcome in zip(pending, outcomes):
        if isinstance(outcome, Exception):
            LOG.warning("%s batched response rejected: %s; retrying alone", entry[1]["patient_id"], outcome)
            outcome = generate_with_retries(model, processor, entry[2], args, entry[1]["patient_id"])
        results.append((entry, outcome))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validate", "test"], default="train")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated_clinical"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-records", type=int, default=20, help="Maximum NEW cases this run; 0 for all")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N input records (sequential mode only)")
    parser.add_argument("--sampling", choices=["sequential", "natural", "hybrid"], default="sequential",
                        help="natural: seeded uniform sample; hybrid: gently boost rare diagnoses")
    parser.add_argument("--target-total", type=int, default=None,
                        help="Total desired output cases; required for natural/hybrid selection")
    parser.add_argument("--seed", type=int, default=42, help="Stable random sampling seed")
    parser.add_argument("--plan-only", action="store_true",
                        help="Build/reuse selection plan and report counts, without loading Gemma")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Token budget per summary; increase for very complex cases")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Cases per generation batch (try 2 on L40S; 1 is safest)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 enables deterministic greedy decoding (faster)")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_records < 0 or args.offset < 0 or args.retries < 0 or args.max_new_tokens < 64 or args.batch_size < 1:
        parser.error("Invalid limits: max-records/offset/retries must be nonnegative; max-new-tokens >= 64; batch-size >= 1")
    if args.sampling != "sequential" and args.target_total is None:
        parser.error("--target-total is required for natural/hybrid")
    if args.target_total is not None and args.target_total <= 0:
        parser.error("--target-total must be positive")
    if args.sampling != "sequential" and args.offset:
        parser.error("--offset is not supported with natural/hybrid; sampling covers the full source")
    if args.plan_only and args.sampling == "sequential":
        parser.error("--plan-only requires natural or hybrid")
    if not 0 <= args.temperature <= 2:
        parser.error("temperature must be 0–2")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be combined")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    source = args.input_dir / f"{args.split}.jsonl"
    output = args.output_dir / f"{args.split}.jsonl"
    if not source.is_file():
        parser.error(f"Missing {source}; run preprocessing first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if output.exists() and not (args.resume or args.overwrite):
        parser.error(f"{output} already exists; use --resume or --overwrite")
    done = completed_ids(output) if args.resume else set()
    selected_ids = None
    if args.sampling != "sequential":
        try:
            plan = build_selection_plan(source, output, args, done)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        selected_ids = set(plan["selected_ids"])
        LOG.info("Plan: %d selected; %d already generated; %d still selected and missing",
                 len(selected_ids), len(done), len(selected_ids - done))
        if args.plan_only:
            return
        if args.target_total <= len(done):
            LOG.info("Target reached: %d / %d", len(done), args.target_total)
            return
    elif args.target_total is not None and len(done) >= args.target_total:
        LOG.info("Target reached: %d / %d", len(done), args.target_total)
        return
    if not torch.cuda.is_available():
        parser.error("CUDA GPU unavailable")
    LOG.info("Loading %s", args.model)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    # Decoder-only causal models require left padding for batched inference.
    processor.tokenizer.padding_side = "left"
    successes = failures = skipped = 0
    pending = []

    def write_pending(out):
        nonlocal successes, failures
        if not pending:
            return
        for (item, generation_input, public_input, flags), sections in generate_pending(
            model, processor, pending, args
        ):
            if sections is None:
                failures += 1
                continue
            patient_id = generation_input["patient_id"]
            features = generation_input.get("patient_features", {})
            record = {
                "patient_id": patient_id,
                "patient_features": features,
                "clinical_summary": format_summary(features, sections),
                "presenting_findings": sections["presenting_findings"],
                "relevant_history": sections["relevant_history"],
                "clinical_evidence": generation_input.get("clinical_evidence", []),
                "target": item["target"],
                "generation_metadata": {
                    "model": args.model, "prompt_version": PROMPT_VERSION,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "batch_size": args.batch_size,
                    "quality_flags": flags,
                    "validation": "json_structure_and_scale_language_only_not_clinically_verified",
                },
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            done.add(patient_id)
            successes += 1
            if successes % 10 == 0:
                LOG.info("Generated=%d, skipped=%d, failed=%d", successes, skipped, failures)
        pending.clear()

    with source.open(encoding="utf-8") as inp, output.open(
        "a" if args.resume else "w", encoding="utf-8"
    ) as out:
        for index, line in enumerate(inp):
            if index < args.offset or not line.strip():
                continue
            if args.max_records and successes >= args.max_records:
                break
            if args.target_total is not None and len(done) >= args.target_total:
                break
            item = json.loads(line)
            generation_input = item["generation_input"]
            patient_id = generation_input["patient_id"]
            if patient_id in done or (selected_ids is not None and patient_id not in selected_ids):
                skipped += 1
                continue
            public_input, flags = make_public_input(generation_input)
            pending.append((item, generation_input, public_input, flags))
            remaining = args.max_records - successes if args.max_records else args.batch_size
            if args.target_total is not None:
                remaining = min(remaining, args.target_total - len(done))
            if len(pending) >= min(args.batch_size, max(1, remaining)):
                write_pending(out)
        write_pending(out)
    LOG.info("Finished: generated=%d skipped=%d failed=%d total=%d file=%s",
             successes, skipped, failures, len(done), output)
    if selected_ids is not None and args.target_total and len(done) < args.target_total:
        LOG.warning("Target not yet reached (%d/%d). Run again with --resume and the same plan settings.",
                    len(done), args.target_total)


if __name__ == "__main__":
    main()
