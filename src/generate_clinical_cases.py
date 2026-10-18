"""Generate third-person clinician-style summaries from preprocessed DDXPlus.

Run from project root:
  uv run python src/generate_clinical_cases.py --split train --max-records 20
  uv run python src/generate_clinical_cases.py --split train --max-records 100 --resume

Requires torch, transformers, accelerate and access to google/gemma-4-12B-it.
The model NEVER receives the target diagnosis or differential diagnoses.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

LOG = logging.getLogger("feddx.clinical_cases")
DEFAULT_MODEL = "google/gemma-4-12B-it"
PROMPT_VERSION = "clinician-v1"

SYSTEM_PROMPT = """You write concise THIRD-PERSON clinical case summaries from decoded
questionnaire evidence for a synthetic medical research dataset.

Rules:
- Use ONLY supplied information. Do not infer a diagnosis, differential, prognosis,
  examination result, treatment, duration, or severity not explicitly supplied.
- Do not mention the diagnosis or guess the likely condition.
- Express a YES answer to an 'A or B' question as 'A or B' (never assert both,
  and never select one without support). Keep other uncertainties explicit.
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
        "Preserve disjunctions such as 'or', avoid unsupported specificity, and "
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


def generate(model, processor, public_input: dict, args) -> dict[str, str]:
    encoded = processor.apply_chat_template(
        make_messages(public_input), tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        kwargs["temperature"] = args.temperature
    with torch.inference_mode():
        result = model.generate(**encoded, **kwargs)
    completion = processor.decode(result[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True)
    del encoded, result
    return parse_output(completion)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validate", "test"], default="train")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated_clinical"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-records", type=int, default=20, help="Maximum NEW cases this run; 0 for all")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N input records")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_records < 0 or args.offset < 0 or args.retries < 0 or args.max_new_tokens < 64:
        parser.error("Invalid limits: max-records/offset/retries must be nonnegative; max-new-tokens >= 64")
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
    if not torch.cuda.is_available():
        parser.error("CUDA GPU unavailable")
    LOG.info("Loading %s", args.model)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    successes = failures = skipped = 0
    with source.open(encoding="utf-8") as inp, output.open("a" if args.resume else "w", encoding="utf-8") as out:
        for index, line in enumerate(inp):
            if index < args.offset or not line.strip():
                continue
            if args.max_records and successes >= args.max_records:
                break
            item = json.loads(line)
            generation_input = item["generation_input"]
            patient_id = generation_input["patient_id"]
            if patient_id in done:
                skipped += 1
                continue
            public_input, flags = make_public_input(generation_input)
            sections = None
            for attempt in range(args.retries + 1):
                try:
                    sections = generate(model, processor, public_input, args)
                    break
                except ValueError as exc:
                    LOG.warning("%s attempt %d rejected: %s", patient_id, attempt + 1, exc)
            if sections is None:
                failures += 1
                continue
            record = {
                "patient_id": patient_id,
                "patient_features": generation_input.get("patient_features", {}),
                "clinical_summary": format_summary(generation_input.get("patient_features", {}), sections),
                "presenting_findings": sections["presenting_findings"],
                "relevant_history": sections["relevant_history"],
                "clinical_evidence": generation_input.get("clinical_evidence", []),
                "target": item["target"],
                "generation_metadata": {
                    "model": args.model, "prompt_version": PROMPT_VERSION,
                    "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
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
    LOG.info("Finished: generated=%d skipped=%d failed=%d file=%s", successes, skipped, failures, output)


if __name__ == "__main__":
    main()
