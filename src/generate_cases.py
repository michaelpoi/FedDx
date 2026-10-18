"""Generate first-person patient narratives from preprocessed DDXPlus JSONL.

Only `generation_input` is sent to the model. The diagnosis stays hidden until
we write the completed output record. Generation runs sequentially to keep GPU
memory predictable on a single L40S. Outputs are append-only and resumable.

Example:
    uv run python src/generate_cases.py --split train --max-records 100
    uv run python src/generate_cases.py --split train --max-records 1000 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

LOG = logging.getLogger("feddx.generate")
MODEL_DEFAULT = "google/gemma-4-12B-it"
# Numeric questionnaire scales without verified meanings should not become prose.
# The original evidence is kept in the output for downstream research.
OMIT_CODES = {"E_58", "E_59"}  # pain-location precision and onset-speed scales
OMIT_NUMERIC_CODES = {"E_56", "E_132", "E_134"}  # uninterpreted ordinal scales

# Questions with "or" do not establish that both statements are true.
# These are excluded from generated patient language, but retained in structured data.
AMBIGUOUS_COMBINED_CODES = {"E_49", "E_104", "E_105", "E_124", "E_175"}

SYSTEM_PROMPT = """Write a realistic FIRST-PERSON symptom description as if a patient
were describing their situation to a clinician. This is a synthetic research record.

Strict rules:
- Use ONLY provided facts. Do not invent diagnoses, symptoms, treatments, durations,
  severity, timing, or test results.
- Never guess the meaning of a numeric clinical scale.
- Never turn an 'A OR B' questionnaire answer into 'A AND B'.
- Preserve explicit negations; missing information is NOT a negative finding.
- Describe medical history as background, not new symptoms.
- Use clear everyday language (e.g. 'headache' rather than 'cephalgia').
- Combine closely related details naturally. Do not write an inventory of questionnaire answers.
- Use age and sex only as specified. Avoid 'presenting with' and 'location precision'.
- No diagnosis guesses, advice, or test recommendations.
- Return ONLY valid JSON in this schema: {"narrative": "..."}
"""


def prepare_generation_input(generation_input: dict) -> tuple[dict, list[str]]:
    """Build diagnosis-free, conservative input and collect review flags.

    Some original evidence is excluded from NARRATIVE generation, not from
    the saved source clinical_evidence in the output JSONL.
    """
    facts = []
    flags = []
    features = generation_input.get("patient_features", {})
    sex = str(features.get("sex", "")).upper()
    for evidence in generation_input.get("clinical_evidence", []):
        code = evidence.get("evidence_code", "")
        question = evidence.get("question", "")
        answer = evidence.get("answer")
        if code in OMIT_CODES or code in OMIT_NUMERIC_CODES:
            flags.append(f"excluded_uninterpreted_numeric:{code}")
            continue
        if code in AMBIGUOUS_COMBINED_CODES:
            flags.append(f"excluded_ambiguous_combined_question:{code}")
            continue
        if sex == "M" and "labia majora" in str(answer).lower():
            flags.append("review_anatomy_demographics:labia_majora")
        if answer is None or answer == "":
            continue
        facts.append({"question": question, "answer": answer,
                      "type": evidence.get("type")})
    return {"patient_features": features, "clinical_evidence": facts}, sorted(set(flags))


def make_messages(generation_input: dict) -> list[dict]:
    public, _ = prepare_generation_input(generation_input)
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": (
            "Rewrite these facts in natural first-person patient language. "
            "Include the major symptoms and relevant history without speculating. "
            "Make it concise, typically 50–150 words; use fewer words for sparse cases. "
            "Do not describe rating scales or numerical precision. "
            "Do not present ambiguous alternatives as jointly confirmed.\n"
            + json.dumps(public, ensure_ascii=False)
        )}]},
    ]


def narrative_quality_flags(narrative: str) -> list[str]:
    """Surface obvious style failures; NOT a medical fact checker."""
    patterns = {
        "scale_language": r"\b(?:location precision|precision of (?:the )?location|onset speed|speed of onset|pain appeared at a speed|rate the precision)\b",
        "clinician_style": r"\b(?:presenting with|patient presents|reports a history of)\b",
    }
    return [name for name, pattern in patterns.items()
            if re.search(pattern, narrative, re.IGNORECASE)]


def parse_response(text: str, min_words: int) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Model did not return valid JSON") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("narrative"), str):
        raise ValueError("Model output must contain a string 'narrative'")
    narrative = obj["narrative"].strip()
    if len(narrative.split()) < min_words:
        raise ValueError(f"Narrative shorter than {min_words} words")
    if len(narrative) > 12_000:
        raise ValueError("Narrative unreasonably long")
    return narrative


def load_completed_ids(path: Path) -> set[str]:
    completed = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as inp:
        for line_number, line in enumerate(inp, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                completed.add(obj["patient_id"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                    f"Cannot safely resume: malformed output line {line_number} in {path}. "
                    "Repair/truncate the line or use a new output file."
                ) from exc
    return completed


def generate_one(model, processor, generation_input, args) -> str:
    messages = make_messages(generation_input)
    encoded = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    encoded = encoded.to(model.device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
    with torch.inference_mode():
        result = model.generate(**encoded, **generation_kwargs)
    result_tokens = result[0, encoded["input_ids"].shape[-1]:]
    text = processor.decode(result_tokens, skip_special_tokens=True)
    del encoded, result, result_tokens
    narrative = parse_response(text, args.min_words)
    flags = narrative_quality_flags(narrative)
    if "scale_language" in flags:
        raise ValueError("Generated narrative contains uninterpreted scale language")
    return narrative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validate", "test"], default="train")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--max-records", type=int, default=100,
                        help="Maximum NEW records to generate this run (default 100); 0 means no limit")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip this many source lines before processing")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Maximum generated tokens per case (default 1024)")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--min-words", type=int, default=15)
    parser.add_argument("--retries", type=int, default=2,
                        help="Additional attempts if JSON / basic structural validation fails")
    parser.add_argument("--resume", action="store_true",
                        help="Append, skipping patient IDs already in output")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing output (cannot combine with --resume)")
    args = parser.parse_args()
    if args.max_records < 0 or args.offset < 0 or args.max_new_tokens < 32 or args.min_words < 1 or args.retries < 0:
        parser.error("Invalid nonnegative limits / max-new-tokens / min-words")
    if args.temperature < 0 or args.temperature > 2:
        parser.error("--temperature must be between 0 and 2")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    source = args.input_dir / f"{args.split}.jsonl"
    destination = args.output_dir / f"{args.split}.jsonl"
    if not source.is_file():
        parser.error(f"Input not found: {source}. Run preprocess_ddxplus.py first.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.resume and not args.overwrite:
        parser.error(f"Output already exists: {destination}; use --resume or --overwrite")
    done = load_completed_ids(destination) if args.resume else set()
    LOG.info("Loading processor: %s", args.model)
    processor = AutoProcessor.from_pretrained(args.model)
    LOG.info("Loading model on GPU (this may consume substantial VRAM)")
    if not torch.cuda.is_available():
        parser.error("No CUDA GPU detected; refusing CPU generation for large-scale dataset processing")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    written = 0
    failed = 0
    skipped = 0
    mode = "a" if args.resume else "w"
    with source.open(encoding="utf-8") as inp, destination.open(mode, encoding="utf-8") as out:
        for line_index, line in enumerate(inp):
            if line_index < args.offset or not line.strip():
                continue
            if args.max_records and written >= args.max_records:
                break
            record = json.loads(line)
            generation_input = record["generation_input"]
            patient_id = generation_input["patient_id"]
            if patient_id in done:
                skipped += 1
                continue
            narrative = None
            error = None
            for _ in range(args.retries + 1):
                try:
                    narrative = generate_one(model, processor, generation_input, args)
                    break
                except ValueError as exc:
                    error = str(exc)
            if narrative is None:
                failed += 1
                LOG.error("Failed %s after %s attempts: %s", patient_id, args.retries + 1, error)
                # Failed cases are deliberately not marked done; can retry on resume.
                continue
            _, source_flags = prepare_generation_input(generation_input)
            output = {
                "patient_id": patient_id,
                "patient_features": generation_input.get("patient_features"),
                "narrative": narrative,
                "clinical_evidence": generation_input.get("clinical_evidence"),
                "target": record["target"],
                "generation_metadata": {
                    "model": args.model,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "validation": "basic_format_and_scale_language_only_not_clinically_verified",
                    "quality_flags": source_flags + narrative_quality_flags(narrative),
                },
            }
            out.write(json.dumps(output, ensure_ascii=False) + "\n")
            out.flush()  # a completed case survives interruption
            done.add(patient_id)
            written += 1
            if written % 10 == 0:
                LOG.info("Generated %d new cases; %d skipped; %d failures", written, skipped, failed)
    LOG.info("Finished: %d new records, %d skipped, %d failures. Saved to %s", written, skipped, failed, destination)
    if failed:
        LOG.warning("Failed cases remain absent from output; rerun with --resume to retry")


if __name__ == "__main__":
    main()
