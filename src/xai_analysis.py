#!/usr/bin/env python3
"""FedDx explainability analysis: contrastive SHAP, plausibility, global explanations.

Three analyses on top of the fine-tuned Gemma 4 diagnosis model:

1. CONTRASTIVE SHAP. Besides the usual SHAP of log P(prediction), explains the
   margin log P(prediction) - log P(rival). The rival is the true label when the
   model is wrong, otherwise the runner-up diagnosis. Answers "why X and not Y?".
   A cheap contrastive deletion check reports whether removing the top-k
   contrastive clauses actually reverses the margin (vs. removing k random ones).

2. PLAUSIBILITY. Each summary clause is aligned back to the patient's DDXPlus
   evidence codes. A clause is "on-list" when its evidence belongs to the
   predicted disease in release_conditions.json (symptoms or antecedents), and
   "discriminative" when it belongs to the prediction but NOT to the rival.
   We compare top-SHAP clauses against the case's own base rate.

3. GLOBAL EXPLANATIONS. Aggregates per-case attributions per predicted
   diagnosis: which evidence codes push each diagnosis, and how much attribution
   comes from demographics vs presenting findings vs history (shortcut screen).

GPU stage (SHAP) and report stage (alignment, plausibility, global, figures) are
separate: --reports-only re-runs alignment with new settings without a GPU.

Run from the FedDx project root:
  uv run python src/xai_analysis.py --model-id 5000 --split test --per-class 4 \
      --adapter-root runs/gemma4_qlora_independent \
      --predictions runs/gemma4_evaluation_test/test/stage-5000/predictions.jsonl \
      --all-errors
  uv run python src/xai_analysis.py ... --reports-only --aligner embed

Scoring uses the TRAINING-matched chat rendering (completed assistant turn),
the same format as train_gemma4_qlora.py, feddx_shap.py and the open-ended
evaluator. SHAP values describe model behavior, not clinical truth or causality.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluate_gemma4_diagnoses import (
    INVALID, PROMPT, adapter_signature, normalized, read_rows, sha, write_json_atomic,
)
from validate_clinical_cases import exact_diagnosis_mention

FORMAT_VERSION = 1
MARKER = "FEDDX_XAI_TARGET_SLOT_51E7A"
SECTIONS = ("Patient", "Presenting findings", "Relevant history")
DEMOGRAPHICS = "DEMOGRAPHICS"
SECTION_RE = re.compile(r"^\s*(Patient|Presenting findings|Relevant history)\s*:\s*(.*)$", re.I)
NEGATION_RE = re.compile(r"\b(?:no|not|denies|denied|without|absence of|negative for)\b", re.I)

# Palette (dataviz reference, light mode): categorical slots 1-3, text, grid.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_TEXT, C_TEXT2, C_GRID, C_SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# --------------------------------------------------------------------------
# 1. Summary features: section-aware clauses with in-distribution masking
# --------------------------------------------------------------------------

def parse_sections(summary: str) -> dict[str, str]:
    """Split a clinician-v2 summary into its three sections (missing -> '')."""
    sections = {name: "" for name in SECTIONS}
    current = None
    for line in summary.strip().splitlines():
        match = SECTION_RE.match(line)
        if match:
            current = next(s for s in SECTIONS if s.lower() == match.group(1).lower())
            sections[current] = match.group(2).strip()
        elif line.strip():
            current = current or "Presenting findings"
            sections[current] = (sections[current] + " " + line.strip()).strip()
    return sections


def split_clauses(text: str) -> list[str]:
    """Split after . ; , followed by whitespace (keeps decimals like 38.5 intact)."""
    if not text.strip() or text.strip().lower().rstrip(".") == "not documented":
        return []
    return [p.strip() for p in re.split(r"(?<=[.;,])\s+", text.strip()) if p.strip()]


def build_features(summary: str, max_features: int) -> list[dict[str, Any]]:
    """Demographics line is one feature; clauses of the other sections are merged
    (shortest adjacent pair within a section first) until <= max_features."""
    if max_features < 3:
        raise ValueError("max_features must be >= 3")
    sections = parse_sections(summary)
    features: list[dict[str, Any]] = []
    if sections["Patient"]:
        features.append({"section": "Patient", "text": f"Patient: {sections['Patient']}"})
    for section in SECTIONS[1:]:
        features += [{"section": section, "text": c} for c in split_clauses(sections[section])]
    while len(features) > max_features:
        best = None
        for i in range(len(features) - 1):
            a, b = features[i], features[i + 1]
            if a["section"] != b["section"] or a["section"] == "Patient":
                continue
            size = len(a["text"]) + len(b["text"])
            if best is None or size < best[0]:
                best = (size, i)
        if best is None:
            break
        i = best[1]
        features[i] = {"section": features[i]["section"],
                       "text": features[i]["text"] + " " + features[i + 1]["text"]}
        del features[i + 1]
    for index, feature in enumerate(features):
        feature["index"] = index
    if len(features) < 2:
        raise ValueError("Need at least two features for SHAP")
    return features


def _tidy(parts: list[str]) -> str:
    text = " ".join(parts).strip().rstrip(" ,;")
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def render_summary(features: list[dict[str, Any]], mask: tuple[int, ...]) -> str:
    """Rebuild a summary in the exact format of generate_clinical_cases.format_summary.

    Masked demographics -> 'Patient: patient.'; empty findings -> 'Not documented.';
    empty history -> line omitted. Masked inputs therefore stay in-distribution.
    """
    kept = [f for f, keep in zip(features, mask) if keep]
    demo = next((f["text"] for f in kept if f["section"] == "Patient"), "Patient: patient.")
    findings = _tidy([f["text"] for f in kept if f["section"] == "Presenting findings"])
    history = _tidy([f["text"] for f in kept if f["section"] == "Relevant history"])
    out = f"{demo}\nPresenting findings: {findings or 'Not documented.'}"
    return out + (f"\nRelevant history: {history}" if history else "")


# --------------------------------------------------------------------------
# 2. Teacher-forced label scorer (training-matched rendering, batched)
# --------------------------------------------------------------------------

class LabelScorer:
    """log P(label | summary) for complete assistant turns.

    'diagnosis' sums the label tokens only (= feddx_shap.diagnosis_log_likelihood);
    'complete' also includes the end-of-turn tokens supervised in training, so a
    label that is a prefix of another (e.g. 'HIV') is not unfairly favored.
    """

    def __init__(self, model, processor, max_length: int = 2048, batch_size: int = 8):
        self.model, self.processor = model, processor
        self.tokenizer = processor.tokenizer
        self.max_length, self.batch_size = max_length, batch_size
        pad = self.tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else self.tokenizer.eos_token_id
        self.forward_passes = 0
        self._keep_supported = True

    def _render(self, question: str, answer: str) -> str:
        return self.processor.apply_chat_template([
            {"role": "user", "content": [{"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ], tokenize=False, add_generation_prompt=False)

    def encode(self, summary: str, label: str) -> tuple[list[int], list[int], list[int]]:
        question = PROMPT.format(summary=summary.strip())
        marked, full = self._render(question, MARKER), self._render(question, label)
        if marked.count(MARKER) != 1 or marked.replace(MARKER, label) != full:
            raise ValueError("Gemma chat template target alignment failed")
        start = marked.index(MARKER)
        end = start + len(label)
        enc = self.tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
        diag = []
        for i, (a, b) in enumerate(offsets):
            if a < start < b:
                raise ValueError(f"Label {label!r} starts inside a token")
            if b > a and start <= a < end:
                diag.append(i)
        if not diag or diag[0] == 0:
            raise ValueError(f"No label tokens for {label!r}")
        suffix = [i for i in range(diag[0] + 1, len(ids))
                  if i not in diag and (offsets[i][1] <= offsets[i][0] or offsets[i][0] >= end)]
        if len(ids) > self.max_length:
            raise ValueError(f"{len(ids)} tokens exceeds max_length={self.max_length}")
        return ids, diag, suffix

    def _forward(self, batch: list[tuple[list[int], list[int], list[int]]]) -> list[dict[str, float]]:
        import torch
        length = max(len(ids) for ids, _, _ in batch)
        first = min(diag[0] for _, diag, _ in batch)
        keep = length - first + 1  # logits[t-1] predict token t
        input_ids = torch.tensor([ids + [self.pad_id] * (length - len(ids)) for ids, _, _ in batch],
                                 dtype=torch.long, device=self.model.device)
        mask = torch.tensor([[1] * len(ids) + [0] * (length - len(ids)) for ids, _, _ in batch],
                            dtype=torch.long, device=self.model.device)
        with torch.inference_mode():
            logits = None
            if self._keep_supported:
                try:
                    logits = self.model(input_ids=input_ids, attention_mask=mask,
                                        use_cache=False, logits_to_keep=keep).logits
                except TypeError:
                    self._keep_supported = False
            if logits is None:
                logits = self.model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits
            offset = length - logits.shape[1]
            results = []
            for row, (ids, diag, suffix) in enumerate(batch):
                positions = diag + suffix
                rows = logits[row, [p - 1 - offset for p in positions], :].float()
                targets = torch.tensor([ids[p] for p in positions], device=rows.device)
                lp = torch.log_softmax(rows, dim=-1).gather(-1, targets[:, None]).squeeze(-1)
                d = float(lp[:len(diag)].sum().item())
                results.append({"diagnosis": d, "complete": float(lp.sum().item())})
        self.forward_passes += 1
        return results

    def score(self, summary: str, labels: list[str]) -> list[dict[str, float]]:
        encoded = [self.encode(summary, label) for label in labels]
        out: list[dict[str, float]] = []
        for i in range(0, len(encoded), self.batch_size):
            out += self._forward(encoded[i:i + self.batch_size])
        return out


# --------------------------------------------------------------------------
# GPU stage: rival selection, multi-output permutation SHAP, contrastive deletion
# --------------------------------------------------------------------------

def explain_case(scorer: LabelScorer, row: dict, labels: list[str], cached_prediction: str | None,
                 args: argparse.Namespace, self_check: bool) -> dict[str, Any]:
    import numpy as np
    import shap

    started = time.time()
    passes_before = scorer.forward_passes
    truth = row["target"]["diagnosis"]
    features = build_features(row["clinical_summary"], args.max_features)
    n = len(features)
    full_text = render_summary(features, (1,) * n)

    ranking = scorer.score(full_text, labels)
    order = sorted(range(len(labels)), key=lambda k: ranking[k][args.score], reverse=True)
    if cached_prediction is not None:
        prediction, prediction_source = cached_prediction, "cached_open_ended"
    else:
        prediction, prediction_source = labels[order[0]], "teacher_forced_argmax"
    if args.rival_mode == "truth_if_wrong" and prediction != truth:
        rival, rival_reason = truth, "true_label"
    else:
        rival = next(labels[k] for k in order if labels[k] != prediction)
        rival_reason = "runner_up"

    cache: dict[tuple[int, ...], tuple[float, float]] = {}

    def value(mask: tuple[int, ...]) -> tuple[float, float]:
        if mask not in cache:
            s = scorer.score(render_summary(features, mask), [prediction, rival])
            cache[mask] = (s[0][args.score], s[1][args.score])
        return cache[mask]

    def model_fn(matrix):
        return np.asarray([value(tuple(int(v > .5) for v in r)) for r in np.atleast_2d(matrix)],
                          dtype=float)

    explainer = shap.Explainer(model_fn, np.zeros((1, n)), algorithm="permutation", seed=args.seed,
                               feature_names=[f["text"][:60] for f in features])
    result = explainer(np.ones((1, n)), max_evals=(2 * n + 1) * args.permutations,
                       batch_size=1, silent=True)
    values = np.asarray(result.values, dtype=float).reshape(n, 2)
    base = np.asarray(result.base_values, dtype=float).reshape(-1)
    full = np.asarray(value((1,) * n))
    phi_pred, phi_rival = values[:, 0], values[:, 1]
    phi_contrast = phi_pred - phi_rival

    # Contrastive deletion: remove top-k contrastive features vs k random ones.
    rng = random.Random(f"{args.seed}:{row['patient_id']}")
    by_contrast = sorted(range(n), key=lambda i: phi_contrast[i], reverse=True)
    deletion = {}
    for k in range(1, min(args.deletion_k, n - 1) + 1):
        def margin_without(removed):
            mask = tuple(0 if i in removed else 1 for i in range(n))
            p, r = value(mask)
            return p - r
        top = margin_without(set(by_contrast[:k]))
        rand = margin_without(set(rng.sample(range(n), k)))
        deletion[str(k)] = {"top_margin": top, "top_reversed": top < 0,
                            "random_margin": rand, "random_reversed": rand < 0}

    record = {
        "patient_id": row["patient_id"], "truth": truth,
        "prediction": prediction, "prediction_source": prediction_source,
        "correct": prediction == truth, "rival": rival, "rival_reason": rival_reason,
        "teacher_forced_top5": [{"label": labels[k], **ranking[k]} for k in order[:5]],
        "prediction_teacher_forced_rank": 1 + next(i for i, k in enumerate(order) if labels[k] == prediction),
        "features": features,
        "render_matches_original": full_text == row["clinical_summary"].strip(),
        "shap_prediction": phi_pred.tolist(), "shap_rival": phi_rival.tolist(),
        "shap_contrastive": phi_contrast.tolist(),
        "base_values": {"prediction": float(base[0]), "rival": float(base[1])},
        "full_values": {"prediction": float(full[0]), "rival": float(full[1]),
                        "margin": float(full[0] - full[1])},
        "additivity_error": float(np.abs(base + values.sum(axis=0) - full).max()),
        "contrastive_deletion": deletion,
        "coalitions_evaluated": len(cache),
        "forward_passes": scorer.forward_passes - passes_before,
        "seconds": round(time.time() - started, 2),
    }
    if self_check:
        from feddx_shap import diagnosis_log_likelihood
        reference = diagnosis_log_likelihood(scorer.model, scorer.processor, PROMPT, full_text, prediction)
        mine = scorer.score(full_text, [prediction])[0]["diagnosis"]
        record["self_check_abs_diff"] = abs(reference - mine)
        if abs(reference - mine) > 0.1:
            print(f"WARNING: batched scorer differs from feddx_shap by {abs(reference - mine):.3f}", flush=True)
    return record


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for SHAP; use --reports-only for cached results")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_storage=dtype,
        ),
    )
    if args.model_id != "vanilla":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path(args)), is_trainable=False)
    model.eval()
    return model, processor


def adapter_path(args: argparse.Namespace) -> Path:
    return args.adapter_root / f"stage-{args.model_id}" / "adapter_final"


# --------------------------------------------------------------------------
# 3. Report stage: alignment, plausibility, shortcuts, global explanations
# --------------------------------------------------------------------------

STOPWORDS = set("""a an the and or of to in on at by for from with as is are was were be been being
do does did done have has had having you your yours he she it its they them their this that these those
any some ever recently currently now still also very more most other than then there here when where which
who whom what why how if into over under about during before after while patient patients reports reported
report reporting describes described notes noted states stated presents presenting history experienced
experiences experiencing feel feels feeling felt including include includes either both one least at""".split())
SUFFIXES = ("ations", "ation", "ness", "ings", "ing", "ies", "ied", "ed", "es", "s", "ly")


def stems(text: str) -> set[str]:
    out = set()
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in STOPWORDS or len(word) < 3:
            continue
        for suffix in SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)] + ("y" if suffix in ("ies", "ied") else "")
                break
        out.add(word)
    return out


def evidence_text(evidence: dict) -> str:
    answer = evidence.get("answer")
    parts = answer if isinstance(answer, list) else [answer]
    words = [str(p) for p in parts if p is not None and not re.fullmatch(r"[\d.+-]+", str(p).strip())
             and str(p).strip().lower() not in ("yes", "y", "true")]
    return " ".join([str(evidence.get("question") or "")] + words)


class EvidenceAligner:
    """Map a summary clause to the patient's own evidence codes.

    lexical: IDF-weighted overlap coefficient of stemmed content words.
    embed:   cosine similarity with a sentence-transformers model.
    Matches >= threshold and >= relative * best score are kept.
    """

    def __init__(self, rows: list[dict], mode: str, threshold: float | None,
                 relative: float, embed_model: str):
        self.mode, self.relative = mode, relative
        self.threshold = threshold if threshold is not None else (0.35 if mode == "lexical" else 0.5)
        documents = {}
        for row in rows:
            for e in row.get("clinical_evidence", []):
                documents[(e.get("evidence_code"), evidence_text(e))] = stems(evidence_text(e))
        df = Counter(w for s in documents.values() for w in s)
        total = max(len(documents), 1)
        self.idf = {w: math.log((1 + total) / (1 + c)) + 1 for w, c in df.items()}
        self._encoder = None
        if mode == "embed":
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(embed_model)
            self._cache: dict[str, Any] = {}

    def _weight(self, words: set[str]) -> float:
        return sum(self.idf.get(w, max(self.idf.values(), default=1.0)) for w in words)

    def _embed(self, texts: list[str]):
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for t, v in zip(missing, self._encoder.encode(missing, normalize_embeddings=True)):
                self._cache[t] = v
        return [self._cache[t] for t in texts]

    def align(self, clause: str, evidence: list[dict]) -> list[tuple[str, float]]:
        candidates = [(e.get("evidence_code"), evidence_text(e)) for e in evidence if e.get("evidence_code")]
        if not candidates:
            return []
        if self.mode == "embed":
            import numpy as np
            vectors = self._embed([clause] + [t for _, t in candidates])
            scores = [float(np.dot(vectors[0], v)) for v in vectors[1:]]
        else:
            c = stems(clause)
            scores = []
            for _, text in candidates:
                e = stems(text)
                overlap = c & e
                denominator = min(self._weight(c), self._weight(e))
                scores.append(self._weight(overlap) / denominator if overlap and denominator else 0.0)
        best = max(scores)
        if best < self.threshold:
            return []
        keep = {}
        for (code, _), s in zip(candidates, scores):
            if s >= self.threshold and s >= self.relative * best:
                keep[code] = max(keep.get(code, 0.0), s)
        return sorted(keep.items(), key=lambda kv: -kv[1])


def load_conditions(path: Path, labels: list[str]) -> dict[str, set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[str, set[str]] = {}
    for key, cond in data.items():
        codes = set(cond.get("symptoms", {})) | set(cond.get("antecedents", {}))
        for name in (key, cond.get("condition_name"), cond.get("cond-name-eng"), cond.get("cond-name-fr")):
            if name:
                lookup[normalized(name)] = codes
    missing = [label for label in labels if normalized(label) not in lookup]
    if missing:
        raise ValueError(f"Labels not found in {path}: {missing}")
    return {label: lookup[normalized(label)] for label in labels}


def annotate(record: dict, row: dict, aligner: EvidenceAligner, conditions: dict[str, set[str]]) -> list[dict]:
    pred_codes = conditions[record["prediction"]]
    rival_codes = conditions[record["rival"]]
    discriminative = pred_codes - rival_codes
    out = []
    for f, sp, sc in zip(record["features"], record["shap_prediction"], record["shap_contrastive"]):
        item = {**f, "shap_prediction": sp, "shap_contrastive": sc, "negated": bool(NEGATION_RE.search(f["text"]))}
        if f["section"] == "Patient":
            item.update(codes=[DEMOGRAPHICS], scores=[1.0], status="demographics", contrast_status="demographics")
        else:
            links = aligner.align(f["text"], row.get("clinical_evidence", []))
            codes = [c for c, _ in links]
            item.update(codes=codes, scores=[round(s, 3) for _, s in links])
            if not codes:
                item.update(status="unlinked", contrast_status="unlinked")
            else:
                item["status"] = "on_list" if any(c in pred_codes for c in codes) else "off_list"
                item["contrast_status"] = ("discriminative" if any(c in discriminative for c in codes)
                                           else "shared" if any(c in pred_codes for c in codes) else "off_list")
        out.append(item)
    return out


def top_rate(items: list[dict], value_key: str, status_key: str, good: str, k: int):
    """Share of `good` among the top-k positive, linked features vs all linked features."""
    linked = [i for i in items if i[status_key] not in ("unlinked", "demographics")]
    if not linked:
        return None
    top = sorted([i for i in linked if i[value_key] > 0], key=lambda i: -i[value_key])[:k]
    if not top:
        return None
    base = sum(i[status_key] == good for i in linked) / len(linked)
    return {"top": sum(i[status_key] == good for i in top) / len(top), "base": base}


def paired_summary(pairs: list[dict]) -> dict:
    if not pairs:
        return {"n_cases": 0}
    top = [p["top"] for p in pairs]
    base = [p["base"] for p in pairs]
    result = {"n_cases": len(pairs), "top_mean": statistics.mean(top), "base_mean": statistics.mean(base),
              "lift": statistics.mean(top) - statistics.mean(base)}
    try:
        from scipy.stats import wilcoxon
        diffs = [t - b for t, b in zip(top, base)]
        result["wilcoxon_p"] = (float(wilcoxon(top, base, alternative="greater").pvalue)
                                if any(abs(d) > 1e-12 for d in diffs) else None)
    except ImportError:
        result["wilcoxon_p"] = None
    return result


def case_flags(record: dict, items: list[dict]) -> list[str]:
    flags = []
    top_pred = max(items, key=lambda i: i["shap_prediction"])
    top_con = max(items, key=lambda i: i["shap_contrastive"])
    pred_ok = top_pred["shap_prediction"] > 1e-6  # only a positive value is a "top supporting" feature
    con_ok = top_con["shap_contrastive"] > 1e-6
    if pred_ok and top_pred["section"] == "Patient":
        flags.append("demographics_top_supporting_feature")
    if con_ok and top_con["section"] == "Patient":
        flags.append("demographics_top_contrastive_feature")
    if con_ok and top_con["section"] == "Relevant history":
        flags.append("history_top_contrastive_feature")
    if (pred_ok and exact_diagnosis_mention(top_pred["text"], record["prediction"])) or \
            (con_ok and exact_diagnosis_mention(top_con["text"], record["prediction"])):
        flags.append("predicted_label_literal_in_top_feature")
    if con_ok and top_con["contrast_status"] == "off_list":
        flags.append("top_contrastive_feature_off_condition_list")
    if con_ok and top_con["contrast_status"] == "unlinked":
        flags.append("top_contrastive_feature_unlinked")
    if con_ok and top_con["negated"]:
        flags.append("top_contrastive_feature_negated")
    if not con_ok:
        flags.append("no_clause_favors_prediction_over_rival")
    if record["full_values"]["margin"] < 1.0:
        flags.append("small_margin_below_1_nat")
    return flags


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def create_reports(dest: Path, records: list[dict], rows_by_id: dict[str, dict],
                   conditions: dict[str, set[str]], aligner: EvidenceAligner, min_cases: int) -> dict:
    explained = [r for r in records if "skipped" not in r]
    skipped = Counter(r["skipped"] for r in records if "skipped" in r)
    annotated = {r["patient_id"]: annotate(r, rows_by_id[r["patient_id"]], aligner, conditions) for r in explained}

    # ---- plausibility
    plaus: dict[str, dict] = {}
    for group, members in (("all", explained), ("correct", [r for r in explained if r["correct"]]),
                           ("incorrect", [r for r in explained if not r["correct"]])):
        plaus[group] = {}
        for k in (1, 3):
            std = [x for r in members if (x := top_rate(annotated[r["patient_id"]], "shap_prediction",
                                                         "status", "on_list", k))]
            con = [x for r in members if (x := top_rate(annotated[r["patient_id"]], "shap_contrastive",
                                                         "contrast_status", "discriminative", k))]
            plaus[group][f"standard_on_list_top{k}"] = paired_summary(std)
            plaus[group][f"contrastive_discriminative_top{k}"] = paired_summary(con)
    all_items = [i for items in annotated.values() for i in items if i["section"] != "Patient"]
    linkage = {"clauses": len(all_items), "linked": sum(bool(i["codes"]) for i in all_items),
               "negated": sum(i["negated"] for i in all_items)}
    linkage["linked_fraction"] = linkage["linked"] / linkage["clauses"] if all_items else None

    # ---- contrastive deletion
    deletion = {}
    ks = sorted({k for r in explained for k in r["contrastive_deletion"]}, key=int)
    for k in ks:
        d = [r["contrastive_deletion"][k] for r in explained if k in r["contrastive_deletion"]]
        deletion[k] = {"n": len(d), "top_reversal_rate": sum(x["top_reversed"] for x in d) / len(d),
                       "random_reversal_rate": sum(x["random_reversed"] for x in d) / len(d),
                       "mean_top_margin_drop": statistics.mean(
                           r["full_values"]["margin"] - r["contrastive_deletion"][k]["top_margin"]
                           for r in explained if k in r["contrastive_deletion"]),
                       "mean_random_margin_drop": statistics.mean(
                           r["full_values"]["margin"] - r["contrastive_deletion"][k]["random_margin"]
                           for r in explained if k in r["contrastive_deletion"])}

    # ---- per-feature and per-case CSVs, flags
    feature_rows, case_rows, flag_counter = [], [], Counter()
    for r in explained:
        items = annotated[r["patient_id"]]
        flags = case_flags(r, items)
        flag_counter.update(flags)
        top_con = max(items, key=lambda i: i["shap_contrastive"])
        case_rows.append([r["patient_id"], r["truth"], r["prediction"], r["correct"], r["rival"],
                          r["rival_reason"], round(r["full_values"]["margin"], 4), top_con["text"],
                          ";".join(top_con["codes"]), top_con["contrast_status"], ";".join(flags)])
        for i in items:
            feature_rows.append([r["patient_id"], r["prediction"], r["rival"], i["index"], i["section"],
                                 i["text"], round(i["shap_prediction"], 5), round(i["shap_contrastive"], 5),
                                 ";".join(i["codes"]), ";".join(map(str, i["scores"])),
                                 i["status"], i["contrast_status"], i["negated"]])
    write_csv(dest / "cases.csv", ["patient_id", "truth", "prediction", "correct", "rival", "rival_reason",
                                   "margin", "top_contrastive_feature", "top_feature_codes",
                                   "top_feature_status", "flags"], case_rows)
    write_csv(dest / "features_aligned.csv", ["patient_id", "prediction", "rival", "feature", "section", "text",
                                              "shap_prediction", "shap_contrastive", "evidence_codes",
                                              "alignment_scores", "status", "contrast_status", "negated"],
              feature_rows)

    # ---- global: evidence attributions per predicted diagnosis
    questions = {e["evidence_code"]: e.get("question", "") for row in rows_by_id.values()
                 for e in row.get("clinical_evidence", []) if e.get("evidence_code")}
    questions[DEMOGRAPHICS] = "Age and sex"
    agg: dict[tuple[str, str], dict] = defaultdict(lambda: {"cases": set(), "std": 0.0, "con": 0.0})
    sections: dict[str, Counter] = defaultdict(Counter)
    cases_per_dx = Counter(r["prediction"] for r in explained)
    for r in explained:
        for i in annotated[r["patient_id"]]:
            section = "Unlinked clause" if not i["codes"] else i["section"]
            for dx in (r["prediction"], "ALL"):
                sections[dx][("standard", section)] += abs(i["shap_prediction"])
                sections[dx][("contrastive", section)] += abs(i["shap_contrastive"])
            for code in i["codes"]:
                cell = agg[(r["prediction"], code)]
                cell["cases"].add(r["patient_id"])
                cell["std"] += i["shap_prediction"] / len(i["codes"])
                cell["con"] += i["shap_contrastive"] / len(i["codes"])
    global_rows = []
    for (dx, code), cell in agg.items():
        n_dx = len(cell["cases"])
        in_list = "n/a" if code == DEMOGRAPHICS else code in conditions[dx]
        global_rows.append([dx, cases_per_dx[dx], code, questions.get(code, ""), n_dx,
                            round(cell["std"] / n_dx, 5), round(cell["con"] / n_dx, 5), in_list])
    global_rows.sort(key=lambda x: (x[0], -x[6]))
    write_csv(dest / "global_evidence_attributions.csv",
              ["prediction", "cases_predicted", "evidence_code", "question", "cases_with_code",
               "mean_shap_prediction", "mean_shap_contrastive", "in_condition_list"], global_rows)

    section_names = ["Patient", "Presenting findings", "Relevant history", "Unlinked clause"]
    share_rows = []
    for dx in sorted(sections, key=lambda d: (d != "ALL", d)):
        for kind in ("standard", "contrastive"):
            total = sum(sections[dx][(kind, s)] for s in section_names) or 1.0
            share_rows.append([dx, cases_per_dx[dx] if dx != "ALL" else len(explained), kind,
                               *[round(sections[dx][(kind, s)] / total, 4) for s in section_names]])
    write_csv(dest / "global_section_shares.csv",
              ["prediction", "cases", "attribution", *[f"share_{s.lower().replace(' ', '_')}" for s in section_names]],
              share_rows)

    lines = ["# Global explanations: top evidence per predicted diagnosis",
             "", f"Mean contrastive SHAP per case where the evidence occurs (diagnoses with >= {min_cases} cases).",
             "On-list = evidence belongs to the diagnosis in DDXPlus release_conditions.json.", ""]
    for dx in sorted(d for d, c in cases_per_dx.items() if c >= min_cases):
        top = [g for g in global_rows if g[0] == dx and g[6] > 0][:5]
        lines += [f"## {dx} ({cases_per_dx[dx]} cases)", "",
                  "| Evidence | Question | Cases | Mean contrastive SHAP | On-list |", "|---|---|---:|---:|:---:|"]
        lines += [f"| {g[2]} | {g[3][:90]} | {g[4]} | {g[6]:+.3f} | "
                  f"{'n/a' if g[7] == 'n/a' else ('yes' if g[7] else '**NO**')} |" for g in top]
        lines.append("")
    (dest / "global_top_evidence.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "format_version": FORMAT_VERSION,
        "cases_explained": len(explained), "cases_skipped": dict(skipped),
        "correct": sum(r["correct"] for r in explained),
        "rival_reasons": dict(Counter(r["rival_reason"] for r in explained)),
        "prediction_sources": dict(Counter(r["prediction_source"] for r in explained)),
        "mean_seconds_per_case": statistics.mean(r["seconds"] for r in explained) if explained else None,
        "max_additivity_error": max((r["additivity_error"] for r in explained), default=None),
        "render_mismatch_cases": sum(not r["render_matches_original"] for r in explained),
        "alignment": {"mode": aligner.mode, "threshold": aligner.threshold, **linkage},
        "plausibility": plaus,
        "contrastive_deletion": deletion,
        "shortcut_flag_counts": dict(flag_counter.most_common()),
        "notes": [
            "DDXPlus samples a patient's evidence from the TRUE condition's list, so standard on-list "
            "rates are near 1 for correct predictions by construction; read the lift over the base rate "
            "and the contrastive 'discriminative' rate instead.",
            "Alignment is automatic (lexical or embedding) and must be spot-checked by hand.",
            "Plausibility = agreement with DDXPlus condition definitions, not clinical validity.",
        ],
    }
    write_json_atomic(dest / "summary.json", summary)
    make_figures(dest, plaus, deletion, share_rows)
    return summary


def _style_axes(ax):
    ax.set_facecolor(C_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT2, labelsize=9)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(dest: Path, plaus: dict, deletion: dict, share_rows: list[list]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker
    except ImportError:
        print("matplotlib unavailable: skipping figures")
        return
    plt.rcParams.update({"font.size": 10, "text.color": C_TEXT, "axes.labelcolor": C_TEXT2})

    # Plausibility: top-k SHAP features vs the case's own base rate.
    keys = [("standard_on_list_top1", "Standard, top 1"), ("standard_on_list_top3", "Standard, top 3"),
            ("contrastive_discriminative_top1", "Contrastive, top 1"),
            ("contrastive_discriminative_top3", "Contrastive, top 3")]
    stats = [plaus["all"][k] for k, _ in keys]
    if any(s.get("n_cases") for s in stats):
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=C_SURFACE)
        x = range(len(keys))
        width = 0.36
        tops = [s.get("top_mean", 0) or 0 for s in stats]
        bases = [s.get("base_mean", 0) or 0 for s in stats]
        ax.bar([i - width / 2 - 0.01 for i in x], tops, width, color=C_BLUE, label="Top-SHAP clauses")
        ax.bar([i + width / 2 + 0.01 for i in x], bases, width, color=C_ORANGE, label="All clauses (base rate)")
        for i, (t, s) in enumerate(zip(tops, stats)):
            p = s.get("wilcoxon_p")
            ax.text(i - width / 2, t + 0.02, f"{t:.2f}" + (f"\np={p:.3g}" if p is not None else ""),
                    ha="center", va="bottom", fontsize=8, color=C_TEXT)
        ax.set_xticks(list(x), [label for _, label in keys])
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Share of clauses")
        ax.set_title("Plausibility: on-list (standard) / discriminative (contrastive) evidence",
                     fontsize=11, loc="left", color=C_TEXT)
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        fig.tight_layout()
        fig.savefig(dest / "plausibility.png", dpi=180)
        plt.close(fig)

    # Contrastive deletion: margin reversal rate, top-k vs random.
    if deletion:
        ks = sorted(deletion, key=int)
        fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=C_SURFACE)
        top = [deletion[k]["top_reversal_rate"] for k in ks]
        rnd = [deletion[k]["random_reversal_rate"] for k in ks]
        ax.plot([int(k) for k in ks], top, color=C_BLUE, linewidth=2, marker="o", markersize=8,
                label="Remove top-k contrastive clauses")
        ax.plot([int(k) for k in ks], rnd, color=C_ORANGE, linewidth=2, marker="o", markersize=8,
                label="Remove k random clauses")
        ax.annotate(f"{top[-1]:.0%}", (int(ks[-1]), top[-1]), textcoords="offset points", xytext=(8, 0),
                    va="center", fontsize=9, color=C_TEXT)
        ax.annotate(f"{rnd[-1]:.0%}", (int(ks[-1]), rnd[-1]), textcoords="offset points", xytext=(8, 0),
                    va="center", fontsize=9, color=C_TEXT)
        ax.set_xticks([int(k) for k in ks])
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("Clauses removed (k)")
        ax.set_ylabel("Cases where the rival overtakes")
        ax.set_title("Contrastive deletion: does removing the explanation flip X vs Y?",
                     fontsize=11, loc="left", color=C_TEXT)
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(dest / "contrastive_deletion.png", dpi=180)
        plt.close(fig)

    # Section shares of contrastive attribution, per diagnosis (>= 1 case) + ALL.
    rows = [r for r in share_rows if r[2] == "contrastive"]
    if rows:
        rows = [rows[0]] + sorted(rows[1:], key=lambda r: -r[5])  # ALL first, then by history share
        labels = [f"{r[0]} ({r[1]})" for r in rows]
        parts = [("Demographics", 3, C_AQUA), ("Presenting findings", 4, C_BLUE),
                 ("History", 5, C_ORANGE), ("Unlinked clauses", 6, "#b9b8b2")]
        fig, ax = plt.subplots(figsize=(9, max(3, 0.28 * len(rows) + 1.2)), facecolor=C_SURFACE)
        left = [0.0] * len(rows)
        for name, col, color in parts:
            vals = [r[col] for r in rows]
            ax.barh(range(len(rows)), vals, left=left, color=color, label=name,
                    edgecolor=C_SURFACE, linewidth=1.5, height=0.7)
            left = [a + b for a, b in zip(left, vals)]
        ax.set_yticks(range(len(rows)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of |contrastive SHAP|")
        ax.set_title("Where does each diagnosis get its evidence from?", fontsize=11, loc="left",
                     color=C_TEXT, pad=30)
        _style_axes(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.8)
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout()
        fig.savefig(dest / "section_shares.png", dpi=180)
        plt.close(fig)


# --------------------------------------------------------------------------
# CLI: selection, manifest, resumable GPU stage, reports
# --------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_rows(rows: list[dict], args: argparse.Namespace, predictions: dict[str, dict]) -> list[dict]:
    rng = random.Random(args.seed)
    if args.per_class:
        by_label = defaultdict(list)
        for r in sorted(rows, key=lambda r: r["patient_id"]):
            by_label[r["target"]["diagnosis"]].append(r)
        selected = []
        for label in sorted(by_label):
            group = by_label[label]
            selected += rng.sample(group, min(args.per_class, len(group)))
    else:
        if not 1 <= args.n <= len(rows):
            raise ValueError(f"--n must be between 1 and {len(rows)}")
        selected = rng.sample(rows, args.n)
    if args.all_errors:
        chosen = {r["patient_id"] for r in selected}
        selected += [r for r in rows if r["patient_id"] not in chosen
                     and r["patient_id"] in predictions
                     and predictions[r["patient_id"]]["prediction"] not in (INVALID, r["target"]["diagnosis"])]
    return selected


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", choices=("vanilla", "1000", "3000", "5000"), required=True)
    p.add_argument("--split", choices=("validate", "test"), default="test")
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--train-data", type=Path, default=Path("data/generated_clinical_v2/train.jsonl"))
    p.add_argument("--conditions", type=Path, default=Path("data/ddxplus/release_conditions.json"))
    p.add_argument("--predictions", type=Path, default=None,
                   help="Open-ended predictions.jsonl to explain (recommended). Without it the "
                        "teacher-forced argmax is explained and recorded as such.")
    p.add_argument("--model", default="google/gemma-4-12B-it")
    p.add_argument("--revision", default=None)
    p.add_argument("--adapter-root", type=Path, default=Path("runs/gemma4_qlora_independent"))
    p.add_argument("--output-root", type=Path, default=Path("runs/xai_analysis"))
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--n", type=int, default=100, help="Random cases (default 100)")
    sel.add_argument("--per-class", type=int, default=None, help="Cases per true diagnosis (stratified)")
    p.add_argument("--all-errors", action="store_true", help="Also add every wrong (non-invalid) prediction")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-features", type=int, default=12)
    p.add_argument("--permutations", type=int, default=2)
    p.add_argument("--score", choices=("complete", "diagnosis"), default="complete",
                   help="complete = label + end-of-turn tokens (default); diagnosis = label tokens only")
    p.add_argument("--rival-mode", choices=("truth_if_wrong", "runner_up"), default="truth_if_wrong")
    p.add_argument("--deletion-k", type=int, default=3)
    p.add_argument("--score-batch", type=int, default=8, help="Labels per forward pass (reduce if OOM)")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--aligner", choices=("lexical", "embed"), default="lexical")
    p.add_argument("--align-threshold", type=float, default=None)
    p.add_argument("--align-relative", type=float, default=0.7)
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--min-cases", type=int, default=3, help="Min predicted cases for global tables")
    p.add_argument("--reports-only", action="store_true")
    args = p.parse_args()
    if args.max_features < 3 or not 1 <= args.permutations <= 20 or args.deletion_k < 1 or args.score_batch < 1:
        p.error("Invalid max-features / permutations / deletion-k / score-batch")
    if args.data is None:
        args.data = Path(f"data/generated_clinical_v2/{args.split}.jsonl")

    rows = read_rows(args.data)
    rows_by_id = {r["patient_id"]: r for r in rows}
    labels = sorted({r["target"]["diagnosis"] for r in read_rows(args.train_data) + rows})
    conditions = load_conditions(args.conditions, labels)
    predictions = {}
    if args.predictions:
        for r in read_jsonl(args.predictions):
            if r["patient_id"] in rows_by_id:
                if r.get("truth") != rows_by_id[r["patient_id"]]["target"]["diagnosis"]:
                    raise ValueError(f"Prediction truth mismatch for {r['patient_id']}")
                predictions[r["patient_id"]] = r
    elif args.all_errors:
        p.error("--all-errors requires --predictions")
    selected = select_rows(rows, args, predictions)

    sample = f"perclass{args.per_class}" if args.per_class else f"n{args.n}"
    dest = (args.output_root / args.split / ("vanilla" if args.model_id == "vanilla" else f"stage-{args.model_id}")
            / f"{sample}{'_errors' if args.all_errors else ''}_seed{args.seed}")
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": FORMAT_VERSION, "model": args.model, "revision": args.revision,
        "model_id": args.model_id,
        "adapter_signature": None if args.model_id == "vanilla" else adapter_signature(adapter_path(args)),
        "data": str(args.data.resolve()),
        "selected_fingerprint": sha([[r["patient_id"], r["clinical_summary"], r["target"]["diagnosis"]]
                                     for r in selected]),
        "predictions_fingerprint": sha(sorted([pid, predictions[pid]["prediction"]] for pid in predictions)
                                       ) if predictions else None,
        "vocabulary": labels, "prompt": PROMPT, "rendering": "training_matched_completed_turn",
        "max_features": args.max_features, "permutations": args.permutations, "seed": args.seed,
        "score": args.score, "rival_mode": args.rival_mode, "deletion_k": args.deletion_k,
    }
    manifest_path, out_path = dest / "manifest.json", dest / "explanations.jsonl"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"Configuration changed for {dest}; use a new --output-root or seed")
    elif out_path.exists():
        raise RuntimeError(f"{out_path} exists without a manifest; refusing to mix runs")
    else:
        write_json_atomic(manifest_path, manifest)

    done = {r["patient_id"]: r for r in read_jsonl(out_path)} if out_path.exists() else {}
    pending = [r for r in selected if r["patient_id"] not in done]
    print(f"{len(selected)} selected; {len(done)} done; {len(pending)} pending -> {dest}", flush=True)

    if pending and not args.reports_only:
        model, processor = load_model(args)
        scorer = LabelScorer(model, processor, args.max_length, args.score_batch)
        with out_path.open("a", encoding="utf-8") as out:
            for index, row in enumerate(pending, 1):
                pid = row["patient_id"]
                cached = None
                if predictions:
                    if pid not in predictions:
                        record = {"patient_id": pid, "skipped": "no_cached_prediction"}
                    elif predictions[pid]["prediction"] == INVALID:
                        record = {"patient_id": pid, "skipped": "invalid_prediction_safe_stop"}
                    else:
                        cached = predictions[pid]["prediction"]
                if not predictions or cached is not None:
                    try:
                        record = explain_case(scorer, row, labels, cached, args, self_check=index == 1)
                    except ValueError as exc:
                        record = {"patient_id": pid, "skipped": f"error: {exc}"}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                done[pid] = record
                if index % 5 == 0 or index == len(pending):
                    print(f"{index}/{len(pending)} explained ({record.get('seconds', '-')} s last)", flush=True)

    records = [done[r["patient_id"]] for r in selected if r["patient_id"] in done]
    if len(records) != len(selected):
        print(f"Reporting on {len(records)}/{len(selected)} completed cases", flush=True)
    if not records:
        raise SystemExit("Nothing to report yet.")
    aligner = EvidenceAligner([rows_by_id[r["patient_id"]] for r in records], args.aligner,
                              args.align_threshold, args.align_relative, args.embed_model)
    summary = create_reports(dest, records, rows_by_id, conditions, aligner, args.min_cases)
    print(json.dumps({k: summary[k] for k in ("cases_explained", "cases_skipped", "alignment",
                                              "contrastive_deletion", "shortcut_flag_counts")}, indent=2))
    print(f"Reports in {dest}")


if __name__ == "__main__":
    main()
