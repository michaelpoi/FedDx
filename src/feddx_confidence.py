"""Uncalibrated, token-level confidence signals for FedDx open-ended generation.

These scores are NOT probabilities that a medical diagnosis is correct. They
measure probabilities of tokens in the actual generated diagnosis under the
model's generation distribution. No second (closed-set) prompt is introduced.
"""
from __future__ import annotations

import math
import re
from typing import Any


def diagnosis_span(decoded: str) -> tuple[int, int] | None:
    """Locate the first diagnosis line, skipping only the known `thought` header."""
    match = re.match(r"\s*(?:thought\s*\n)?\s*", decoded, re.IGNORECASE)
    start = match.end() if match else 0
    tail = decoded[start:]
    first_line = tail.splitlines()[0].strip() if tail else ""
    if not first_line:
        return None
    offset = tail.find(first_line)
    return start + offset, start + offset + len(first_line)


def score_generated_diagnosis(tokenizer: Any, sequences: Any, step_scores: Any,
                              input_length: int) -> dict[str, Any]:
    """Extract per-token scores using original generated IDs and step logits.

    Each step_scores[i] is the generation-time next-token logits/scores of
    shape (1, vocab_size); no re-tokenization of the output is used. Decode
    prefixes to associate generated token positions with the *visible* answer.
    If boundaries cannot be established, report unavailable rather than score
    channel-markup or padding tokens as the diagnosis.
    """
    import torch

    ids = sequences[0, input_length:].tolist()
    n = min(len(ids), len(step_scores))
    ids = ids[:n]
    decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    span = diagnosis_span(decoded)
    if span is None:
        return {"available": False, "reason": "No diagnosis text was generated."}

    # Compute token-to-text character intervals by decoding the prefixes.
    # This handles leading 'thought' channel artifacts and special tokens.
    prefix_lengths = [0]
    for i in range(1, n + 1):
        prefix = tokenizer.decode(ids[:i], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
        prefix_lengths.append(len(prefix))
    if prefix_lengths[-1] != len(decoded) or any(b < a for a, b in zip(prefix_lengths, prefix_lengths[1:])):
        return {"available": False, "reason": "Generated text could not be aligned to tokens."}

    selected = [i for i in range(n)
                if prefix_lengths[i] < span[1] and prefix_lengths[i + 1] > span[0]]
    if not selected:
        return {"available": False, "reason": "Diagnosis tokens could not be identified."}
    # Ensure no token straddles unrelated text outside of diagnosis (apart from whitespace).
    for i in selected:
        before = decoded[prefix_lengths[i]:min(span[0], prefix_lengths[i + 1])]
        after = decoded[max(span[1], prefix_lengths[i]):prefix_lengths[i + 1]]
        if before.strip() or after.strip():
            return {"available": False, "reason": "A token overlaps the diagnosis boundary."}

    log_probs = []
    for i in selected:
        logits = step_scores[i][0].float()
        logp = float(torch.log_softmax(logits, dim=-1)[ids[i]].item())
        if not math.isfinite(logp):
            return {"available": False, "reason": "Non-finite token probability."}
        log_probs.append(logp)

    mean_logp = sum(log_probs) / len(log_probs)
    geometric_mean = math.exp(mean_logp)
    minimum = math.exp(min(log_probs))
    joint = math.exp(sum(log_probs))
    return {
        "available": True,
        "method": "generated_diagnosis_token_logprobs",
        "geometric_mean_token_probability": geometric_mean,
        "minimum_token_probability": minimum,
        "joint_diagnosis_token_probability": joint,
        "mean_token_log_probability": mean_logp,
        "number_of_diagnosis_tokens": len(log_probs),
        "token_probabilities": [math.exp(v) for v in log_probs],
        "decoded_answer": decoded[span[0]:span[1]],
        "note": "Uncalibrated token likelihoods, not probabilities of clinical correctness.",
    }


def confidence_band(score: float) -> tuple[str, str, str]:
    """Heuristic display categories; NOT calibrated against clinical accuracy."""
    if score >= 0.80:
        return "HIGH", "#166534", "#dcfce7"
    if score >= 0.50:
        return "MODERATE", "#854d0e", "#fef9c3"
    return "LOW", "#991b1b", "#fee2e2"


def render_confidence_html(stats: dict[str, Any] | None, *, cached: bool = False) -> str:
    """A standalone HTML view for gr.HTML; only computed numeric values enter markup."""
    if not stats or not stats.get("available"):
        reason = ("Saved evaluation predictions do not contain token scores. "
                  "Uncheck 'Use saved test predictions' and predict again to calculate them."
                  if cached else (stats or {}).get("reason", "Run live prediction to calculate token scores."))
        from html import escape
        return ("<div style='padding:14px;border-radius:12px;background:#e5e7eb;color:#374151'>"
                "<b>Confidence unavailable</b><div style='margin-top:5px;font-size:0.9em'>"
                f"{escape(reason)}</div></div>")
    score = float(stats["geometric_mean_token_probability"])
    band, fg, bg = confidence_band(score)
    pct = 100 * score
    minpct = 100 * float(stats["minimum_token_probability"])
    jointpct = 100 * float(stats["joint_diagnosis_token_probability"])
    count = int(stats["number_of_diagnosis_tokens"])
    return f"""<div class="feddx-confidence" style="border-radius:12px;padding:15px;background:{bg};color:#111827 !important;-webkit-text-fill-color:#111827 !important">
      <div style="font-size:0.82em;font-weight:700;color:#111827 !important;-webkit-text-fill-color:#111827 !important">{band} TOKEN CONFIDENCE · UNCALIBRATED</div>
      <div style="font-size:1.9em;font-weight:750;margin-top:3px;color:#111827 !important;-webkit-text-fill-color:#111827 !important">{pct:.1f}%</div>
      <div style="height:9px;background:#ffffffbb;border-radius:6px;margin:8px 0;overflow:hidden">
        <div style="height:100%;width:{pct:.3f}%;background:{fg};border-radius:6px"></div>
      </div>
      <div style="font-size:0.86em;color:#111827 !important;-webkit-text-fill-color:#111827 !important">Geometric mean probability of {count} generated diagnosis tokens</div>
      <div style="font-size:0.84em;margin-top:10px;color:#111827 !important;-webkit-text-fill-color:#111827 !important">Weakest token: <b>{minpct:.1f}%</b> &nbsp;·&nbsp;
           Whole diagnosis token sequence: <b>{jointpct:.1f}%</b></div>
      <div style="font-size:0.79em;margin-top:9px;color:#111827 !important;-webkit-text-fill-color:#111827 !important">Colors use illustrative 50% / 80% thresholds.
        This is <b>not</b> a calibrated probability that the diagnosis is correct.</div>
    </div>"""
