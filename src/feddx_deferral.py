"""Calibrated confidence and human-deferral zones for FedDx.

Shared by `evaluate_trust.py` (which fits the policy) and `app_gradio.py`
(which applies it to live predictions). Pure Python: no GPU or sklearn needed
at application time.

A deferral policy maps an uncalibrated confidence score to a calibrated
probability that the predicted label is correct (Platt scaling), then to one of
three zones:

  AUTO_ACCEPT   calibrated p >= accept_threshold
  REVIEW        stop_threshold <= p < accept_threshold  -> clinician review
  SAFE_STOP     p < stop_threshold, invalid label, or no score -> no model output shown as a diagnosis

Thresholds are fitted on a calibration split and must be evaluated on a
separate split. They describe synthetic DDXPlus behaviour only.
"""
from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any

ZONES = ("AUTO_ACCEPT", "REVIEW", "SAFE_STOP")
EPS = 1e-6

ZONE_STYLE = {
    "AUTO_ACCEPT": ("Auto-accept", "#166534", "#dcfce7",
                    "Calibrated confidence meets the acceptance target. Logged and sample-audited."),
    "REVIEW": ("Clinician review", "#854d0e", "#fef9c3",
               "Show prediction and SHAP evidence to a clinician, who confirms or overrides."),
    "SAFE_STOP": ("Safe stop", "#991b1b", "#fee2e2",
                  "Do not present a model diagnosis. Route to standard clinical workup."),
}


def logit(p: float) -> float:
    p = min(max(float(p), EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def calibrate(calibrator: dict[str, Any], score: float) -> float:
    kind = calibrator.get("type")
    if kind == "identity":
        return min(max(float(score), 0.0), 1.0)
    if kind == "platt":
        return sigmoid(calibrator["a"] * logit(score) + calibrator["b"])
    raise ValueError(f"Unknown calibrator type: {kind!r}")


def assign_zone(policy: dict[str, Any], score: float | None, valid_label: bool = True) -> tuple[str, float | None]:
    """Return (zone, calibrated_probability). Invalid or unscored -> SAFE_STOP."""
    if not valid_label or score is None or not math.isfinite(float(score)):
        return "SAFE_STOP", None
    p = calibrate(policy["calibrator"], score)
    accept = policy.get("accept_threshold")
    if accept is not None and p >= accept:
        return "AUTO_ACCEPT", p
    if p < policy["stop_threshold"]:
        return "SAFE_STOP", p
    return "REVIEW", p


def load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("score", "calibrator", "stop_threshold"):
        if key not in policy:
            raise ValueError(f"Deferral policy {path} lacks {key!r}")
    return policy


def render_zone_html(policy: dict[str, Any] | None, zone: str | None, calibrated: float | None) -> str:
    if policy is None or zone is None:
        return ""
    label, fg, bg, action = ZONE_STYLE[zone]
    prob = "n/a" if calibrated is None else f"{100 * calibrated:.1f}%"
    accept = policy.get("accept_threshold")
    accept_txt = "none (no auto-accept)" if accept is None else f"{100 * accept:.1f}%"
    return (
        f"<div class='feddx-confidence' style='margin-top:10px;border-radius:12px;padding:13px;"
        f"background:{bg};border-left:6px solid {fg};color:#111827'>"
        f"<div style='font-size:0.8em;font-weight:700'>DEFERRAL ZONE</div>"
        f"<div style='font-size:1.35em;font-weight:750;color:{fg}'>{escape(label)}</div>"
        f"<div style='font-size:0.88em;margin-top:4px'>Calibrated P(correct label): <b>{prob}</b> · "
        f"accept ≥ {accept_txt} · stop &lt; {100 * policy['stop_threshold']:.1f}%</div>"
        f"<div style='font-size:0.84em;margin-top:6px'>{escape(action)}</div>"
        f"<div style='font-size:0.76em;margin-top:6px'>Policy fitted on {escape(str(policy.get('fitted_on_n', '?')))} "
        f"synthetic cases ({escape(str(policy.get('fitted_on', 'unknown')))}). Research demo, not a clinical rule.</div>"
        "</div>"
    )
