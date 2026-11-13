#!/usr/bin/env python3
"""FedDx trust evaluation: calibration, calibrated deferral, subgroup behaviour.

CPU only. Reads saved prediction files; no model is loaded.

Supported inputs (auto-detected):
  token       evaluate_feddx_confidence.py predictions.jsonl
              score = geometric-mean probability of generated diagnosis tokens
              (the score the Gradio app shows live)
  closed_set  evaluate_gemma4_closed_set.py predictions.jsonl
              score = softmax of the stored top-3 mean log-probabilities

Pipeline:
  1. Fit Platt scaling (calibrated P(label correct)) on the CALIBRATION split.
  2. Choose deferral thresholds on the CALIBRATION split:
       auto-accept: lowest threshold whose accepted cases have a Wilson 95%
                    lower bound on accuracy >= --target-accuracy
       safe-stop:   calibrated p < --stop-threshold, invalid label, or no score
       review:      everything in between
  3. Report everything on the held-out EVALUATION split: reliability diagram,
     ECE/Brier/AUROC before vs after calibration, zones, risk-coverage curve,
     and per-subgroup (sex, age band) accuracy and deferral rates with CIs.

Examples (run from the FedDx root):
  # Separate calibration and evaluation files (preferred)
  uv run python src/evaluate_trust.py \
      --calib runs/feddx_confidence/validate/stage-5000/n1000_seed42/predictions.jsonl \
      --eval  runs/feddx_confidence/test/stage-5000/n1000_seed42/predictions.jsonl \
      --output-dir runs/trust/token_5k

  # One file, split deterministically 50/50 into calibration/evaluation halves
  uv run python src/evaluate_trust.py \
      --calib runs/gemma4_evaluation_closed_set/validate/stage-5000/predictions.jsonl \
      --output-dir runs/trust/closed_set_5k

Outputs: trust_metrics.json, deferral_policy.json (load in the app with
--deferral-policy), reliability_diagram.png, risk_coverage.png,
subgroups.csv, subgroups.png, zones.csv, trust_report.md.

All results describe synthetic DDXPlus cases and LLM-written summaries. They
are evidence for a research prototype, not a validated clinical rule.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from feddx_deferral import ZONES, assign_zone, calibrate, logit

INVALID = "[INVALID / OUT OF VOCABULARY]"
AGE_BANDS = ((0, 17, "0-17"), (18, 39, "18-39"), (40, 64, "40-64"), (65, 200, "65+"))


# ----------------------------------------------------------------- loading
def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Bad JSON at {path}:{n}") from exc
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def detect_source(rows: list[dict]) -> str:
    first = rows[0]
    if "top3" in first:
        return "closed_set"
    if "confidence" in first:
        return "token"
    raise ValueError("Predictions contain no confidence scores (open-ended predictions.jsonl stores none). "
                     "Use evaluate_feddx_confidence.py or evaluate_gemma4_closed_set.py output.")


def record_score(row: dict, source: str) -> float | None:
    if source == "token":
        c = row.get("confidence") or {}
        if not c.get("available"):
            return None
        return float(c["geometric_mean_token_probability"])
    lps = [float(t["mean_logprob"]) for t in row.get("top3", [])]
    if not lps:
        return None
    m = max(lps)
    return 1.0 / sum(math.exp(lp - m) for lp in lps)


def load_predictions(path: Path, source: str) -> tuple[list[dict], str]:
    rows = read_jsonl(path)
    detected = detect_source(rows)
    if source != "auto" and source != detected:
        raise ValueError(f"{path} looks like {detected!r} output, not {source!r}")
    seen, records = set(), []
    for row in rows:
        pid = row["patient_id"]
        if pid in seen:
            raise ValueError(f"Duplicate patient_id {pid} in {path}")
        seen.add(pid)
        prediction = row.get("prediction") or ""
        valid = bool(prediction) and prediction != INVALID
        records.append({
            "patient_id": pid, "truth": row["truth"], "prediction": prediction,
            "valid": valid, "correct": valid and prediction == row["truth"],
            "score": record_score(row, detected) if valid else None,
        })
    return records, detected


def load_features(paths: list[Path]) -> dict[str, dict]:
    features = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            features[row["patient_id"]] = row.get("patient_features", {}) or {}
    return features


def split_records(records: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic hash split: calibration gets ~fraction of patients."""
    calib, evaluation = [], []
    for r in records:
        h = int(hashlib.sha256(f"{seed}:{r['patient_id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (calib if h < fraction else evaluation).append(r)
    return calib, evaluation


# ---------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def calibration_metrics(probs: list[float], correct: list[bool], bins: int) -> dict:
    n = len(probs)
    if n == 0:
        return {"n": 0}
    table = []
    ece = mce = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not idx:
            table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "mean_confidence": None, "accuracy": None})
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(correct[i] for i in idx) / len(idx)
        gap = abs(acc - conf)
        ece += len(idx) / n * gap
        mce = max(mce, gap)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(idx), "mean_confidence": conf, "accuracy": acc})
    brier = sum((p - c) ** 2 for p, c in zip(probs, correct)) / n
    auroc = None
    if 0 < sum(correct) < n:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score([int(c) for c in correct], probs))
    return {"n": n, "accuracy": sum(correct) / n, "mean_confidence": sum(probs) / n,
            "ece": ece, "mce": mce, "brier": brier, "auroc_correctness": auroc, "bins": table}


# ---------------------------------------------------------------- fitting
def fit_platt(records: list[dict]) -> tuple[dict, str | None]:
    scored = [r for r in records if r["score"] is not None]
    ys = [int(r["correct"]) for r in scored]
    if len(scored) < 20 or len(set(ys)) < 2:
        return ({"type": "identity"},
                f"Platt scaling skipped: {len(scored)} scored calibration cases with "
                f"{sum(ys)} correct / {len(ys) - sum(ys)} incorrect. Need >=20 cases and both outcomes. "
                "Raw scores are used unchanged; enlarge the calibration split.")
    from sklearn.linear_model import LogisticRegression
    x = [[logit(r["score"])] for r in scored]
    model = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000).fit(x, ys)
    warning = None
    if min(sum(ys), len(ys) - sum(ys)) < 10:
        warning = (f"Only {min(sum(ys), len(ys) - sum(ys))} examples of the minority outcome in the "
                   "calibration split; the Platt fit is unstable.")
    return {"type": "platt", "a": float(model.coef_[0][0]), "b": float(model.intercept_[0])}, warning


def choose_accept_threshold(records: list[dict], calibrator: dict, target: float, min_n: int) -> dict:
    scored = sorted(((calibrate(calibrator, r["score"]), r["correct"]) for r in records if r["score"] is not None),
                    key=lambda pc: -pc[0])
    best = None
    k = 0
    for i, (p, c) in enumerate(scored, 1):
        k += c
        # A threshold is only valid at the end of a run of tied probabilities.
        if i < len(scored) and scored[i][0] == p:
            continue
        lower = wilson(k, i)[0]
        if i >= min_n and lower is not None and lower >= target:
            best = {"threshold": p, "accepted_n": i, "accepted_correct": k,
                    "coverage": i / len(records), "wilson_lower": lower}
    return best or {"threshold": None, "reason":
                    f"No threshold reaches Wilson lower bound >= {target:.3f} with >= {min_n} accepted cases."}


# ---------------------------------------------------------------- evaluation
def zone_table(records: list[dict]) -> dict:
    n = len(records)
    out = {}
    for zone in ZONES:
        subset = [r for r in records if r["zone"] == zone]
        k = sum(r["correct"] for r in subset)
        lo, hi = wilson(k, len(subset))
        out[zone] = {"n": len(subset), "share": len(subset) / n if n else None,
                     "correct": k, "errors": len(subset) - k,
                     "accuracy": k / len(subset) if subset else None, "accuracy_ci95": [lo, hi]}
    return out


def risk_coverage(records: list[dict]) -> dict:
    scored = sorted((r for r in records if r["calibrated"] is not None), key=lambda r: -r["calibrated"])
    coverage, risk, errors = [], [], 0
    for i, r in enumerate(scored, 1):
        errors += not r["correct"]
        coverage.append(i / len(records))
        risk.append(errors / i)
    aurc = sum(risk) / len(risk) if risk else None
    return {"coverage": coverage, "risk": risk, "aurc": aurc}


def age_band(age) -> str:
    try:
        a = int(age)
    except (TypeError, ValueError):
        return "unknown"
    for lo, hi, name in AGE_BANDS:
        if lo <= a <= hi:
            return name
    return "unknown"


def subgroup_table(records: list[dict], features: dict, min_n: int) -> tuple[list[dict], dict]:
    groups = defaultdict(list)
    for r in records:
        f = features.get(r["patient_id"], {})
        sex = str(f.get("sex") or "unknown").upper()
        groups[("sex", sex)].append(r)
        groups[("age_band", age_band(f.get("age")))].append(r)
    rows = []
    order = {"sex": 0, "age_band": 1}
    band_order = {name: i for i, (_, _, name) in enumerate(AGE_BANDS)}
    for (dim, level), subset in sorted(groups.items(), key=lambda kv: (order[kv[0][0]], band_order.get(kv[0][1], 9), kv[0][1])):
        n = len(subset)
        k = sum(r["correct"] for r in subset)
        lo, hi = wilson(k, n)
        correct_cases = [r for r in subset if r["correct"]]
        rows.append({
            "dimension": dim, "group": level, "n": n,
            "accuracy": k / n, "accuracy_ci_low": lo, "accuracy_ci_high": hi,
            "auto_accept_rate": sum(r["zone"] == "AUTO_ACCEPT" for r in subset) / n,
            "review_rate": sum(r["zone"] == "REVIEW" for r in subset) / n,
            "safe_stop_rate": sum(r["zone"] == "SAFE_STOP" for r in subset) / n,
            # Equal-opportunity analogue: correct predictions that reach auto-accept.
            "auto_accept_rate_given_correct": (sum(r["zone"] == "AUTO_ACCEPT" for r in correct_cases) / len(correct_cases)
                                               if correct_cases else None),
            "silent_errors": sum(r["zone"] == "AUTO_ACCEPT" and not r["correct"] for r in subset),
            "small_sample": n < min_n,
        })
    disparity = {}
    for dim in ("sex", "age_band"):
        sized = [r for r in rows if r["dimension"] == dim and not r["small_sample"] and r["group"] != "unknown"]
        if len(sized) < 2:
            disparity[dim] = {"note": f"Fewer than two groups with n >= {min_n}"}
            continue
        accs = [r["accuracy"] for r in sized]
        acc_rates = [r["auto_accept_rate"] for r in sized]
        disparity[dim] = {
            "groups_compared": [r["group"] for r in sized],
            "accuracy_gap_max_minus_min": max(accs) - min(accs),
            # Group parity index (lecture: GPI = P(favourable|A=a)/P(favourable|A=b)),
            # with auto-accept treated as the "favourable" fast path.
            "auto_accept_parity_index_min_over_max": (min(acc_rates) / max(acc_rates)) if max(acc_rates) > 0 else None,
        }
    return rows, disparity


# ---------------------------------------------------------------- plotting
def plot_reliability(raw_probs: list[float], cal_probs: list[float], correct: list[bool],
                     raw: dict, cal: dict, dest: Path, title: str, bins: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Zoom to the range that contains scores: saturated scores would otherwise
    # leave most of the axis empty and pack nearly all cases into one bar.
    # ECE/Brier in the titles still use the standard [0, 1] bins.
    lo = min(min(raw_probs), min(cal_probs))
    lo = max(0.0, math.floor(lo * 20) / 20)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for col, (name, probs, m) in enumerate((("Raw score", raw_probs, raw), ("Platt-calibrated", cal_probs, cal))):
        ax, hx = axes[0][col], axes[1][col]
        width = (1 - lo) / bins
        centres, accs, confs, counts = [], [], [], []
        for b in range(bins):
            a, z = lo + b * width, lo + (b + 1) * width
            idx = [i for i, p in enumerate(probs) if a <= p < z or (b == bins - 1 and p >= z)]
            centres.append((a + z) / 2)
            counts.append(len(idx))
            accs.append(sum(correct[i] for i in idx) / len(idx) if idx else 0)
            confs.append(sum(probs[i] for i in idx) / len(idx) if idx else 0)
        ax.bar(centres, accs, width=width * 0.9, color="#3b82f6", alpha=0.8, label="Observed accuracy")
        ax.bar(centres, [abs(c - a) if n else 0 for c, a, n in zip(confs, accs, counts)],
               bottom=[min(c, a) for c, a in zip(confs, accs)],
               width=width * 0.9, color="none", edgecolor="#ef4444", hatch="//", lw=1,
               label="Gap to mean confidence")
        for x, a, c, n in zip(centres, accs, confs, counts):
            if n:
                ax.text(x, min(max(a, c) + 0.012, 1.04), f"n={n}", ha="center", va="bottom", fontsize=7)
        ax.plot([lo, 1], [lo, 1], "k--", lw=1, label="Perfect calibration")
        ax.set(ylim=(0, 1.08), ylabel="Accuracy",
               title=f"{name}: ECE={m.get('ece', float('nan')):.3f}, Brier={m.get('brier', float('nan')):.3f}")
        hx.bar(centres, counts, width=width * 0.9, color="#6b7280")
        hx.set_yscale("symlog", linthresh=10)
        hx.set(xlim=(lo, 1), xlabel="Predicted P(correct)", ylabel="Cases (log)")
    # One shared legend between the title and the panels, clear of the bars.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(title, y=0.995)
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def plot_risk_coverage(rc: dict, op_coverage: float | None, dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cov, risk = rc["coverage"], rc["risk"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(cov, risk, color="#1d4ed8", lw=2, label="Error rate among accepted")
    # Coverage reached before the first accepted error (most confident first).
    zero = max((c for c, r in zip(cov, risk) if r == 0), default=0.0)
    if op_coverage:
        ax.axvline(op_coverage, color="#166534", ls="--", label=f"Auto-accept cut-off: {op_coverage:.1%} of cases")
    if cov and cov[-1] < 0.999:
        ax.axvspan(cov[-1], 1, color="#fee2e2", label=f"No valid label/score: {1 - cov[-1]:.1%} (safe stop)")
    # Zoom when the curve is flat at zero for most of the range.
    lo = max(0.0, math.floor((min(zero, op_coverage or 1) - 0.05) * 20) / 20) if zero > 0.3 else 0.0
    top = max(risk + [0.01]) * 1.25
    if zero > 0:
        ax.annotate(f"0 errors in the most confident\n{zero:.1%} of cases", xy=(zero, 0), xytext=(lo + 0.01, top * 0.55),
                    arrowprops={"arrowstyle": "->", "color": "#374151"}, fontsize=9)
    ax.set(xlabel="Coverage (share of all cases accepted, most confident first)",
           ylabel="Error rate among accepted", xlim=(lo, 1), ylim=(0, top),
           title=f"Risk-coverage on evaluation split (AURC={rc['aurc']:.4f})"
                 + (f" — x-axis starts at {lo:.0%}" if lo > 0 else ""))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def plot_subgroups(rows: list[dict], overall: float, dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{r['dimension'].replace('_', ' ')}: {r['group']} (n={r['n']})" for r in rows]
    y = list(range(len(rows)))[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.5))
    for yi, r in zip(y, rows):
        color = "#9ca3af" if r["small_sample"] else "#1d4ed8"
        ax.errorbar(r["accuracy"], yi, xerr=[[r["accuracy"] - r["accuracy_ci_low"]], [r["accuracy_ci_high"] - r["accuracy"]]],
                    fmt="o", color=color, capsize=3)
    ax.axvline(overall, color="#111827", ls="--", lw=1)
    ax.set_yticks(y, labels)
    ax.set(xlabel="Accuracy with Wilson 95% CI" + (" (grey: small sample)" if any(r["small_sample"] for r in rows) else ""),
           title=f"Subgroup accuracy (evaluation split) — dashed line: overall {overall:.1%}")
    fig.tight_layout()
    fig.savefig(dest, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- report
def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def write_report(dest: Path, m: dict) -> None:
    z = m["zones"]
    p = m["policy"]
    lines = [
        f"# FedDx trust evaluation — {m['source']} score",
        "",
        f"Calibration split: {m['calibration']['n']} cases · Evaluation split: {m['evaluation']['n']} cases "
        f"({m['split_method']}).",
        "",
        "## Calibration (evaluation split)",
        "",
        "| | ECE | Brier | AUROC (correct vs wrong) | Mean confidence | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, key in (("Raw score", "raw"), ("Platt-calibrated", "calibrated")):
        c = m["calibration_metrics"][key]
        auroc = "n/a" if c.get("auroc_correctness") is None else f"{c['auroc_correctness']:.3f}"
        lines.append(f"| {name} | {c.get('ece', float('nan')):.3f} | {c.get('brier', float('nan')):.3f} | "
                     f"{auroc} | {pct(c.get('mean_confidence'))} | {pct(c.get('accuracy'))} |")
    accept = p["accept_threshold"]
    lines += [
        "",
        "## Deferral policy (fitted on calibration split)",
        "",
        f"- Auto-accept: calibrated P(correct) ≥ {pct(accept) if accept is not None else 'none — no threshold met the target'}"
        f" (target: Wilson 95% lower bound ≥ {pct(p['target_accuracy'])})",
        f"- Safe stop: calibrated P(correct) < {pct(p['stop_threshold'])}, invalid label, or no score",
        "- Review: everything in between",
        "",
        "## Zones on evaluation split",
        "",
        "| Zone | Cases | Share | Accuracy (95% CI) | Errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for zone in ZONES:
        q = z[zone]
        ci = q["accuracy_ci95"]
        ci_txt = "n/a" if q["accuracy"] is None else f"{pct(q['accuracy'])} ({pct(ci[0])}–{pct(ci[1])})"
        lines.append(f"| {zone} | {q['n']} | {pct(q['share'])} | {ci_txt} | {q['errors']} |")
    lines += [
        "",
        f"**Silent errors** (wrong but auto-accepted): {z['AUTO_ACCEPT']['errors']} of {m['evaluation']['n']} "
        f"· **Clinician review workload:** {pct(z['REVIEW']['share'])} · "
        f"**Errors caught by review or safe stop:** {m['errors_caught']} of {m['errors_total']}",
        "",
        "## Subgroups (evaluation split)",
        "",
        "| Group | n | Accuracy (95% CI) | Auto-accept | Review | Safe stop | Silent errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in m["subgroups"]:
        flag = " ⚠ small n" if r["small_sample"] else ""
        lines.append(f"| {r['dimension']}: {r['group']}{flag} | {r['n']} | {pct(r['accuracy'])} "
                     f"({pct(r['accuracy_ci_low'])}–{pct(r['accuracy_ci_high'])}) | {pct(r['auto_accept_rate'])} | "
                     f"{pct(r['review_rate'])} | {pct(r['safe_stop_rate'])} | {r['silent_errors']} |")
    lines += ["", "Disparity summary:", ""]
    for dim, d in m["disparity"].items():
        if "note" in d:
            lines.append(f"- {dim}: {d['note']}")
        else:
            gpi = d["auto_accept_parity_index_min_over_max"]
            lines.append(f"- {dim}: accuracy gap {100 * d['accuracy_gap_max_minus_min']:.1f} pp; "
                         f"auto-accept parity index {'n/a' if gpi is None else f'{gpi:.2f}'} (1.00 = equal)")
    if m["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in m["warnings"]]
    lines += ["", "_Synthetic DDXPlus data with LLM-generated summaries. Not clinical evidence._", ""]
    (dest / "trust_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib", type=Path, required=True, help="Predictions used to fit calibration and thresholds")
    p.add_argument("--eval", type=Path, default=None, help="Held-out predictions; omit to split --calib")
    p.add_argument("--source", choices=("auto", "token", "closed_set"), default="auto")
    p.add_argument("--data", type=Path, nargs="+", default=[Path("data/generated_clinical_v2/validate.jsonl"),
                                                            Path("data/generated_clinical_v2/test.jsonl")],
                   help="Case files providing patient_features (age, sex)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target-accuracy", type=float, default=0.95,
                   help="Wilson 95%% lower bound required for auto-accepted calibration cases")
    p.add_argument("--stop-threshold", type=float, default=0.5)
    p.add_argument("--min-accept-n", type=int, default=30)
    p.add_argument("--min-group-n", type=int, default=30)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--calib-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not 0 < args.target_accuracy < 1 or not 0 <= args.stop_threshold < 1 or not 0 < args.calib_fraction < 1:
        p.error("Probabilities must lie in (0, 1)")

    warnings: list[str] = []
    calib, source = load_predictions(args.calib, args.source)
    if args.eval is not None:
        evaluation, eval_source = load_predictions(args.eval, args.source)
        if eval_source != source:
            p.error(f"--calib is {source} but --eval is {eval_source}")
        overlap = {r["patient_id"] for r in calib} & {r["patient_id"] for r in evaluation}
        if overlap:
            p.error(f"{len(overlap)} patients appear in both calibration and evaluation files")
        split_method = f"separate files: {args.calib} → {args.eval}"
    else:
        calib, evaluation = split_records(calib, args.calib_fraction, args.seed)
        split_method = f"deterministic hash split of {args.calib} (calibration fraction {args.calib_fraction}, seed {args.seed})"
        warnings.append("Calibration and evaluation come from one file split in two; prefer a separate --eval file.")
    if not evaluation or not calib:
        p.error("Calibration or evaluation split is empty")

    calibrator, warn = fit_platt(calib)
    if warn:
        warnings.append(warn)
    accept = choose_accept_threshold(calib, calibrator, args.target_accuracy, args.min_accept_n)
    stop = args.stop_threshold
    if accept["threshold"] is not None and stop >= accept["threshold"]:
        warnings.append(f"stop-threshold {stop} ≥ accept threshold {accept['threshold']:.3f}; lowered to the accept threshold.")
        stop = accept["threshold"]
    if accept["threshold"] is None:
        warnings.append(accept["reason"] + " Every scored case goes to review or safe stop.")
    policy = {
        "format_version": 1, "score": source,
        "score_definition": ("geometric_mean_token_probability" if source == "token"
                             else "softmax_top3_mean_logprob"),
        "calibrator": calibrator, "accept_threshold": accept["threshold"], "stop_threshold": stop,
        "target_accuracy": args.target_accuracy, "min_accept_n": args.min_accept_n,
        "accept_fit": accept, "fitted_on": str(args.calib), "fitted_on_n": len(calib),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    for r in evaluation:
        zone, cal = assign_zone(policy, r["score"], r["valid"])
        r["zone"], r["calibrated"] = zone, cal
    scored = [r for r in evaluation if r["score"] is not None]
    if len(scored) < len(evaluation):
        warnings.append(f"{len(evaluation) - len(scored)} evaluation cases had an invalid label or no score (→ safe stop).")
    correct = [r["correct"] for r in scored]
    raw_m = calibration_metrics([r["score"] for r in scored], correct, args.bins)
    cal_m = calibration_metrics([r["calibrated"] for r in scored], correct, args.bins)
    if raw_m.get("n") and min(sum(correct), len(correct) - sum(correct)) < 10:
        warnings.append(f"Evaluation split has only {len(correct) - sum(correct)} errors among scored cases; "
                        "ECE and the reliability diagram rest on very few failures.")
    zones = zone_table(evaluation)
    rc = risk_coverage(evaluation)
    features = load_features(args.data)
    missing = sum(r["patient_id"] not in features for r in evaluation)
    if missing:
        warnings.append(f"{missing} evaluation cases lack patient_features (grouped as 'unknown'); pass --data.")
    sub_rows, disparity = subgroup_table(evaluation, features, args.min_group_n)

    errors_total = sum(not r["correct"] for r in evaluation)
    metrics = {
        "source": source, "split_method": split_method,
        "calibration": {"n": len(calib), "accuracy": sum(r["correct"] for r in calib) / len(calib)},
        "evaluation": {"n": len(evaluation), "accuracy": sum(r["correct"] for r in evaluation) / len(evaluation),
                       "scored_n": len(scored)},
        "policy": policy, "calibration_metrics": {"raw": raw_m, "calibrated": cal_m},
        "zones": zones, "errors_total": errors_total,
        "errors_caught": errors_total - zones["AUTO_ACCEPT"]["errors"],
        "aurc": rc["aurc"], "subgroups": sub_rows, "disparity": disparity, "warnings": warnings,
    }

    dest = args.output_dir
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "deferral_policy.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    (dest / "trust_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (dest / "zones.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "truth", "prediction", "correct", "raw_score", "calibrated", "zone"])
        for r in evaluation:
            w.writerow([r["patient_id"], r["truth"], r["prediction"], r["correct"], r["score"], r["calibrated"], r["zone"]])
    with (dest / "subgroups.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sub_rows[0]))
        w.writeheader()
        w.writerows(sub_rows)
    if scored:
        plot_reliability([r["score"] for r in scored], [r["calibrated"] for r in scored], correct,
                         raw_m, cal_m, dest / "reliability_diagram.png",
                         f"FedDx reliability ({source} score, evaluation split n={len(scored)})", args.bins)
        plot_risk_coverage(rc, zones["AUTO_ACCEPT"]["share"], dest / "risk_coverage.png")
    plot_subgroups(sub_rows, metrics["evaluation"]["accuracy"], dest / "subgroups.png")
    write_report(dest, metrics)

    print(f"Source: {source} | calibration n={len(calib)} | evaluation n={len(evaluation)}")
    print(f"ECE raw={raw_m.get('ece', float('nan')):.4f} → calibrated={cal_m.get('ece', float('nan')):.4f}; "
          f"Brier raw={raw_m.get('brier', float('nan')):.4f} → calibrated={cal_m.get('brier', float('nan')):.4f}")
    print(f"Accept threshold={accept['threshold']} stop threshold={stop}")
    for zone in ZONES:
        q = zones[zone]
        print(f"  {zone:11s} n={q['n']:5d} share={pct(q['share']):>6s} accuracy={pct(q['accuracy']):>6s} errors={q['errors']}")
    print(f"Errors caught before auto-accept: {metrics['errors_caught']}/{errors_total}")
    for w in warnings:
        print("WARNING:", w)
    print(f"Wrote {dest}/trust_report.md and figures")


if __name__ == "__main__":
    main()