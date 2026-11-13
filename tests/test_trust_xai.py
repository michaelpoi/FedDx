"""CPU tests for FedDx calibration, deferral, SHAP chunking and the deletion test.

Run from the FedDx root:  uv run pytest -q tests
No GPU, model or `shap` package needed: SHAP is replaced by exact Shapley
values computed on a small fake model.
"""
import itertools
import json
import math
import random
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate_trust as et  # noqa: E402
import shap_faithfulness as sf  # noqa: E402
from feddx_deferral import assign_zone, calibrate  # noqa: E402
from feddx_shap import split_clinical_text  # noqa: E402


# ---------------------------------------------------------------- chunking
def test_chunker_splits_real_summary_format_on_clauses():
    s = ("Patient: 55-year-old male.\nPresenting findings: Chest pain that is sharp, "
         "worse when lying down.\nRelevant history: Smoker; prior heart attack.")
    chunks = split_clinical_text(s, 12)
    assert "".join(chunks) == s
    assert chunks[0] == "Patient: 55-year-old male.\n"
    assert any(c.startswith("Presenting findings: Chest pain that is sharp") for c in chunks)
    assert len(chunks) == 5


def test_chunker_merges_to_max_features():
    s = ", ".join(f"finding {i}" for i in range(30)) + "."
    chunks = split_clinical_text(s, 8)
    assert "".join(chunks) == s and len(chunks) <= 8


# ---------------------------------------------------------------- statistics
def test_wilson_interval():
    lo, hi = et.wilson(95, 100)
    assert 0.88 < lo < 0.90 and 0.97 < hi < 0.99
    assert et.wilson(0, 0) == (None, None)


def test_ece_perfect_and_overconfident():
    probs = [0.8] * 10
    perfect = et.calibration_metrics(probs, [True] * 8 + [False] * 2, 10)
    over = et.calibration_metrics([0.99] * 10, [True] * 5 + [False] * 5, 10)
    assert perfect["ece"] == pytest.approx(0.0, abs=1e-9)
    assert over["ece"] == pytest.approx(0.49, abs=1e-9)


def _synthetic(n, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        q = rng.betavariate(5, 1.2)
        correct = rng.random() < q
        rows.append({"patient_id": f"p{seed}_{i}", "truth": "A", "prediction": "A" if correct else "B",
                     "valid": True, "correct": correct, "score": min(0.999, q ** 0.3)})
    return rows


def test_platt_reduces_ece_on_overconfident_scores():
    calib, evaluation = _synthetic(800, 1), _synthetic(800, 2)
    calibrator, _ = et.fit_platt(calib)
    assert calibrator["type"] == "platt"
    correct = [r["correct"] for r in evaluation]
    raw = et.calibration_metrics([r["score"] for r in evaluation], correct, 10)["ece"]
    cal = et.calibration_metrics([calibrate(calibrator, r["score"]) for r in evaluation], correct, 10)["ece"]
    assert cal < raw / 2


def test_platt_falls_back_without_errors():
    rows = [{"score": 0.9, "correct": True} for _ in range(50)]
    calibrator, warning = et.fit_platt(rows)
    assert calibrator == {"type": "identity"} and warning


def test_accept_threshold_meets_target_on_calibration_split():
    calib = _synthetic(1000, 3)
    calibrator, _ = et.fit_platt(calib)
    fit = et.choose_accept_threshold(calib, calibrator, target=0.85, min_n=30)
    assert fit["threshold"] is not None and fit["wilson_lower"] >= 0.85 and fit["accepted_n"] >= 30


def test_zones_invalid_and_unscored_go_to_safe_stop():
    policy = {"calibrator": {"type": "identity"}, "accept_threshold": 0.9, "stop_threshold": 0.5}
    assert assign_zone(policy, 0.95, True)[0] == "AUTO_ACCEPT"
    assert assign_zone(policy, 0.7, True)[0] == "REVIEW"
    assert assign_zone(policy, 0.3, True)[0] == "SAFE_STOP"
    assert assign_zone(policy, 0.99, False)[0] == "SAFE_STOP"
    assert assign_zone(policy, None, True)[0] == "SAFE_STOP"
    policy["accept_threshold"] = None
    assert assign_zone(policy, 0.99, True)[0] == "REVIEW"


def test_evaluate_trust_end_to_end(tmp_path, monkeypatch):
    rng = random.Random(0)
    for split in ("validate", "test"):
        cases, preds = [], []
        for i in range(400):
            pid = f"{split}_{i}"
            q = rng.betavariate(5, 1.2)
            pred = "A" if rng.random() < q else "B"
            cases.append({"patient_id": pid, "patient_features": {"age": rng.randint(1, 90), "sex": rng.choice("MF")}})
            preds.append({"patient_id": pid, "truth": "A", "prediction": pred, "correct": pred == "A",
                          "confidence": {"available": True, "geometric_mean_token_probability": min(0.999, q ** 0.3)}})
        (tmp_path / f"{split}_cases.jsonl").write_text("\n".join(map(json.dumps, cases)))
        (tmp_path / f"{split}_preds.jsonl").write_text("\n".join(map(json.dumps, preds)))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["evaluate_trust.py", "--calib", str(tmp_path / "validate_preds.jsonl"),
                                      "--eval", str(tmp_path / "test_preds.jsonl"), "--output-dir", str(out),
                                      "--target-accuracy", "0.8", "--data", str(tmp_path / "validate_cases.jsonl"),
                                      str(tmp_path / "test_cases.jsonl")])
    et.main()
    metrics = json.loads((out / "trust_metrics.json").read_text())
    assert sum(z["n"] for z in metrics["zones"].values()) == 400
    assert {r["group"] for r in metrics["subgroups"] if r["dimension"] == "sex"} == {"M", "F"}
    for name in ("deferral_policy.json", "reliability_diagram.png", "risk_coverage.png", "subgroups.png", "trust_report.md"):
        assert (out / name).is_file()


# ---------------------------------------------------------------- deletion test
WEIGHTS = {"chest pain": 3.0, "sweating": 1.5, "smoker": 1.0, "cough": -1.0}


def fake_ll(model, proc, prompt, summary, diagnosis):
    return -6 + sum(w for k, w in WEIGHTS.items() if k in summary)


def exact_shap(model, proc, prompt, summary, diagnosis, max_features, permutations, seed, cache):
    chunks = split_clinical_text(summary, max_features=max_features)
    n = len(chunks)

    def v(mask):
        if mask not in cache:
            cache[mask] = fake_ll(None, None, prompt, "".join(c for c, k in zip(chunks, mask) if k), diagnosis)
        return cache[mask]
    phi = [0.0] * n
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            for subset in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                without = tuple(1 if j in subset else 0 for j in range(n))
                with_i = tuple(1 if (j in subset or j == i) else 0 for j in range(n))
                phi[i] += w * (v(with_i) - v(without))
    base, full = v(tuple([0] * n)), v(tuple([1] * n))
    return {"chunks": chunks, "shap_values": phi, "baseline_log_likelihood": base,
            "full_log_likelihood": full, "additivity_error": abs(base + sum(phi) - full)}


def fake_generate(model, proc, summary, labels, max_new_tokens):
    return ("Unstable angina" if "chest pain" in summary else "URTI"), "raw"


def test_deletion_test_detects_faithful_attributions():
    args = types.SimpleNamespace(max_features=12, permutations=2, seed=42, k_max=2, random_repeats=3,
                                 flip_test=True, max_new_tokens=48)
    rng = random.Random(1)
    fillers = ["mild fatigue", "no fever", "normal appetite", "works nights"]
    cases = []
    for i in range(8):
        parts = rng.sample(list(WEIGHTS) + fillers, 6)
        summary = "Patient: 55-year-old male.\nPresenting findings: " + ", ".join(parts) + "."
        record = {"patient_id": f"t{i}", "clinical_summary": summary, "target": {"diagnosis": "Unstable angina"}}
        cases.append(sf.run_case(None, None, record, ["Unstable angina", "URTI"], args,
                                 explain_fn=exact_shap, loglik_fn=fake_ll, generate_fn=fake_generate))
    summary = sf.summarize(cases, 42)
    k1 = summary["per_k"]["1"]
    assert summary["cases_scored"] == 8
    assert k1["mean_drop_shap_top"] > k1["mean_drop_random"] > k1["mean_drop_shap_bottom"]
    assert k1["flip_rate_shap_top"] >= k1["flip_rate_random"]
    assert summary["median_additivity_error"] < 1e-9
