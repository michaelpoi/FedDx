#!/usr/bin/env python3
"""Deletion (comprehensiveness) test of FedDx phrase-level SHAP explanations.

Question: when SHAP says a clause supports the predicted diagnosis, does
removing that clause actually lower the model's likelihood of that diagnosis
more than removing clauses at random?

For each sampled case:
  1. Greedy-generate the diagnosis exactly as the Gradio app does.
  2. Run feddx_shap.explain_summary on that prediction.
  3. For k = 1..K remove
       - the k clauses with the HIGHEST SHAP value   (should hurt most)
       - k RANDOM clauses (averaged over R draws)    (reference)
       - the k clauses with the LOWEST SHAP value    (sanity: should hurt least / help)
     and record the drop in log P(predicted diagnosis):
        drop = log P(full summary) - log P(summary without those clauses)
  4. Optionally re-generate after removing the top-k SHAP clauses and k random
     clauses, and record whether the predicted label changes (flip rate).

SHAP is faithful to the model (in this sense) if top-k drops are clearly larger
than random drops. This measures explanation fidelity, not clinical validity.

Run from the FedDx root (GPU):
  uv run python src/shap_faithfulness.py --model-id 5000 \
      --adapter-root runs/gemma4_qlora_independent --n 20

Reports only (CPU, from saved cases):
  uv run python src/shap_faithfulness.py --model-id 5000 --n 20 --reports-only

Each completed case is appended to cases.jsonl; reruns resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

from evaluate_gemma4_diagnoses import INVALID, predicted_class
from feddx_shap import diagnosis_log_likelihood, explain_summary, split_clinical_text

PROMPT = (
    "You are a clinical diagnosis classifier. Given the clinical summary, "
    "output only the single most likely diagnosis name, without explanation.\n\n"
    "Clinical summary:\n{summary}"
)


# ------------------------------------------------------------------ helpers
def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def json_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def masked_text(chunks: list[str], mask: tuple[int, ...]) -> str:
    return "".join(c for c, keep in zip(chunks, mask) if keep)


def removal_mask(n: int, removed: list[int]) -> tuple[int, ...]:
    gone = set(removed)
    return tuple(0 if i in gone else 1 for i in range(n))


def bootstrap_ci(values: list[float], seed: int, reps: int = 2000) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


# ------------------------------------------------------------------ model
def load_model(args):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (use --reports-only to summarise saved cases)")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_storage=dtype,
        ),
    )
    if args.model_id != "vanilla":
        from peft import PeftModel
        adapter = args.adapter_root / f"stage-{args.model_id}" / "adapter_final"
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Adapter missing: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return model, processor


def generate_label(model, processor, summary: str, labels: list[str], max_new_tokens: int) -> tuple[str, str]:
    """Same generation path as app_gradio.FedDxApp.predict."""
    import torch
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT.format(summary=summary.strip())}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    pad = processor.tokenizer.pad_token_id
    if pad is None:
        pad = processor.tokenizer.eos_token_id
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad)
    raw = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    return predicted_class(raw, labels), raw


# ------------------------------------------------------------------ one case
def run_case(model, processor, record: dict, labels: list[str], args, explain_fn=explain_summary,
             loglik_fn=diagnosis_log_likelihood, generate_fn=generate_label) -> dict:
    pid = record["patient_id"]
    summary = record["clinical_summary"].strip()
    started = time.perf_counter()
    prediction, raw = generate_fn(model, processor, summary, labels, args.max_new_tokens)
    base = {"patient_id": pid, "truth": record["target"]["diagnosis"], "prediction": prediction,
            "raw_response": raw, "correct": prediction == record["target"]["diagnosis"]}
    if prediction == INVALID:
        return {**base, "status": "skipped", "reason": "invalid / out-of-vocabulary prediction"}
    chunks = split_clinical_text(summary, max_features=args.max_features)
    n = len(chunks)
    if n < 3:
        return {**base, "status": "skipped", "reason": f"only {n} text chunks"}

    cache: dict[tuple[int, ...], float] = {}
    shap_result = explain_fn(model, processor, PROMPT, summary, prediction,
                             max_features=args.max_features, permutations=args.permutations,
                             seed=args.seed, cache=cache)
    if shap_result["chunks"] != chunks:
        raise RuntimeError("SHAP chunking differs from deletion chunking")
    phi = [float(v) for v in shap_result["shap_values"]]

    def ll(mask: tuple[int, ...]) -> float:
        if mask not in cache:
            cache[mask] = loglik_fn(model, processor, PROMPT, masked_text(chunks, mask), prediction)
        return cache[mask]

    full = ll(tuple([1] * n))
    top_order = sorted(range(n), key=lambda i: phi[i], reverse=True)
    bottom_order = sorted(range(n), key=lambda i: phi[i])
    rng = random.Random(f"{args.seed}:{pid}")
    k_max = min(args.k_max, n - 1)
    deletion, flips = {}, {}
    for k in range(1, k_max + 1):
        draws = [sorted(rng.sample(range(n), k)) for _ in range(args.random_repeats)]
        random_drops = [full - ll(removal_mask(n, d)) for d in draws]
        deletion[str(k)] = {
            "shap_top": full - ll(removal_mask(n, top_order[:k])),
            "shap_bottom": full - ll(removal_mask(n, bottom_order[:k])),
            "random_mean": statistics.fmean(random_drops),
            "random_draws": random_drops,
            "removed_top_chunks": [chunks[i].strip() for i in top_order[:k]],
        }
        if args.flip_test:
            top_label, _ = generate_fn(model, processor, masked_text(chunks, removal_mask(n, top_order[:k])),
                                       labels, args.max_new_tokens)
            rand_label, _ = generate_fn(model, processor, masked_text(chunks, removal_mask(n, draws[0])),
                                        labels, args.max_new_tokens)
            flips[str(k)] = {"shap_top_flipped": top_label != prediction, "shap_top_new_label": top_label,
                             "random_flipped": rand_label != prediction, "random_new_label": rand_label}
    return {
        **base, "status": "ok", "n_chunks": n, "chunks": chunks, "shap_values": phi,
        "full_log_likelihood": full, "baseline_log_likelihood": shap_result["baseline_log_likelihood"],
        "additivity_error": shap_result["additivity_error"],
        "top_chunk": chunks[top_order[0]].strip(),
        "demographics_in_top3": any(chunks[i].lstrip().startswith("Patient:") for i in top_order[:3]),
        "deletion": deletion, "flips": flips, "model_evaluations": len(cache),
        "seconds": time.perf_counter() - started,
    }


# ------------------------------------------------------------------ report
def summarize(cases: list[dict], seed: int) -> dict:
    ok = [c for c in cases if c["status"] == "ok"]
    out = {"cases_total": len(cases), "cases_scored": len(ok),
           "skipped": [{"patient_id": c["patient_id"], "reason": c["reason"]} for c in cases if c["status"] != "ok"]}
    if not ok:
        return out
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    per_k = {}
    ks = sorted({int(k) for c in ok for k in c["deletion"]})
    for k in ks:
        rows = [c for c in ok if str(k) in c["deletion"]]
        top = [c["deletion"][str(k)]["shap_top"] for c in rows]
        rnd = [c["deletion"][str(k)]["random_mean"] for c in rows]
        bot = [c["deletion"][str(k)]["shap_bottom"] for c in rows]
        diff = [t - r for t, r in zip(top, rnd)]
        entry = {
            "n": len(rows),
            "mean_drop_shap_top": statistics.fmean(top), "mean_drop_random": statistics.fmean(rnd),
            "mean_drop_shap_bottom": statistics.fmean(bot),
            "mean_paired_difference_top_minus_random": statistics.fmean(diff),
            "difference_ci95_bootstrap": bootstrap_ci(diff, seed + k) if len(diff) > 1 else None,
            "win_rate_top_beats_random": sum(d > 0 for d in diff) / len(diff),
            "wilcoxon_p_one_sided": None,
        }
        if wilcoxon is not None and len(diff) >= 6 and any(d != 0 for d in diff):
            entry["wilcoxon_p_one_sided"] = float(wilcoxon(top, rnd, alternative="greater").pvalue)
        flipped = [c["flips"][str(k)] for c in rows if str(k) in c.get("flips", {})]
        if flipped:
            entry["flip_rate_shap_top"] = sum(f["shap_top_flipped"] for f in flipped) / len(flipped)
            entry["flip_rate_random"] = sum(f["random_flipped"] for f in flipped) / len(flipped)
        per_k[str(k)] = entry
    out.update({
        "per_k": per_k,
        "accuracy_on_scored": sum(c["correct"] for c in ok) / len(ok),
        "median_additivity_error": statistics.median(c["additivity_error"] for c in ok),
        "mean_seconds_per_case": statistics.fmean(c["seconds"] for c in ok),
        "mean_model_evaluations": statistics.fmean(c["model_evaluations"] for c in ok),
        "mean_chunks": statistics.fmean(c["n_chunks"] for c in ok),
        "share_demographics_in_top3": sum(c["demographics_in_top3"] for c in ok) / len(ok),
    })
    return out


def plot(summary: dict, dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = sorted(summary["per_k"], key=int)
    x = [int(k) for k in ks]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for key, label, color in (("mean_drop_shap_top", "Remove top-k SHAP clauses", "#dc2626"),
                              ("mean_drop_random", "Remove k random clauses", "#6b7280"),
                              ("mean_drop_shap_bottom", "Remove bottom-k SHAP clauses", "#2563eb")):
        ax.plot(x, [summary["per_k"][k][key] for k in ks], marker="o", color=color, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xticks=x, xlabel="Clauses removed (k)",
           ylabel="Drop in log P(predicted diagnosis)",
           title=f"SHAP deletion test (n={summary['cases_scored']} cases)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def write_report(summary: dict, dest: Path, args) -> None:
    lines = [f"# SHAP faithfulness — deletion test ({args.model_id}, {args.split})", "",
             f"Cases scored: {summary['cases_scored']} / {summary['cases_total']} · "
             f"max chunks {args.max_features} · permutations {args.permutations} · random draws {args.random_repeats}", ""]
    if summary["cases_scored"]:
        lines += ["| k | Drop: top-k SHAP | Drop: random k | Drop: bottom-k SHAP | Top − random (95% CI) | Top beats random | Wilcoxon p | Flip: SHAP / random |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, e in sorted(summary["per_k"].items(), key=lambda kv: int(kv[0])):
            ci = e["difference_ci95_bootstrap"]
            ci_txt = f"{e['mean_paired_difference_top_minus_random']:.2f} ({ci[0]:.2f}, {ci[1]:.2f})" if ci else "n/a"
            p = "n/a" if e["wilcoxon_p_one_sided"] is None else f"{e['wilcoxon_p_one_sided']:.3g}"
            flip = (f"{100 * e['flip_rate_shap_top']:.0f}% / {100 * e['flip_rate_random']:.0f}%"
                    if "flip_rate_shap_top" in e else "n/a")
            lines.append(f"| {k} | {e['mean_drop_shap_top']:.2f} | {e['mean_drop_random']:.2f} | "
                         f"{e['mean_drop_shap_bottom']:.2f} | {ci_txt} | {100 * e['win_rate_top_beats_random']:.0f}% | {p} | {flip} |")
        lines += ["", "Drops are in log-likelihood units (natural log) of the fixed predicted diagnosis; "
                      "larger = removing the clauses made the model less likely to output that diagnosis.", "",
                  f"- Median SHAP additivity error: {summary['median_additivity_error']:.3g}",
                  f"- Runtime: {summary['mean_seconds_per_case']:.1f} s/case, "
                  f"{summary['mean_model_evaluations']:.0f} forward passes/case, {summary['mean_chunks']:.1f} chunks/case",
                  f"- Demographic line among top-3 SHAP clauses: {100 * summary['share_demographics_in_top3']:.0f}% of cases",
                  f"- Accuracy on scored cases: {100 * summary['accuracy_on_scored']:.0f}%"]
    if summary["skipped"]:
        lines += ["", f"Skipped {len(summary['skipped'])} cases: " +
                  "; ".join(f"{s['patient_id']} ({s['reason']})" for s in summary["skipped"])]
    lines += ["", "_Fidelity to the model on synthetic DDXPlus summaries, not clinical importance or causality._", ""]
    (dest / "faithfulness_report.md").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", choices=("vanilla", "1000", "5000"), required=True)
    p.add_argument("--adapter-root", type=Path, default=Path("runs/gemma4_qlora"))
    p.add_argument("--split", choices=("test", "validate"), default="test")
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--train-data", type=Path, default=Path("data/generated_clinical_v2/train.jsonl"))
    p.add_argument("--model", default="google/gemma-4-12B-it")
    p.add_argument("--revision", default=None)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-features", type=int, default=12)
    p.add_argument("--permutations", type=int, default=2)
    p.add_argument("--k-max", type=int, default=3)
    p.add_argument("--random-repeats", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--no-flip-test", dest="flip_test", action="store_false",
                   help="Skip re-generation after deletion (saves 2*k-max generations per case)")
    p.add_argument("--output-root", type=Path, default=Path("runs/shap_faithfulness"))
    p.add_argument("--reports-only", action="store_true")
    args = p.parse_args()
    if args.data is None:
        args.data = Path(f"data/generated_clinical_v2/{args.split}.jsonl")
    if args.k_max < 1 or args.random_repeats < 1 or args.n < 1:
        p.error("--k-max, --random-repeats and --n must be positive")

    rows = read_jsonl(args.data)
    selected = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    labels = sorted({r["target"]["diagnosis"] for r in read_jsonl(args.train_data) + rows})
    tag = "vanilla" if args.model_id == "vanilla" else f"stage-{args.model_id}"
    dest = args.output_root / args.split / tag / f"n{args.n}_seed{args.seed}"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1, "model": args.model, "revision": args.revision, "model_id": args.model_id,
        "adapter": None if args.model_id == "vanilla" else str(args.adapter_root / f"stage-{args.model_id}" / "adapter_final"),
        "data": str(args.data), "selected_hash": json_hash([[r["patient_id"], r["clinical_summary"]] for r in selected]),
        "prompt": PROMPT, "max_features": args.max_features, "permutations": args.permutations,
        "k_max": args.k_max, "random_repeats": args.random_repeats, "flip_test": args.flip_test, "seed": args.seed,
    }
    mpath = dest / "manifest.json"
    if mpath.exists():
        if json.loads(mpath.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"Settings changed since {mpath} was written; use a different --output-root")
    else:
        mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    cases_path = dest / "cases.jsonl"
    done = {c["patient_id"]: c for c in read_jsonl(cases_path)} if cases_path.exists() else {}
    pending = [r for r in selected if r["patient_id"] not in done]
    print(f"{len(done)}/{len(selected)} cases done; {len(pending)} pending", flush=True)
    if pending and not args.reports_only:
        model, processor = load_model(args)
        with cases_path.open("a", encoding="utf-8") as f:
            for i, record in enumerate(pending, 1):
                result = run_case(model, processor, record, labels, args)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                done[record["patient_id"]] = result
                k1 = result.get("deletion", {}).get("1", {})
                print(f"[{i}/{len(pending)}] {record['patient_id']} {result['status']} "
                      f"pred={result['prediction']!r} drop@1 top={k1.get('shap_top', float('nan')):.2f} "
                      f"rand={k1.get('random_mean', float('nan')):.2f} ({result.get('seconds', 0):.0f}s)", flush=True)
    cases = [done[r["patient_id"]] for r in selected if r["patient_id"] in done]
    summary = summarize(cases, args.seed)
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, dest, args)
    if summary["cases_scored"]:
        plot(summary, dest / "deletion_curve.png")
    print((dest / "faithfulness_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
