#!/usr/bin/env python3
"""FedDx case explorer: original evidence, generated summary, and Gemma predictions.

Run from the FedDx project root:
    uv add gradio
    uv run python src/app_gradio.py
    uv run python src/app_gradio.py --deferral-policy runs/trust/token_5k/deferral_policy.json

Live inference additionally requires torch, transformers, peft, accelerate,
including bitsandbytes and access to google/gemma-4-12B-it.

No patient data are sent to an external API; Gradio is bound to 127.0.0.1
by default. This is a research demonstration, not clinical software.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import threading
from pathlib import Path
from typing import Any

from feddx_confidence import render_confidence_html, score_generated_diagnosis
from feddx_deferral import assign_zone, load_policy, render_zone_html
from feddx_shap import explain_summary, render_shap_html, attribution_rows

PROMPT = (
    "You are a clinical diagnosis classifier. Given the clinical summary, "
    "output only the single most likely diagnosis name, without explanation.\n\n"
    "Clinical summary:\n{summary}"
)

MODEL_CHOICES = {
    "Vanilla Gemma 4 12B": None,
    "QLoRA 1k": Path("runs/gemma4_qlora/stage-1000/adapter_final"),
    "QLoRA 5k (independent)": Path("runs/gemma4_qlora_independent/stage-5000/adapter_final"),
}

CACHE_CHOICES = {
    "Vanilla Gemma 4 12B": [
        Path("runs/gemma4_evaluation_test/test/vanilla/predictions.jsonl"),
        Path("runs/gemma4_evaluation/test/vanilla/predictions.jsonl"),
    ],
    "QLoRA 1k": [
        Path("runs/gemma4_evaluation_test/test/stage-1000/predictions.jsonl"),
        Path("runs/gemma4_evaluation/test/stage-1000/predictions.jsonl"),
    ],
    "QLoRA 5k (independent)": [
        Path("runs/gemma4_evaluation_test/test/stage-5000/predictions.jsonl"),
        Path("runs/gemma4_evaluation_independent/test/stage-5000/predictions.jsonl"),
    ],
}

INVALID = "[INVALID / OUT OF VOCABULARY]"


def read_test_cases(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Test dataset not found: {path}")
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            pid = record["patient_id"]
            if pid in records:
                raise ValueError(f"Duplicate patient ID at {path}:{line_number}: {pid}")
            if not record.get("clinical_summary", "").strip():
                raise ValueError(f"Missing clinical summary at {path}:{line_number}")
            records[pid] = record
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def readable_answer(value: Any) -> str:
    """Preserve the original meaning; don't interpret arbitrary categorical scales."""
    if value is None or value == "":
        return "Not recorded"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        return "; ".join(readable_answer(v) for v in value) or "Not recorded"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    s = str(value).strip()
    return {"y": "Yes", "n": "No", "yes": "Yes", "no": "No", "true": "Yes", "false": "No"}.get(s.casefold(), s)


def evidence_rows(record: dict[str, Any]) -> list[list[str]]:
    rows = []
    for evidence in record.get("clinical_evidence", []):
        kind = str(evidence.get("type", "unspecified")).replace("_", " ").title()
        question = str(evidence.get("question") or "Question unavailable")
        answer = readable_answer(evidence.get("answer"))
        code = str(evidence.get("evidence_code") or "")
        rows.append([kind, question, answer, code])
    return rows


def clean_response(raw: str) -> str:
    """Remove only a known leading decoded channel name, not clinical content."""
    text = raw.strip()
    text = re.sub(r"^thought\s*\n", "", text, count=1, flags=re.IGNORECASE)
    return text.strip()


def parse_diagnosis(raw: str, vocabulary: set[str]) -> tuple[str, bool]:
    text = clean_response(raw)
    first = text.splitlines()[0].strip() if text else ""
    first = re.sub(r"^(?:diagnosis|answer)\s*:\s*", "", first, flags=re.IGNORECASE)
    first = re.sub(r"^(?:the most likely diagnosis is|most likely diagnosis is)\s+", "", first, flags=re.IGNORECASE)
    lookup = {label.casefold().strip(): label for label in vocabulary}
    label = lookup.get(first.casefold().strip(" .\"'`"))
    return (label if label is not None else (first or "(empty response)"), label is not None)


class FedDxApp:
    def __init__(self, cases: dict[str, dict[str, Any]], model_name: str, cache_paths: dict[str, list[Path]],
                 deferral_policy: dict[str, Any] | None = None):
        self.cases = cases
        # Calibrated deferral policy fitted by evaluate_trust.py on the token score.
        self.deferral_policy = deferral_policy
        self.ids = sorted(cases)
        self.vocabulary = {r["target"]["diagnosis"] for r in cases.values()}
        self.model_name = model_name
        self.cache_paths = cache_paths
        self.cache: dict[str, dict[str, dict[str, Any]]] = {}
        self._model = None
        self._processor = None
        self._loaded_choice = None
        self._model_lock = threading.RLock()
        self._rng = random.SystemRandom()
        self._last_prediction: dict[tuple[str, str], tuple[str, str]] = {}

    def load_patient(self, pid: str):
        pid = (pid or "").strip()
        record = self.cases.get(pid)
        if record is None:
            return (
                f"**Patient ID not found:** `{pid}`. Select Random patient or enter an ID from the test split.",
                [], "", "", "", render_confidence_html(None), "", "",
            )
        feats = record.get("patient_features", {})
        age = feats.get("age", "unknown")
        sex = feats.get("sex", "unknown")
        evidence = evidence_rows(record)
        metadata = (
            f"**Patient:** `{pid}` · **Age:** {age} · **Sex (source code):** {sex} "
            f"· **Evidence entries:** {len(evidence)}"
        )
        return (
            metadata, evidence, record["clinical_summary"],
            record["target"]["diagnosis"], "", render_confidence_html(None), "", "",
        )

    def random_patient(self):
        pid = self._rng.choice(self.ids)
        return (pid, *self.load_patient(pid))

    def _cached_predictions(self, choice: str) -> dict[str, dict[str, Any]]:
        if choice in self.cache:
            return self.cache[choice]
        predictions = {}
        for path in self.cache_paths.get(choice, []):
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    pid = row.get("patient_id")
                    if pid in self.cases and row.get("truth") == self.cases[pid]["target"]["diagnosis"]:
                        # Prefer earlier paths in CACHE_CHOICES; don't mix multiple
                        # evaluation sources for the same patient.
                        predictions.setdefault(pid, row)
        self.cache[choice] = predictions
        return predictions

    def _ensure_model(self, choice: str):
        if self._loaded_choice == choice and self._model is not None:
            return
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError("Live model inference requires an NVIDIA CUDA GPU. Cached predictions work without one.")
        adapter = MODEL_CHOICES[choice]
        if adapter is not None and not adapter.joinpath("adapter_config.json").is_file():
            raise FileNotFoundError(f"Adapter missing: {adapter}")
        # Only one quantized Gemma model should occupy GPU memory at a time.
        self._model = None
        self._processor = None
        self._loaded_choice = None
        gc.collect()
        torch.cuda.empty_cache()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        processor = AutoProcessor.from_pretrained(self.model_name)
        processor.tokenizer.padding_side = "left"
        model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_storage=dtype,
            ),
        )
        if adapter is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()
        self._model, self._processor, self._loaded_choice = model, processor, choice

    def predict(self, pid: str, choice: str, use_cache: bool, edited_summary: str):
        pid = (pid or "").strip()
        record = self.cases.get(pid)
        if record is None:
            return "", render_confidence_html(None), "", "Choose a valid test patient first."
        if not choice in MODEL_CHOICES:
            return "", render_confidence_html(None), "", "Unknown model choice."
        # An edited summary must never be silently replaced with the stored one.
        clinical_summary = (edited_summary or "").strip()
        if not clinical_summary:
            return "", render_confidence_html(None), "", "Clinical summary is empty; enter text before predicting."
        original_summary = record["clinical_summary"].strip()
        is_modified = clinical_summary != original_summary
        # Avoid showing an old patient/model prediction as current after errors.
        self._last_prediction.pop((pid, choice), None)
        # A cached prediction corresponds ONLY to the exact original clinical summary.
        # Modified input always requires a new live inference.
        if use_cache and not is_modified:
            cached = self._cached_predictions(choice).get(pid)
            if cached is not None:
                raw = cached.get("raw_response", "")
                prediction, valid = parse_diagnosis(raw, self.vocabulary)
                if valid:
                    self._last_prediction[(pid, choice)] = (prediction, clinical_summary)
                return (
                    prediction,
                    render_confidence_html(None, cached=True),
                    raw,
                    f"**Source:** saved evaluation prediction · **Allowed diagnosis label:** {'yes' if valid else 'no'}. "
                    "No token probabilities were stored in this evaluation cache; disable saved predictions for confidence scoring.",
                )
        with self._model_lock:
            try:
                self._ensure_model(choice)
                import torch
                processor = self._processor
                model = self._model
                text = PROMPT.format(summary=clinical_summary)
                messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                pad = processor.tokenizer.pad_token_id
                if pad is None:
                    pad = processor.tokenizer.eos_token_id
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=48,
                        do_sample=False,
                        pad_token_id=pad,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                completion_ids = generated.sequences[0, inputs["input_ids"].shape[1]:]
                raw = processor.tokenizer.decode(completion_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
                prediction, valid = parse_diagnosis(raw, self.vocabulary)
                if prediction and prediction != "(empty response)":
                    self._last_prediction[(pid, choice)] = (prediction, clinical_summary)
                try:
                    stats = score_generated_diagnosis(
                        processor.tokenizer, generated.sequences, generated.scores,
                        inputs["input_ids"].shape[1],
                    )
                except Exception as score_error:
                    stats = {"available": False, "reason": f"Token scoring failed: {type(score_error).__name__}"}
                zone_html = ""
                if self.deferral_policy is not None:
                    score = stats.get("geometric_mean_token_probability") if stats.get("available") else None
                    zone, calibrated = assign_zone(self.deferral_policy, score, valid)
                    zone_html = render_zone_html(self.deferral_policy, zone, calibrated)
                return (
                    prediction,
                    render_confidence_html(stats) + zone_html,
                    raw,
                    (f"**Source:** live Gemma inference · **Allowed diagnosis label:** {'yes' if valid else 'no'}. "
                     + ("**Input:** edited clinical summary. " if is_modified else "**Input:** original clinical summary. ")
                     + "Confidence panel shows uncalibrated token likelihoods (not clinical correctness)."),
                )
            except Exception as exc:
                return "", render_confidence_html(None), "", f"**Inference error:** `{type(exc).__name__}: {exc}`"


    def explain(self, pid: str, choice: str, displayed_prediction: str,
                max_features: int, permutations: int, edited_summary: str):
        pid = (pid or "").strip()
        if pid not in self.cases or choice not in MODEL_CHOICES:
            return render_shap_html(None), [], "Select a valid patient and model."
        last = self._last_prediction.get((pid, choice))
        summary = (edited_summary or "").strip()
        if not last or displayed_prediction != last[0] or summary != last[1]:
            return render_shap_html(None), [], "Predict this exact clinical summary with the selected model before SHAP."
        prediction = last[0]
        if not 2 <= int(max_features) <= 24 or not 1 <= int(permutations) <= 5:
            return render_shap_html(None), [], "Invalid SHAP settings."
        with self._model_lock:
            try:
                self._ensure_model(choice)
                result = explain_summary(
                    self._model, self._processor, PROMPT,
                    summary, prediction,
                    max_features=int(max_features), permutations=int(permutations),
                )
                return (
                    render_shap_html(result),
                    attribution_rows(result),
                    f"**SHAP for:** {prediction} · **Text chunks:** {len(result['chunks'])} "
                    f"· **Model evaluations:** {result['model_evaluations']} "
                    f"· **Additivity error:** {result['additivity_error']:.3g}. "
                    "Scores are changes in fixed-diagnosis log-likelihood relative "
                    "to an empty-summary reference, not calibrated probabilities or causal effects.",
                )
            except Exception as exc:
                return render_shap_html(None), [], f"**SHAP error:** `{type(exc).__name__}: {exc}`"


def build_ui(controller: FedDxApp):
    import gradio as gr
    with gr.Blocks(
        title="FedDx · Clinical Case Explorer",
        css="""
        .feddx-confidence, .feddx-confidence * {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }
        """,
    ) as app:
        gr.Markdown(
            "# FedDx · Clinical Case Explorer\n"
            "Explore **synthetic DDXPlus patient evidence**, its **LLM-generated clinician-style summary**, "
            "and **Gemma 4 diagnosis predictions**. **Research demonstration only; not clinical advice.**"
        )
        with gr.Row():
            patient_id = gr.Textbox(label="Test patient ID", placeholder="e.g. test_0002432", scale=3)
            view = gr.Button("Load patient", variant="primary", scale=1)
            random_button = gr.Button("Random patient", scale=1)
        identity = gr.Markdown("Choose a patient to begin.")
        with gr.Row():
            with gr.Column(scale=6):
                gr.Markdown("### 1 · Original patient evidence")
                evidence = gr.Dataframe(
                    headers=["Type", "Question", "Recorded answer", "Evidence code"],
                    datatype=["str", "str", "str", "str"],
                    type="array",
                    interactive=False,
                    wrap=True,
                    label="Decoded questionnaire evidence",
                )
                gr.Markdown(
                    "*Answers reflect the synthetic source. A positive question with 'A or B' "
                    "does not establish both alternatives. Uninterpreted numeric values are shown as recorded.*"
                )
            with gr.Column(scale=5):
                gr.Markdown("### 2 · Clinical summary (editable)")
                summary = gr.Textbox(
                    label="clinician-v2 summary — edit to run what-if predictions",
                    lines=13, interactive=True,
                )
                gr.Markdown("*Edits are temporary; the saved dataset remains unchanged. "
                            "Predictions and SHAP use the currently displayed text. "
                            "A modified summary bypasses the prediction cache.*")
                truth = gr.Textbox(label="Dataset reference diagnosis (research label)", interactive=False)
        gr.Markdown("### 3 · Model diagnosis")
        with gr.Row():
            choice = gr.Dropdown(
                choices=list(MODEL_CHOICES),
                value="QLoRA 5k (independent)",
                label="Model",
                scale=3,
            )
            cache_toggle = gr.Checkbox(value=False, label="Use saved test predictions (no confidence scores)", scale=2)
            predict_button = gr.Button("Predict diagnosis", variant="primary", scale=2)
        with gr.Row():
            diagnosis = gr.Textbox(label="Predicted diagnosis", interactive=False, scale=3)
            confidence = gr.HTML(value=render_confidence_html(None), label="Model token confidence", scale=2)
        gr.Markdown("*Confidence is based on generated diagnosis-token likelihoods, not calibrated diagnostic reliability. "
                    "Green/yellow/red are heuristic visualization thresholds.*")
        result_status = gr.Markdown("Choose a model and press Predict diagnosis.")
        with gr.Accordion("Raw decoded model response", open=False):
            raw = gr.Textbox(label="Raw response", lines=4, interactive=False)

        gr.Markdown("### 4 · Explain predicted diagnosis with SHAP")
        gr.Markdown(
            "SHAP estimates how **clauses in the clinical summary** change the model's "
            "log-likelihood of its **predicted diagnosis**. This takes multiple GPU "
            "forward passes, especially with more chunks or permutations. "
            "Red supports the diagnosis; blue opposes it. "
            "The baseline is an empty summary, so interpret these as model behavior, "
            "**not medical evidence quality or causality**."
        )
        with gr.Row():
            shap_features = gr.Slider(4, 24, value=12, step=1, label="Maximum text chunks")
            shap_permutations = gr.Slider(1, 5, value=1, step=1,
                                          label="SHAP permutation repetitions")
            explain_button = gr.Button("Explain with SHAP", variant="secondary")
        shap_html = gr.HTML(value=render_shap_html(None), label="SHAP highlight")
        shap_status = gr.Markdown("Run a prediction first, then Explain with SHAP.")
        shap_table = gr.Dataframe(
            headers=["Clinical summary fragment", "SHAP (log-likelihood)", "Effect"],
            datatype=["str", "str", "str"], type="array", interactive=False,
            wrap=True, label="SHAP attribution ranking")

        patient_outputs = [identity, evidence, summary, truth, diagnosis, confidence, raw, result_status]
        view.click(controller.load_patient, inputs=patient_id, outputs=patient_outputs)
        patient_id.submit(controller.load_patient, inputs=patient_id, outputs=patient_outputs)
        random_button.click(controller.random_patient, outputs=[patient_id, *patient_outputs])
        predict_button.click(controller.predict, inputs=[patient_id, choice, cache_toggle, summary],
                             outputs=[diagnosis, confidence, raw, result_status])
        choice.change(lambda: ("", render_confidence_html(None), "", "Model changed; press Predict diagnosis."),
                      outputs=[diagnosis, confidence, raw, result_status])
        explain_button.click(
            controller.explain,
            inputs=[patient_id, choice, diagnosis, shap_features, shap_permutations, summary],
            outputs=[shap_html, shap_table, shap_status],
        )
        def clear_prediction_on_edit():
            return "", render_confidence_html(None), "", "Clinical summary changed; run prediction again."
        summary.input(clear_prediction_on_edit, outputs=[diagnosis, confidence, raw, result_status])
        def clear_shap():
            return render_shap_html(None), [], "Run prediction, then Explain with SHAP."
        view.click(clear_shap, outputs=[shap_html, shap_table, shap_status])
        patient_id.submit(clear_shap, outputs=[shap_html, shap_table, shap_status])
        random_button.click(clear_shap, outputs=[shap_html, shap_table, shap_status])
        choice.change(clear_shap, outputs=[shap_html, shap_table, shap_status])
        predict_button.click(clear_shap, outputs=[shap_html, shap_table, shap_status])
        summary.input(clear_shap, outputs=[shap_html, shap_table, shap_status])
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/generated_clinical_v2/test.jsonl"))
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--host", default="127.0.0.1", help="Default is private loopback only")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--deferral-policy", type=Path, default=None,
                        help="deferral_policy.json from evaluate_trust.py (token score) to show the deferral zone")
    args = parser.parse_args()
    policy = None
    if args.deferral_policy is not None:
        policy = load_policy(args.deferral_policy)
        if policy["score"] != "token":
            parser.error("The app computes the token score live; fit the policy on evaluate_feddx_confidence output")
        print(f"Deferral policy: accept >= {policy['accept_threshold']}, stop < {policy['stop_threshold']}", flush=True)
    controller = FedDxApp(read_test_cases(args.data), args.model, CACHE_CHOICES, policy)
    print(f"Loaded {len(controller.ids)} test patients from {args.data}", flush=True)
    build_ui(controller).queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=False, show_error=True
    )


if __name__ == "__main__":
    main()
