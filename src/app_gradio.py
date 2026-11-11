
#!/usr/bin/env python3
"""FedDx case explorer: original evidence, generated summary, and Gemma predictions.

Run from the FedDx project root:
    uv add gradio
    uv run python src/app_gradio.py

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

PROMPT = (
    "You are a clinical diagnosis classifier. Given the clinical summary, "
    "output only the single most likely diagnosis name, without explanation.\n\n"
    "Clinical summary:\n{summary}"
)

MODEL_CHOICES = {
    "Vanilla Gemma 4 12B": None,
    "QLoRA 1k": Path("runs/gemma4_qlora/stage-1000/adapter_final"),
    "QLoRA 5k (independent)": Path(
        "runs/gemma4_qlora_independent/stage-5000/adapter_final"
    ),
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
        Path(
            "runs/gemma4_evaluation_independent/test/stage-5000/predictions.jsonl"
        ),
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
                raise ValueError(
                    f"Duplicate patient ID at {path}:{line_number}: {pid}"
                )

            if not record.get("clinical_summary", "").strip():
                raise ValueError(
                    f"Missing clinical summary at {path}:{line_number}"
                )

            records[pid] = record

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def readable_answer(value: Any) -> str:
    """Preserve meaning without interpreting arbitrary categorical scales."""
    if value is None or value == "":
        return "Not recorded"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, list):
        return (
            "; ".join(readable_answer(v) for v in value)
            or "Not recorded"
        )

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    s = str(value).strip()

    return {
        "y": "Yes",
        "n": "No",
        "yes": "Yes",
        "no": "No",
        "true": "Yes",
        "false": "No",
    }.get(s.casefold(), s)


def evidence_rows(record: dict[str, Any]) -> list[list[str]]:
    rows = []

    for evidence in record.get("clinical_evidence", []):
        kind = str(
            evidence.get("type", "unspecified")
        ).replace("_", " ").title()

        question = str(
            evidence.get("question") or "Question unavailable"
        )

        answer = readable_answer(evidence.get("answer"))
        code = str(evidence.get("evidence_code") or "")

        rows.append([kind, question, answer, code])

    return rows


def clean_response(raw: str) -> str:
    """Remove only a known leading decoded channel name."""
    text = raw.strip()

    text = re.sub(
        r"^thought\s*\n",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )

    return text.strip()


def parse_diagnosis(
    raw: str,
    vocabulary: set[str],
) -> tuple[str, bool]:
    text = clean_response(raw)

    first = (
        text.splitlines()[0].strip()
        if text else ""
    )

    first = re.sub(
        r"^(?:diagnosis|answer)\s*:\s*",
        "",
        first,
        flags=re.IGNORECASE,
    )

    first = re.sub(
        r"^(?:the most likely diagnosis is|most likely diagnosis is)\s+",
        "",
        first,
        flags=re.IGNORECASE,
    )

    lookup = {
        label.casefold().strip(): label
        for label in vocabulary
    }

    label = lookup.get(
        first.casefold().strip(" .\"'`")
    )

    return (
        label if label is not None else (first or "(empty response)"),
        label is not None,
    )


class FedDxApp:
    def __init__(
        self,
        cases: dict[str, dict[str, Any]],
        model_name: str,
        cache_paths: dict[str, list[Path]],
    ):
        self.cases = cases
        self.ids = sorted(cases)

        self.vocabulary = {
            r["target"]["diagnosis"]
            for r in cases.values()
        }

        self.model_name = model_name
        self.cache_paths = cache_paths
        self.cache: dict[str, dict[str, dict[str, Any]]] = {}

        self._model = None
        self._processor = None
        self._loaded_choice = None

        self._model_lock = threading.RLock()
        self._rng = random.SystemRandom()

    def load_patient(self, pid: str):
        pid = (pid or "").strip()
        record = self.cases.get(pid)

        if record is None:
            return (
                f"**Patient ID not found:** `{pid}`. "
                "Select Random patient or enter an ID from the test split.",
                [],
                "",
                "",
                "",
                render_confidence_html(None),
                "",
                "",
            )

        feats = record.get("patient_features", {})

        age = feats.get("age", "unknown")
        sex = feats.get("sex", "unknown")

        evidence = evidence_rows(record)

        metadata = (
            f"**Patient:** `{pid}` "
            f"· **Age:** {age} "
            f"· **Sex (source code):** {sex} "
            f"· **Evidence entries:** {len(evidence)}"
        )

        return (
            metadata,
            evidence,
            record["clinical_summary"],
            record["target"]["diagnosis"],
            "",
            render_confidence_html(None),
            "",
            "",
        )

    def random_patient(self):
        pid = self._rng.choice(self.ids)
        return (pid, *self.load_patient(pid))

    def _cached_predictions(
        self,
        choice: str,
    ) -> dict[str, dict[str, Any]]:

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

                    if (
                        pid in self.cases
                        and row.get("truth")
                        == self.cases[pid]["target"]["diagnosis"]
                    ):
                        # Prefer earlier cache paths for the same patient.
                        predictions.setdefault(pid, row)

        self.cache[choice] = predictions
        return predictions

    def _ensure_model(self, choice: str):
        if (
            self._loaded_choice == choice
            and self._model is not None
        ):
            return

        import torch

        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Live model inference requires an NVIDIA CUDA GPU. "
                "Cached predictions work without one."
            )

        adapter = MODEL_CHOICES[choice]

        if (
            adapter is not None
            and not adapter.joinpath("adapter_config.json").is_file()
        ):
            raise FileNotFoundError(
                f"Adapter missing: {adapter}"
            )

        # Keep only one quantized model in GPU memory at a time.
        self._model = None
        self._processor = None
        self._loaded_choice = None

        gc.collect()
        torch.cuda.empty_cache()

        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        processor = AutoProcessor.from_pretrained(
            self.model_name
        )

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

            model = PeftModel.from_pretrained(
                model,
                str(adapter),
                is_trainable=False,
            )

        model.eval()

        self._model = model
        self._processor = processor
        self._loaded_choice = choice

    def predict(
        self,
        pid: str,
        choice: str,
        use_cache: bool,
    ):
        pid = (pid or "").strip()
        record = self.cases.get(pid)

        if record is None:
            return (
                "",
                render_confidence_html(None),
                "",
                "Choose a valid test patient first.",
            )

        if choice not in MODEL_CHOICES:
            return (
                "",
                render_confidence_html(None),
                "",
                "Unknown model choice.",
            )

        if use_cache:
            cached = self._cached_predictions(choice).get(pid)

            if cached is not None:
                raw = cached.get("raw_response", "")
                prediction, valid = parse_diagnosis(
                    raw,
                    self.vocabulary,
                )

                return (
                    prediction,
                    render_confidence_html(
                        None,
                        cached=True,
                    ),
                    raw,
                    (
                        "**Source:** saved evaluation prediction "
                        f"· **Allowed diagnosis label:** "
                        f"{'yes' if valid else 'no'}. "
                        "No token probabilities were stored in this "
                        "evaluation cache; disable saved predictions "
                        "for confidence scoring."
                    ),
                )

        with self._model_lock:
            try:
                self._ensure_model(choice)

                import torch

                processor = self._processor
                model = self._model

                text = PROMPT.format(
                    summary=record["clinical_summary"].strip()
                )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                            }
                        ],
                    }
                ]

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

                completion_ids = generated.sequences[
                    0,
                    inputs["input_ids"].shape[1]:,
                ]

                raw = processor.tokenizer.decode(
                    completion_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                prediction, valid = parse_diagnosis(
                    raw,
                    self.vocabulary,
                )

                try:
                    stats = score_generated_diagnosis(
                        processor.tokenizer,
                        generated.sequences,
                        generated.scores,
                        inputs["input_ids"].shape[1],
                    )
                except Exception as score_error:
                    stats = {
                        "available": False,
                        "reason": (
                            "Token scoring failed: "
                            f"{type(score_error).__name__}"
                        ),
                    }

                return (
                    prediction,
                    render_confidence_html(stats),
                    raw,
                    (
                        "**Source:** live Gemma inference "
                        f"· **Allowed diagnosis label:** "
                        f"{'yes' if valid else 'no'}. "
                        "Confidence panel shows uncalibrated token "
                        "likelihoods (not clinical correctness)."
                    ),
                )

            except Exception as exc:
                return (
                    "",
                    render_confidence_html(None),
                    "",
                    (
                        "**Inference error:** "
                        f"`{type(exc).__name__}: {exc}`"
                    ),
                )


def build_ui(controller: FedDxApp):
    import gradio as gr

    with gr.Blocks(
        title="FedDx · Clinical Case Explorer"
    ) as app:

        gr.Markdown(
            "# FedDx · Clinical Case Explorer\n"
            "Explore **synthetic DDXPlus patient evidence**, "
            "its **LLM-generated clinician-style summary**, "
            "and **Gemma 4 diagnosis predictions**. "
            "**Research demonstration only; not clinical advice.**"
        )

        # Patient selection
        with gr.Row():
            patient_id = gr.Textbox(
                label="Test patient ID",
                placeholder="e.g. test_0002432",
                scale=3,
            )

            view = gr.Button(
                "Load patient",
                variant="primary",
                scale=1,
            )

            random_button = gr.Button(
                "Random patient",
                scale=1,
            )

        identity = gr.Markdown(
            "Choose a patient to begin."
        )

        # Evidence and clinical summary
        with gr.Row():
            with gr.Column(scale=6):
                gr.Markdown(
                    "### 1 · Original patient evidence"
                )

                evidence = gr.Dataframe(
                    headers=[
                        "Type",
                        "Question",
                        "Recorded answer",
                        "Evidence code",
                    ],
                    datatype=[
                        "str",
                        "str",
                        "str",
                        "str",
                    ],
                    type="array",
                    interactive=False,
                    wrap=True,
                    label="Decoded questionnaire evidence",
                )

                gr.Markdown(
                    "*Answers reflect the synthetic source. "
                    "A positive question with 'A or B' does not "
                    "establish both alternatives. "
                    "Uninterpreted numeric values are shown "
                    "as recorded.*"
                )

            with gr.Column(scale=5):
                gr.Markdown(
                    "### 2 · LLM-generated clinical summary"
                )

                summary = gr.Textbox(
                    label="clinician-v2 summary (read-only)",
                    lines=13,
                    interactive=False,
                )

                truth = gr.Textbox(
                    label="Dataset reference diagnosis (research label)",
                    interactive=False,
                )

        # Model inference controls
        gr.Markdown(
            "### 3 · Model diagnosis"
        )

        with gr.Row():
            choice = gr.Dropdown(
                choices=list(MODEL_CHOICES),
                value="QLoRA 5k (independent)",
                label="Model",
                scale=3,
            )

            cache_toggle = gr.Checkbox(
                value=False,
                label=(
                    "Use saved test predictions "
                    "(no confidence scores)"
                ),
                scale=2,
            )

            predict_button = gr.Button(
                "Predict diagnosis",
                variant="primary",
                scale=2,
            )

        # Prediction results
        with gr.Row():
            diagnosis = gr.Textbox(
                label="Predicted diagnosis",
                interactive=False,
                scale=3,
            )

            confidence = gr.HTML(
                value=render_confidence_html(None),
                label="Model token confidence",
                scale=2,
            )

        gr.Markdown(
            "*Confidence is based on generated "
            "diagnosis-token likelihoods, not calibrated "
            "diagnostic reliability. Green/yellow/red "
            "are heuristic visualization thresholds.*"
        )

        result_status = gr.Markdown(
            "Choose a model and press Predict diagnosis."
        )

        with gr.Accordion(
            "Raw decoded model response",
            open=False,
        ):
            raw = gr.Textbox(
                label="Raw response",
                lines=4,
                interactive=False,
            )

        # Wire up events
        patient_outputs = [
            identity,
            evidence,
            summary,
            truth,
            diagnosis,
            confidence,
            raw,
            result_status,
        ]

        view.click(
            controller.load_patient,
            inputs=patient_id,
            outputs=patient_outputs,
        )

        patient_id.submit(
            controller.load_patient,
            inputs=patient_id,
            outputs=patient_outputs,
        )

        random_button.click(
            controller.random_patient,
            outputs=[
                patient_id,
                *patient_outputs,
            ],
        )

        predict_button.click(
            controller.predict,
            inputs=[
                patient_id,
                choice,
                cache_toggle,
            ],
            outputs=[
                diagnosis,
                confidence,
                raw,
                result_status,
            ],
        )

        choice.change(
            lambda: (
                "",
                render_confidence_html(None),
                "",
                "Model changed; press Predict diagnosis.",
            ),
            outputs=[
                diagnosis,
                confidence,
                raw,
                result_status,
            ],
        )

    return app


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "data/generated_clinical_v2/test.jsonl"
        ),
    )

    parser.add_argument(
        "--model",
        default="google/gemma-4-12B-it",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Default is private loopback only",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
    )

    args = parser.parse_args()

    controller = FedDxApp(
        read_test_cases(args.data),
        args.model,
        CACHE_CHOICES,
    )

    print(
        f"Loaded {len(controller.ids)} test patients "
        f"from {args.data}",
        flush=True,
    )

    build_ui(controller).queue(
        default_concurrency_limit=1
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
