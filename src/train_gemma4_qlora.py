#!/usr/bin/env python3
"""FedDx staged QLoRA: 1k -> 3k -> 5k with strict per-stage resume.

Each stage is a separate training run. --resume resumes *that stage's*
optimizer/scheduler/RNG checkpoint. --init-adapter starts a *new stage*
from a previous stage's adapter (optimizer is intentionally reset).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

PROMPT = (
    "You are a clinical diagnosis classifier. Given the clinical summary, "
    "output only the single most likely diagnosis name, without explanation.\n\n"
    "Clinical summary:\n{summary}"
)


def load_records(path):
    rows = []
    ids = set()
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row["patient_id"]
            if pid in ids:
                raise ValueError(f"Duplicate patient_id {pid} at {path}:{n}")
            ids.add(pid)
            if not isinstance(row.get("clinical_summary"), str) or not row["clinical_summary"].strip():
                raise ValueError(f"Missing clinical_summary for {pid}")
            if not isinstance(row.get("target", {}).get("diagnosis"), str):
                raise ValueError(f"Missing diagnosis for {pid}")
            rows.append(row)
    return rows


def fingerprint(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(
            [row["patient_id"], row["clinical_summary"], row["target"]["diagnosis"]],
            ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def make_manifest(args, ordered_train, val):
    return {
        "version": 1,
        "model": args.model,
        "train_path": str(args.train.resolve()),
        "independent": args.independent,
        "train_fingerprint": fingerprint(ordered_train),
        "val_path": str(args.validation.resolve()),
        "val_fingerprint": fingerprint(val),
        "selection_seed": args.seed,
        "stage_size": args.stage_size,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "init_adapter": str(args.init_adapter.resolve()) if args.init_adapter else None,
    }


def assert_or_create_manifest(path, manifest, resume):
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != manifest:
            differences = [k for k in manifest if old.get(k) != manifest[k]]
            raise RuntimeError(f"Stage settings or dataset changed ({differences}); do not reuse {path.parent}")
    else:
        if resume:
            raise RuntimeError(f"Cannot resume: missing {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def latest_checkpoint(output_dir):
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        if (path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
                and all((path / fname).exists() for fname in ("trainer_state.json", "optimizer.pt", "scheduler.pt"))
                and (path / "adapter_config.json").exists()):
            candidates.append((int(path.name.removeprefix("checkpoint-")), path))
    if not candidates:
        raise RuntimeError(f"No complete Trainer checkpoint in {output_dir}")
    return max(candidates)[1]



def find_text_lora_targets(model):
    """Resolve real Gemma 4 text projection modules, excluding vision/audio.

    Gemma4ClippableLinear wraps an nn.Linear in `.linear`, which PEFT can adapt.
    Quantized text projections may instead be bitsandbytes Linear4bit modules.
    """
    import torch

    projections = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = []
    for name, module in model.named_modules():
        if not (name.startswith("language_model.") or name.startswith("text_model.") or ".language_model." in name or ".text_model." in name):
            continue
        if any(x in name for x in ("vision_tower", "audio_tower", "vision_model", "audio_model")):
            continue
        if ".layers." not in name:
            continue
        parts = name.split(".")
        if parts[-1] in projections or (parts[-1] == "linear" and len(parts) > 1 and parts[-2] in projections):
            # Reject wrappers, but allow torch Linear and bitsandbytes Linear4bit/8bit.
            is_linear = isinstance(module, torch.nn.Linear) or module.__class__.__name__ in {"Linear4bit", "Linear8bitLt"}
            if is_linear:
                found.append(name)
    if not found:
        examples = [n for n, _ in model.named_modules() if n.endswith("q_proj") or n.endswith("q_proj.linear")][:8]
        raise RuntimeError(f"No supported Gemma 4 text LoRA target layers found. q_proj examples: {examples}")
    return sorted(found)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=Path("data/generated_clinical_v2/train.jsonl"))
    p.add_argument("--validation", type=Path, default=Path("data/generated_clinical_v2/validate.jsonl"))
    p.add_argument("--model", default="google/gemma-4-12B-it")
    p.add_argument("--stage-size", type=int, choices=(1000, 3000, 5000), required=True)
    p.add_argument("--output-root", type=Path, default=Path("runs/gemma4_qlora"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--init-adapter", type=Path, help="Previous stage's adapter directory; starts new optimizer")
    p.add_argument("--resume", action="store_true", help="Resume a checkpoint inside THIS stage, including optimizer state")
    p.add_argument("--independent", action="store_true", help="Train a new LoRA adapter from the original Gemma 4 model")
    p.add_argument("--dry-run", action="store_true", help="Validate files and show stage composition without GPU")
    args = p.parse_args()
    if args.resume and args.init_adapter:
        p.error("--resume and --init-adapter have different meanings and cannot be combined")
    if args.batch_size < 1 or args.grad_accum < 1 or args.epochs <= 0 or args.max_length < 128:
        p.error("Invalid training hyperparameters")
    if args.stage_size == 1000 and args.init_adapter:
        p.error("The first 1000-case stage must start from the base model")

    if args.independent and args.init_adapter:
        p.error("--independent cannot be combined with --init-adapter")

    if (
        args.stage_size > 1000
        and not args.resume
        and not args.init_adapter
        and not args.independent
    ):
        p.error(
            "For independent training, specify --independent. "
            "For progressive training, specify --init-adapter."
        )

    if args.init_adapter and not (args.init_adapter / "adapter_config.json").is_file():
        p.error(f"Not an adapter directory: {args.init_adapter}")

    train = load_records(args.train)
    val = load_records(args.validation)
    if len(train) < args.stage_size:
        p.error(f"Only {len(train)} training examples available, need {args.stage_size}")
    overlap = {r["patient_id"] for r in train} & {r["patient_id"] for r in val}
    if overlap:
        p.error(f"Train/validation overlapping IDs: {list(overlap)[:5]}")
    # Stable shuffled order: 1k is a strict subset of 3k, which is a subset of 5k.
    train.sort(key=lambda r: hashlib.sha256(f'{args.seed}:{r["patient_id"]}'.encode()).hexdigest())
    chosen = train[:args.stage_size]
    stage_dir = args.output_root / f"stage-{args.stage_size}"
    manifest = make_manifest(args, train, val)
    if args.resume and (stage_dir / "run_manifest.json").is_file():
        # The initialization adapter is only needed when creating a new stage.
        # A same-stage resume instead loads adapter weights from the Trainer checkpoint.
        previous = json.loads((stage_dir / "run_manifest.json").read_text(encoding="utf-8"))
        manifest["init_adapter"] = previous.get("init_adapter")
    print(f"Stage {args.stage_size}: train={len(chosen)}, validation={len(val)}, diagnoses={len(set(r['target']['diagnosis'] for r in chosen))}")
    print("Stage class counts:", dict(Counter(r["target"]["diagnosis"] for r in chosen).most_common()))
    if args.dry_run:
        return
    assert_or_create_manifest(stage_dir / "run_manifest.json", manifest, args.resume)
    if (stage_dir / "complete.json").exists():
        raise RuntimeError(f"Stage already completed: {stage_dir}. Keep its adapter and do not retrain this stage.")
    if not args.resume and list(stage_dir.glob("checkpoint-*")):
        raise RuntimeError("Stage already has checkpoints. Use --resume, or a different output root.")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Gemma tokenizer lacks a pad token")
    tokenizer.padding_side = "right"

    def tokenize_record(record):
        question = PROMPT.format(summary=record["clinical_summary"].strip())
        answer = record["target"]["diagnosis"].strip()
        prompt_messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        full_messages = prompt_messages + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        # Gemma 4 may render add_generation_prompt=True differently from the
        # beginning of a completed assistant turn (e.g. a final-turn marker).
        # Do not assume prompt rendering is a string prefix of full rendering.
        full_text = processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        # Locate the actual assistant-content slot by rendering the same
        # conversation with a unique marker. This is safer than rfind(answer),
        # because a diagnosis could also appear in the clinical history.
        marker = "FEDDX_ASSISTANT_CONTENT_SLOT_7BC91"
        marker_messages = prompt_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": marker}]}
        ]
        marked_text = processor.apply_chat_template(
            marker_messages, tokenize=False, add_generation_prompt=False
        )
        if marked_text.count(marker) != 1 or marked_text.replace(marker, answer) != full_text:
            raise RuntimeError(
                f"Cannot isolate assistant output with Gemma template for {record['patient_id']}"
            )
        answer_start = marked_text.index(marker)
        answer_end = answer_start + len(answer)
        encoded = tokenizer(
            full_text, add_special_tokens=False, return_offsets_mapping=True
        )
        full_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        # Mask everything before the assistant diagnosis. Offset-based masking
        # handles template variants where prompt/full tokenizations differ.
        labels = [-100] * len(full_ids)
        answer_token_start = None
        for i, (start, end) in enumerate(offsets):
            if end <= start:  # special token with no text span
                continue
            if start < answer_start < end:
                raise RuntimeError(
                    f"Diagnosis begins inside a token for {record['patient_id']}; "
                    "refusing to train on part of the prompt"
                )
            if start >= answer_start and start < answer_end:
                labels[i] = full_ids[i]
                if answer_token_start is None:
                    answer_token_start = i
        if answer_token_start is None:
            raise RuntimeError(f"No supervised diagnosis tokens for {record['patient_id']}")
        # Include any assistant end-of-turn/EOS tokens after the answer, which
        # Gemma needs to learn to stop generating after the diagnosis.
        for i in range(answer_token_start + 1, len(full_ids)):
            start, end = offsets[i]
            if (end <= start or start >= answer_end):
                labels[i] = full_ids[i]
        if len(full_ids) > args.max_length:
            raise RuntimeError(f"Example {record['patient_id']} has {len(full_ids)} tokens > {args.max_length}; increase --max-length (no silent truncation)")
        active = [t for t in labels if t != -100]
        if not active or answer.lower() not in tokenizer.decode(active, skip_special_tokens=True).lower():
            raise RuntimeError(f"Completion missing from supervised region for {record['patient_id']}")
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    def collate(samples):
        max_len = max(len(s["input_ids"]) for s in samples)
        ids, masks, labels = [], [], []
        for s in samples:
            k = max_len - len(s["input_ids"])
            ids.append(s["input_ids"] + [tokenizer.pad_token_id] * k)
            masks.append(s["attention_mask"] + [0] * k)
            labels.append(s["labels"] + [-100] * k)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks), "labels": torch.tensor(labels)}

    print("Preparing tokenized datasets...", flush=True)
    training_data = Dataset.from_list([tokenize_record(r) for r in chosen])
    validation_data = Dataset.from_list([tokenize_record(r) for r in val])

    print("Loading quantized Gemma 4...", flush=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
        ),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    if args.init_adapter:
        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
    else:
        targets = find_text_lora_targets(model)
        print(f"Found {len(targets)} LoRA text projections (first: {targets[:5]})", flush=True)
        config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
            target_modules=targets,
        )
        from peft import get_peft_model
        model = get_peft_model(model, config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(stage_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=10,
        logging_steps=10,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=3,
        eval_strategy="steps", eval_steps=args.save_steps,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", report_to="none",
        remove_unused_columns=False, save_only_model=False,
        load_best_model_at_end=False, seed=args.seed, data_seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=training_data,
        eval_dataset=validation_data, data_collator=collate,
        processing_class=tokenizer,
    )
    checkpoint = str(latest_checkpoint(stage_dir)) if args.resume else None
    print(f"Training stage={args.stage_size} checkpoint={checkpoint or 'new'}", flush=True)
    trainer.train(resume_from_checkpoint=checkpoint)
    adapter_dir = stage_dir / "adapter_final"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    (stage_dir / "complete.json").write_text(json.dumps({"stage": args.stage_size, "adapter": str(adapter_dir), "global_step": trainer.state.global_step}, indent=2) + "\n", encoding="utf-8")
    print(f"Stage finished; adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
