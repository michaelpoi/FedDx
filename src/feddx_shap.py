"""Model-agnostic, phrase-level SHAP for FedDx Gemma diagnosis generation.

Approximates Shapley values of the *fixed predicted diagnosis* teacher-forced
log-likelihood, using a missing-text baseline. It does not attribute clinical
correctness, provide calibrated confidence, or establish causality.
"""
from __future__ import annotations

from html import escape
import math
import re
from typing import Any


def split_clinical_text(text: str, max_features: int = 16) -> list[str]:
    """Contiguous text fragments, reconstructing original text exactly."""
    if not 2 <= max_features <= 64:
        raise ValueError('max_features must be between 2 and 64')
    if not text.strip():
        raise ValueError('Empty clinical summary')
    # Use sentence and clause boundaries, preserving all original whitespace.
    # Each clause keeps its delimiter plus any following whitespace/newlines,
    # so "male.\nPresenting" splits cleanly (the old pattern failed on ".\n",
    # which every FedDx summary contains, and fell back to word groups).
    matches = re.findall(r'.*?(?:[,;.!?]+\s*|\n+|$)', text, flags=re.S)
    chunks = [m for m in matches if m]
    if ''.join(chunks) != text:
        chunks = [text]
    # For single long narrative, fall back to contiguous groups of words.
    if len(chunks) == 1:
        words = re.findall(r'\s*\S+', text)
        if ''.join(words) == text and len(words) > 1:
            stride = max(1, math.ceil(len(words) / min(max_features, 12)))
            chunks = [''.join(words[i:i+stride]) for i in range(0, len(words), stride)]
    if len(chunks) > max_features:
        stride = math.ceil(len(chunks) / max_features)
        chunks = [''.join(chunks[i:i+stride]) for i in range(0, len(chunks), stride)]
    assert ''.join(chunks) == text
    return chunks


def diagnosis_log_likelihood(model: Any, processor: Any, prompt_template: str,
                             summary: str, diagnosis: str) -> float:
    """Score *only* diagnosis tokens in a completed Gemma assistant turn.

    Uses the same full-chat rendering and offset-based target identification
    as the FedDx training script, rather than add_generation_prompt=True
    (which may open a thought channel).
    """
    import torch
    tokenizer = processor.tokenizer
    question = prompt_template.format(summary=summary.strip())
    user = [{'role':'user', 'content':[{'type':'text','text':question}]}]
    assistant = [{'role':'assistant','content':[{'type':'text','text':diagnosis}]}]
    marker = 'FEDDX_SHAP_TARGET_SLOT_93C2B'
    marked = processor.apply_chat_template(
        user + [{'role':'assistant','content':[{'type':'text','text':marker}]}],
        tokenize=False, add_generation_prompt=False)
    full = processor.apply_chat_template(user + assistant, tokenize=False,
                                         add_generation_prompt=False)
    if marked.count(marker) != 1 or marked.replace(marker, diagnosis) != full:
        raise ValueError('Gemma chat template target alignment failed')
    start = marked.index(marker)
    end = start + len(diagnosis)
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    offsets = encoded['offset_mapping']
    target_positions = []
    for i, (a, b) in enumerate(offsets):
        if a < start < b:
            raise ValueError('Target starts inside a token')
        if b > a and start <= a < end:
            target_positions.append(i)
    if not target_positions or min(target_positions) == 0:
        raise ValueError('No target diagnosis tokens')
    if len(ids) > 2048:
        raise ValueError('Prompt exceeds 2048 tokens; shorten clinical summary')
    inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs))
        # In causal LM, logits[t-1] predict token[t].
        logits = outputs.logits[0, [i-1 for i in target_positions], :].float()
        target = inputs[0, target_positions]
        lp = torch.log_softmax(logits, dim=-1).gather(-1, target[:, None]).sum()
    return float(lp.item())


def explain_summary(model: Any, processor: Any, prompt_template: str, summary: str,
                    diagnosis: str, *, max_features: int = 12,
                    permutations: int = 2, seed: int = 42,
                    cache: dict[tuple[int, ...], float] | None = None) -> dict[str, Any]:
    """Compute phrase SHAP on diagnosis log P using masked-out phrases.

    SHAP's model input is a binary mask of present chunks; the baseline is
    the all-absent summary. We cache coalition model evaluations. Pass a
    `cache` dict to reuse those evaluations afterwards (e.g. deletion tests);
    it is only valid for the same summary, diagnosis, model and max_features.
    """
    import numpy as np
    import shap
    if not diagnosis.strip() or diagnosis.startswith('['):
        raise ValueError('A non-empty generated diagnosis is required')
    chunks = split_clinical_text(summary, max_features=max_features)
    n = len(chunks)
    if n < 2:
        raise ValueError('At least two text chunks are needed for SHAP')
    if not 1 <= permutations <= 20:
        raise ValueError('permutations must be between 1 and 20')
    if cache is None:
        cache = {}

    def evaluate(mask_matrix):
        result = []
        for row in np.atleast_2d(mask_matrix):
            mask = tuple(int(v > .5) for v in row)
            if mask not in cache:
                masked_summary = ''.join(chunk for chunk, keep in zip(chunks,mask) if keep)
                cache[mask] = diagnosis_log_likelihood(
                    model, processor, prompt_template, masked_summary, diagnosis)
            result.append(cache[mask])
        return np.asarray(result, dtype=float)

    background = np.zeros((1, n), dtype=float)
    explainer = shap.Explainer(evaluate, background, algorithm='permutation', seed=seed,
                              feature_names=[c.strip()[:60] or '(space)' for c in chunks])
    values = explainer(np.ones((1,n),dtype=float), max_evals=(2*n+1)*permutations,
                       batch_size=1, silent=True)
    attributions = np.asarray(values.values[0],dtype=float).reshape(-1)
    base = float(np.asarray(values.base_values).reshape(-1)[0])
    full = float(evaluate(np.ones((1,n),dtype=float))[0])
    return {'chunks': chunks, 'shap_values': attributions.tolist(),
            'baseline_log_likelihood':base,'full_log_likelihood':full,
            'additivity_error':float(abs((base+attributions.sum())-full)),
            'model_evaluations':len(cache), 'diagnosis':diagnosis,
            'max_features':max_features,'permutations':permutations,
            'method':'shap_permutation_masked_clauses_fixed_diagnosis_log_likelihood'}


def render_shap_html(result: dict[str, Any] | None) -> str:
    if result is None:
        return ('<div style="padding:14px;border-radius:10px;background:#e5e7eb;'
                'color:#111827!important">Run a prediction, then select Explain with SHAP.</div>')
    chunks = result['chunks']; values = result['shap_values']
    scale = max((abs(float(x)) for x in values), default=0.0) or 1.0
    spans = []
    for chunk, value in zip(chunks, values):
        normalized = abs(float(value))/scale
        alpha = .12 + .50*normalized
        bg = f'rgba(239,68,68,{alpha:.3f})' if value >= 0 else f'rgba(59,130,246,{alpha:.3f})'
        tooltip = escape(f'SHAP {float(value):+.4f} log-likelihood units', quote=True)
        spans.append(f'<span title="{tooltip}" style="background:{bg};color:#111827!important;'
                     f'border-radius:3px;padding:1px 0">{escape(chunk)}</span>')
    return (f'<div style="background:#f8fafc;color:#111827!important;padding:16px;'
            f'border-radius:10px;line-height:1.9;white-space:pre-wrap">'
            f'{"".join(spans)}</div>'
            f'<div style="color:#111827!important;margin-top:9px;font-size:0.9em">'
            f'<b style="color:#111827!important">Red</b>: supports the predicted label &nbsp; '
            f'<b style="color:#111827!important">Blue</b>: opposes it &nbsp; '
            f'Intensity = relative SHAP magnitude. These are not clinical importance ratings.</div>')

def attribution_rows(result: dict[str, Any]) -> list[list[str]]:
    indexed = sorted(zip(result['chunks'], result['shap_values']),
                     key=lambda cv:abs(cv[1]), reverse=True)
    return [[chunk.strip() or '(whitespace)', f'{float(value):+.4f}',
             'Supports' if value>0 else 'Opposes' if value<0 else 'Neutral']
            for chunk,value in indexed]
