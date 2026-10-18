
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-12B-it"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "Rewrite clinical evidence as a first-person patient "
                    "description. Do not invent symptoms, medical history, "
                    "or diagnoses."
                ),
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Age: 35\n"
                    "Sex: female\n"
                    "Symptoms: fever, dry cough, fatigue\n"
                    "Negative findings: no shortness of breath\n\n"
                    "Generate a patient description."
                ),
            }
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

with torch.inference_mode():
    generated = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

new_tokens = generated[0, inputs["input_ids"].shape[-1]:]

print(processor.decode(new_tokens, skip_special_tokens=True))
