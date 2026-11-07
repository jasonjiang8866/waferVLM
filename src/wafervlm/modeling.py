"""Qwen3.5 multimodal input and language-only LoRA helpers."""

import re

from PIL import Image

from .core import PROMPT, checked_image

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def messages(image, answer=None):
    result = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": PROMPT},
    ]}]
    if answer is not None:
        result.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return result


def encode(processor, image, answer=None):
    # Render template flags separately: Transformers 5.3 otherwise forwards custom
    # flags into image processing and emits misleading "ignored" warnings.
    text = processor.apply_chat_template(
        messages(image, answer), tokenize=False,
        add_generation_prompt=answer is None, enable_thinking=False,
    )
    return processor(text=[text], images=[image], return_tensors="pt", add_special_tokens=False)


def mask_answer(full_ids, prompt_ids):
    import torch

    n = prompt_ids.shape[-1]
    if n >= full_ids.shape[-1] or not torch.equal(full_ids[:, :n], prompt_ids):
        raise ValueError("Chat-template prefix mismatch; refusing to train with incorrect loss mask")
    labels = full_ids.clone()
    labels[:, :n] = -100
    return labels


class AnswerCollator:
    """One image per microbatch; reject truncation instead of damaging vision tokens."""

    def __init__(self, processor, root, max_length=1024):
        self.processor, self.root, self.max_length = processor, root, max_length

    def __call__(self, rows):
        if len(rows) != 1:
            raise ValueError("This baseline requires microbatch size 1; use gradient accumulation")
        row = rows[0]
        with Image.open(checked_image(self.root, row)) as source:
            image = source.convert("RGB")
        prompt = encode(self.processor, image)
        full = encode(self.processor, image, row["label"])
        if full["input_ids"].shape[-1] > self.max_length:
            raise ValueError("Example exceeds max_length; lower image resolution or raise the explicit limit")
        full["labels"] = mask_answer(full["input_ids"], prompt["input_ids"])
        return full


def add_lora(model, rank=8):
    import torch
    from peft import LoraConfig, get_peft_model

    # Exact fully qualified names keep identically named vision projections frozen.
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and "language_model.layers." in name]
    if not targets or not any("linear_attn" in name for name in targets):
        raise ValueError("Expected Qwen3.5 hybrid language layers were not found")
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=rank, lora_alpha=rank * 2,
        lora_dropout=0.0, bias="none", target_modules=targets,
    ))
    assert_frozen_vision(model)
    return model, targets


def assert_frozen_vision(model):
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n or "language_model" not in n for n in trainable):
        raise ValueError("Unexpected trainable parameters outside language LoRA")
    vision = [p for name, p in model.named_parameters() if ".visual." in name]
    if not vision or any(p.requires_grad for p in vision):
        raise ValueError("Vision tower missing or not frozen")


def load_model(revision=MODEL_REVISION, device="cuda"):
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Model revision must be an immutable 40-character commit SHA")
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError("CUDA with BF16 support is required for this configuration")
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision, trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=revision, trust_remote_code=False,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    return model, processor
