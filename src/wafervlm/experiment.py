"""Training, evaluation and provenance. No RunPod account access in the worker."""

import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

from .core import LABELS, PROMPT, audit_dataset, metrics, parse_label, sha256_file, write_json, write_jsonl
from .modeling import MODEL_ID, MODEL_REVISION, AnswerCollator, add_lora, encode, load_model


def provenance(dataset, revision, config):
    import torch

    def git(*args):
        try:
            p = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        except FileNotFoundError:
            # The slim worker image intentionally has no Git binary/checkout.
            return None
        return p.stdout.strip() if p.returncode == 0 else None

    return {
        "model_id": MODEL_ID, "model_revision": revision, "config": config,
        "dataset_sha256": sha256_file(Path(dataset) / "dataset.json"),
        "source_sha": git("rev-parse", "HEAD") or os.environ.get("WAFERVLM_SOURCE_SHA"),
        "git_status": git("status", "--porcelain"),
        "image_digest": os.environ.get("WAFERVLM_IMAGE_DIGEST"),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "peft", "accelerate", "Pillow", "numpy", "pandas")},
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda, "prompt": PROMPT, "labels": list(LABELS),
    }


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def seal_output(output):
    output = Path(output)
    hashes = {str(p.relative_to(output)): sha256_file(p)
              for p in sorted(output.rglob("*")) if p.is_file() and p.name != "checksums.json"}
    write_json(output / "checksums.json", hashes)


def train(dataset, output, steps=100, seed=17, accumulation=8, rank=8, learning_rate=0.0002,
          revision=MODEL_REVISION, device="cuda", max_length=1024, resume_from=None,
          lower_learning_rate=False):
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    if min(steps, accumulation, rank, max_length) < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Training parameters must be positive")
    if lower_learning_rate and not resume_from:
        raise ValueError("LR intervention requires a complete resume checkpoint")
    rows = audit_dataset(dataset)
    config = dict(steps=steps, seed=seed, accumulation=accumulation, rank=rank,
                  learning_rate=learning_rate, max_length=max_length, device=device)
    from .resume import check_resume, lower_constant_learning_rate
    resumed = check_resume(resume_from, dataset, revision, config,
                           lower_learning_rate=lower_learning_rate) if resume_from else None
    start_step = resumed["global_step"] if resumed else 0
    output = new_output(output)
    set_seed(seed)
    model, processor = load_model(revision, device)
    model, targets = add_lora(model, rank)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    diagnostics = {"lora_targets": targets, "gradient_checks": 0, "nonzero_gradient_checks": 0,
                   "learning_rate_trace": [], "learning_rate_intervention": None}

    logged_losses = []

    class GradientCheck(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.global_step != start_step:
                raise ValueError("Trainer failed to restore global step")
            if resumed:
                optimizer = kwargs['optimizer']
                state_steps = sorted({float(v['step']) for v in optimizer.state.values() if 'step' in v})
                if state_steps != [float(start_step)]:
                    raise ValueError("Adam optimizer state was not restored at the expected step")
                diagnostics['optimizer_steps_at_resume'] = state_steps
                if lower_learning_rate:
                    diagnostics['learning_rate_intervention'] = lower_constant_learning_rate(
                        optimizer, kwargs['lr_scheduler'], resumed['parent_learning_rate'], learning_rate)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                logged_losses.append(float(logs["loss"]))

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            rates = [float(g['lr']) for g in kwargs['optimizer'].param_groups]
            if not rates or any(not math.isclose(lr, learning_rate, rel_tol=1e-12) for lr in rates):
                raise ValueError("Effective optimizer LR differs from requested constant LR")
            diagnostics['learning_rate_trace'].append({'step': state.global_step + 1, 'rates': rates})
            grads = [p.grad for p in kwargs["model"].parameters() if p.requires_grad and p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise ValueError("Missing or non-finite adapter gradients")
            diagnostics["gradient_checks"] += 1
            if any(torch.count_nonzero(g).item() for g in grads):
                diagnostics["nonzero_gradient_checks"] += 1

    run_info = provenance(dataset, revision, config)
    run_info["resume"] = resumed
    write_json(output / "run.json", run_info)
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), max_steps=steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=accumulation,
        learning_rate=learning_rate, lr_scheduler_type="constant", warmup_steps=0,
        weight_decay=0.0, max_grad_norm=1.0, seed=seed, data_seed=seed,
        bf16=device == "cuda", use_cpu=device == "cpu", report_to="none",
        remove_unused_columns=False, dataloader_num_workers=0,
        save_strategy="steps", save_steps=max(1, steps - start_step) if resumed else max(1, steps // 2), save_total_limit=2,
        logging_steps=1, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_torch",
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer = Trainer(model=model, args=args, train_dataset=rows["train"],
                      data_collator=AnswerCollator(processor, dataset, max_length),
                      callbacks=[GradientCheck()])
    result = trainer.train(resume_from_checkpoint=str(resume_from) if resumed else None)
    if trainer.state.global_step != steps or diagnostics["gradient_checks"] != steps - start_step:
        raise ValueError("Unexpected optimizer-step count after resume")
    if not math.isfinite(result.training_loss):
        raise ValueError("Non-finite training loss")
    if not diagnostics["nonzero_gradient_checks"]:
        raise ValueError("No nonzero adapter gradient observed")
    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    processor.save_pretrained(adapter)
    write_json(output / "training.json", {
        **result.metrics, **diagnostics,
        "trainer_reported_train_loss": result.training_loss,
        "train_loss": sum(logged_losses) / len(logged_losses),
        "start_step": start_step, "end_step": trainer.state.global_step,
        "executed_steps": diagnostics["gradient_checks"],
        "elapsed_seconds": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "status": "training-completed-quality-not-yet-evaluated",
    })
    seal_output(output)
    return output


def evaluate(dataset, output, split="test", adapter=None, revision=MODEL_REVISION,
             device="cuda", max_new_tokens=16, include_loss=False):
    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import GenerationConfig, set_seed
    from .core import checked_image

    if split not in ("validation", "test") or max_new_tokens < 1:
        raise ValueError("Invalid evaluation split or token cap")
    rows = audit_dataset(dataset)[split]
    if adapter:
        adapter = Path(adapter)
        run = json.loads((adapter.parent / "run.json").read_text())
        if run["model_revision"] != revision or run["model_id"] != MODEL_ID:
            raise ValueError("Adapter/base model mismatch")
        if run["dataset_sha256"] != sha256_file(Path(dataset) / "dataset.json"):
            raise ValueError("Adapter trained on a different dataset protocol")
        checksums = json.loads((adapter.parent / "checksums.json").read_text())
        if not checksums:
            raise ValueError("Empty adapter integrity manifest")
        for name, digest in checksums.items():
            p = (adapter.parent / name).resolve()
            if not p.is_relative_to(adapter.parent.resolve()) or sha256_file(p) != digest:
                raise ValueError("Adapter artifact checksum mismatch")
    output = new_output(output)
    set_seed(17)
    model, processor = load_model(revision, device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    eos = processor.tokenizer.eos_token_id
    # A fresh config prevents checkpoint-level sampling overrides.
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                  eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id or eos,
                                  use_cache=True)
    config = dict(split=split, adapter=bool(adapter), device=device, max_new_tokens=max_new_tokens,
                  include_loss=include_loss,
                  do_sample=False, enable_thinking=False)
    info = provenance(dataset, revision, config)
    if adapter:
        info["adapter_checksums_sha256"] = sha256_file(adapter.parent / "checksums.json")
    write_json(output / "run.json", info)
    predictions = []
    loss_sum, answer_tokens = 0.0, 0
    loss_collator = AnswerCollator(processor, dataset, 1024) if include_loss else None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for row in rows:
        if loss_collator is not None:
            batch = loss_collator([row]).to(device)
            tokens = int((batch["labels"][:, 1:] != -100).sum().item())
            with torch.inference_mode():
                answer_loss = float(model(**batch, use_cache=False).loss.item())
            if tokens < 1 or not math.isfinite(answer_loss):
                raise ValueError("Invalid validation answer loss")
            loss_sum += answer_loss * tokens
            answer_tokens += tokens
            del batch
        with Image.open(checked_image(dataset, row)) as source:
            inputs = encode(processor, source.convert("RGB")).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, generation_config=generation)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        tokens = generated[0, inputs["input_ids"].shape[-1]:].tolist()
        truncated = len(tokens) >= max_new_tokens and eos not in tokens
        text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
        predictions.append(dict(id=row["id"], label=row["label"], image_sha256=row["image_sha256"],
                                prediction=parse_label(text, truncated), response=text,
                                truncated=truncated, generated_tokens=len(tokens), seconds=elapsed))
        # Persist incremental results if a worker is interrupted.
        write_jsonl(output / "predictions.jsonl", predictions)
    report = metrics(predictions)
    report.update(mean_generation_seconds=sum(r["seconds"] for r in predictions) / len(predictions),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None)
    if include_loss:
        report.update(answer_token_loss=loss_sum / answer_tokens, answer_tokens=answer_tokens)
    write_json(output / "metrics.json", report)
    seal_output(output)
    return report


def compare(baseline, tuned, output):
    from .core import read_jsonl

    baseline, tuned = Path(baseline), Path(tuned)
    runs = [json.loads((p / "run.json").read_text()) for p in (baseline, tuned)]
    for key in ("dataset_sha256", "model_id", "model_revision", "prompt", "labels"):
        if runs[0][key] != runs[1][key]:
            raise ValueError(f"Incomparable runs: {key}")
    for key in ("split", "max_new_tokens", "do_sample", "enable_thinking", "device"):
        if runs[0]["config"][key] != runs[1]["config"][key]:
            raise ValueError(f"Incomparable inference settings: {key}")
    if runs[0]["config"]["adapter"] or not runs[1]["config"]["adapter"]:
        raise ValueError("Expected base model followed by adapted model")
    preds = [read_jsonl(p / "predictions.jsonl") for p in (baseline, tuned)]
    keys = lambda rows: [(r["id"], r["label"], r["image_sha256"]) for r in rows]
    if keys(preds[0]) != keys(preds[1]):
        raise ValueError("Evaluation examples differ")
    scores = [metrics(rows) for rows in preds]
    report = {"baseline": scores[0], "tuned": scores[1],
              "macro_f1_delta": scores[1]["macro_f1"] - scores[0]["macro_f1"],
              "accuracy_delta": scores[1]["accuracy"] - scores[0]["accuracy"],
              "interpretation": "Single-seed comparison; not production qualification or NVIDIA-score reproduction"}
    output = new_output(output)
    write_json(output / "comparison.json", report)
    seal_output(output)
    return report
