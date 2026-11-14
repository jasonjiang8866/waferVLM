"""DINOv3 visual baselines. No provider access; validation-only model selection."""

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from .core import LABELS, audit_dataset, checked_image, metrics, sha256_file, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]


def recipe():
    return json.loads((ROOT / "configs/dinov3.json").read_text())


def check_data(dataset):
    cfg = recipe()
    root = Path(dataset)
    if (sha256_file(root / "dataset.json") != cfg["dataset_sha256"]
            or sha256_file(root / "validation.jsonl") != cfg["validation_sha256"]):
        raise ValueError("DINOv3 requires the frozen Qwen dataset and validation manifest")
    rows = audit_dataset(root)
    if {k: len(v) for k, v in rows.items()} != dict(train=790, validation=160, test=160):
        raise ValueError("Frozen split counts mismatch")
    # Historical test files are integrity-audited, never used by training or inference.
    return {k: rows[k] for k in ("train", "validation")}


def preprocess(image):
    """Preserve the existing entire 256px wafer: no resize, crop or augmentation."""
    import torch

    cfg = recipe()
    if image.size != (cfg["image_size"], cfg["image_size"]):
        raise ValueError("Expected the original 256x256 wafer rendering")
    pixels = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32)).permute(2, 0, 1) / 255
    mean = torch.tensor(cfg["image_mean"])[:, None, None]
    std = torch.tensor(cfg["image_std"])[:, None, None]
    return (pixels - mean) / std


def load_backbone(device="cuda", local_files_only=False):
    import torch
    from transformers import AutoModel

    cfg = recipe()
    if device not in ("cuda", "cpu"):
        raise ValueError("Expected cuda or cpu")
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError("CUDA with BF16 support is required")
    model, loading = AutoModel.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], trust_remote_code=False,
        local_files_only=local_files_only, dtype=torch.float32,
        attn_implementation="sdpa", output_loading_info=True,
    )
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Pretrained DINOv3 weights did not load completely")
    if (model.config.model_type != "dinov3_vit" or model.config.patch_size != 16
            or model.config.hidden_size != 384 or model.config.num_register_tokens != 4):
        raise ValueError("Unexpected ViT-S/16 architecture")
    return model.to(device)


def prepare_weights(output):
    """CPU preflight: check gated access and fingerprint the pinned official weights."""
    from huggingface_hub import hf_hub_download
    from .experiment import new_output, seal_output
    cfg = recipe()
    files = {name: Path(hf_hub_download(cfg["model_id"], name, revision=cfg["model_revision"]))
             for name in ("config.json", "model.safetensors")}
    model = load_backbone("cpu", local_files_only=True)
    output = new_output(output)
    report = {"model_id": cfg["model_id"], "model_revision": cfg["model_revision"],
              "files_sha256": {name: sha256_file(path) for name, path in files.items()},
              "parameters": sum(p.numel() for p in model.parameters()),
              "status": "weights-access-and-load-verified", "paid_compute_started": False}
    write_json(output / "weights.json", report)
    seal_output(output)
    return report


def make_classifier(backbone, mode, rank=8):
    import torch
    from peft import LoraConfig, get_peft_model

    if mode not in ("linear", "lora") or type(rank) is not int or rank < 1:
        raise ValueError("Invalid adaptation mode or rank")
    config = backbone.config
    backbone.requires_grad_(False)
    # Initialize the same head before adapter RNG consumption in either arm.
    head = torch.nn.Linear(2 * config.hidden_size, len(LABELS))
    targets = []
    if mode == "lora":
        targets = [n for n, m in backbone.named_modules() if isinstance(m, torch.nn.Linear)
                   and n.startswith("layer.") and n.endswith(("attention.q_proj", "attention.v_proj"))]
        if len(targets) != 2 * config.num_hidden_layers:
            raise ValueError("DINOv3 attention LoRA target coverage mismatch")
        # Generic PEFT wrapper: do not use language-model task signatures for pixel_values.
        backbone = get_peft_model(backbone, LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias="none", target_modules=targets))

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = head
            self.mode, self.targets = mode, targets

        def train(self, mode=True):
            super().train(mode)
            # eval() does NOT disable gradients. It disables DINO positional jitter,
            # stochastic depth and dropout in both arms, keeping the comparison fixed.
            self.backbone.eval()
            return self

        def forward(self, pixel_values):
            with torch.no_grad() if self.mode == "linear" else nullcontext():
                tokens = self.backbone(pixel_values=pixel_values).last_hidden_state
                patches = tokens[:, 1 + config.num_register_tokens:]
                features = torch.cat((tokens[:, 0], patches.mean(dim=1)), dim=-1)
            return self.head(features.float())

    model = Classifier().to(next(backbone.parameters()).device)
    assert_trainable(model)
    return model


def assert_trainable(model):
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not {"head.weight", "head.bias"}.issubset(trainable):
        raise ValueError("Missing trainable classification head")
    adapter = [n for n in trainable if n.startswith("backbone.")]
    if any(not n.startswith("head.") and not (n.startswith("backbone.") and ".lora_" in n)
           for n in trainable):
        raise ValueError("Unexpected trainable backbone weights")
    if bool(adapter) != (model.mode == "lora"):
        raise ValueError("Adaptation mode does not match trainable parameters")
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "head": sum(p.numel() for p in model.head.parameters()),
            "adapter": sum(p.numel() for n, p in model.named_parameters()
                           if n.startswith("backbone.") and p.requires_grad)}


def save_weights(model, path):
    from safetensors.torch import save_file
    save_file({n: p.detach().cpu().contiguous() for n, p in model.named_parameters()
               if p.requires_grad}, str(path))


def restore_weights(model, path):
    import torch
    from safetensors.torch import load_file
    state = load_file(str(path))
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != set(parameters):
        raise ValueError("Classifier/adapter tensor coverage mismatch")
    with torch.no_grad():
        for name, tensor in state.items():
            if tensor.shape != parameters[name].shape or not torch.isfinite(tensor).all():
                raise ValueError("Invalid saved classifier/adapter tensor")
            parameters[name].copy_(tensor)


def batch_inputs(root, rows, device):
    import torch
    images = []
    for row in rows:
        with Image.open(checked_image(root, row)) as image:
            images.append(preprocess(image))
    return torch.stack(images).to(device), torch.tensor([LABELS.index(r["label"]) for r in rows], device=device)


def synchronize(device):
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def autocast(device):
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("DINOv3 workload time limit reached")


def predict(model, dataset, rows, device, batch_size=8, deadline=None):
    import torch
    model.eval()
    predictions, loss_sum = [], 0.0
    for offset in range(0, len(rows), batch_size):
        check_deadline(deadline)
        part = rows[offset:offset + batch_size]
        synchronize(device)
        started = time.perf_counter()
        pixels, labels = batch_inputs(dataset, part, device)
        synchronize(device)
        forward_start = time.perf_counter()
        with torch.inference_mode(), autocast(device):
            logits = model(pixels).float()
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite validation logits")
            loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            probabilities = logits.softmax(dim=-1).cpu().tolist()
        synchronize(device)
        forward_seconds = time.perf_counter() - forward_start
        seconds = time.perf_counter() - started
        for row, probs in zip(part, probabilities):
            predicted = LABELS[max(range(len(LABELS)), key=probs.__getitem__)]
            predictions.append(dict(id=row["id"], label=row["label"], image_sha256=row["image_sha256"],
                prediction=predicted, probabilities=probs, response=predicted, truncated=False,
                seconds=seconds / len(part), forward_seconds=forward_seconds / len(part)))
    score = metrics(predictions)
    score.update(cross_entropy=loss_sum / len(rows), batch_size=batch_size,
                 mean_end_to_end_seconds=sum(p["seconds"] for p in predictions) / len(rows),
                 mean_forward_seconds=sum(p["forward_seconds"] for p in predictions) / len(rows),
                 timing="Includes checksum/decode/preprocess/transfer; forward includes argmax probabilities transfer; no warmup")
    return predictions, score


def train(dataset, output, mode, seed=17, learning_rate=None, epochs=None, device="cuda",
          timeout_seconds=None):
    import torch
    from transformers import set_seed
    from .experiment import new_output, provenance, seal_output

    cfg = recipe()
    epochs = cfg["epochs"] if epochs is None else epochs
    cap = cfg["max_work_seconds"] if timeout_seconds is None else timeout_seconds
    rates = cfg.get(f"{mode}_learning_rates", [])
    learning_rate = rates[0] if learning_rate is None and rates else learning_rate
    if (mode not in cfg["modes"] or type(seed) is not int or seed not in cfg["seeds"]
            or type(epochs) is not int or not 1 <= epochs <= cfg["epochs"]
            or type(cap) is not int or not 1 <= cap <= cfg["max_work_seconds"]
            or learning_rate not in rates or device not in ("cuda", "cpu")):
        raise ValueError("Training arguments outside the declared DINOv3 recipe")
    rows = check_data(dataset)
    output = new_output(output)
    started = time.monotonic()
    deadline = started + cap
    try:
        set_seed(seed)
        backbone = load_backbone(device)
        model = make_classifier(backbone, mode, cfg["rank"])
        counts = assert_trainable(model)
        config = {**cfg, "mode": mode, "seed": seed, "epochs": epochs,
                  "learning_rate": learning_rate, "device": device, "timeout_seconds": cap}
        info = provenance(dataset, cfg["model_revision"], config)
        info.update(model_id=cfg["model_id"], prompt=None, model_family="dinov3",
                    validation_sha256=cfg["validation_sha256"], parameter_counts=counts,
                    backbone_training_mode=False, attention="sdpa", precision="bf16_autocast_fp32_parameters" if device == "cuda" else "fp32")
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(cfg["model_id"], "model.safetensors", revision=cfg["model_revision"])
        if device == "cuda" and not isinstance(cached, str):
            raise ValueError("Cannot fingerprint cached official base weights")
        info["base_weights_sha256"] = sha256_file(cached) if isinstance(cached, str) else None
        write_json(output / "run.json", info)
        groups = [{"params": list(model.head.parameters()),
                   "lr": learning_rate if mode == "linear" else cfg["lora_head_learning_rate"]}]
        if mode == "lora":
            groups.append({"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": learning_rate})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
        generator = torch.Generator().manual_seed(seed)
        history, best, seen, updates = [], -math.inf, 0, 0
        adapter_updates = 0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(rows["train"]), generator=generator).tolist()
            loss_sum, epoch_start = 0.0, time.monotonic()
            for offset in range(0, len(order), cfg["batch_size"]):
                check_deadline(deadline)
                part = [rows["train"][i] for i in order[offset:offset + cfg["batch_size"]]]
                pixels, labels = batch_inputs(dataset, part, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device):
                    loss = torch.nn.functional.cross_entropy(model(pixels).float(), labels)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                trainable = [p for p in model.parameters() if p.requires_grad]
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
                    raise ValueError("Missing/non-finite trainable gradients")
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise ValueError("Frozen backbone received gradients")
                if mode == "lora" and any(p.grad.abs().sum().item() > 0
                                         for p in model.backbone.parameters() if p.requires_grad):
                    adapter_updates += 1
                torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                loss_sum += loss.item() * len(part)
                seen += len(part)
                updates += 1
            synchronize(device)
            training_seconds = time.monotonic() - epoch_start
            predictions, score = predict(model, dataset, rows["validation"], device, cfg["batch_size"], deadline)
            history.append(dict(epoch=epoch, train_loss=loss_sum / len(order), validation=score,
                                examples_seen=seen, optimizer_steps=updates, training_seconds=training_seconds))
            # Strict improvement preserves the earliest epoch on a tie.
            if score["macro_f1"] > best:
                best, selected_epoch = score["macro_f1"], epoch
                save_weights(model, output / "selected.safetensors")
                write_jsonl(output / "predictions.jsonl", predictions)
                write_json(output / "metrics.json", score)
            write_json(output / "history.json", history)
        if mode == "lora" and not adapter_updates:
            raise ValueError("No nonzero LoRA gradient observed")
        assert_trainable(model)
        write_json(output / "training.json", dict(status="completed", selected_epoch=selected_epoch,
            selected_macro_f1=best, examples_seen=seen, optimizer_steps=updates,
            nonzero_adapter_gradient_steps=adapter_updates, elapsed_seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
            test_evaluated=False, holdout_accessed=False))
    except BaseException as exc:
        write_json(output / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        raise
    else:
        write_json(output / "status.json", {"status": "completed", "quality_claim": "validation_only"})
    finally:
        seal_output(output)
    return output


def evaluate(dataset, trained, output, device="cuda"):
    """Reload selected weights and measure validation at batch size one."""
    import torch
    from huggingface_hub import try_to_load_from_cache
    from transformers import set_seed
    from .confirmation import verify_tree
    from .experiment import new_output, provenance, seal_output
    rows = check_data(dataset)["validation"]
    trained = Path(trained)
    hashes = verify_tree(trained)
    info = json.loads((trained / "run.json").read_text())
    cfg = recipe()
    if (json.loads((trained / "status.json").read_text())["status"] != "completed"
            or info["model_id"] != cfg["model_id"] or info["model_revision"] != cfg["model_revision"]
            or info["dataset_sha256"] != cfg["dataset_sha256"]
            or info["validation_sha256"] != cfg["validation_sha256"]):
        raise ValueError("Training result identity or completion mismatch")
    run = info["config"]
    for key in ("image_size", "image_mean", "image_std", "pooling", "rank"):
        if run[key] != cfg[key]:
            raise ValueError("Training preprocessing/adapter recipe differs: " + key)
    set_seed(run["seed"])
    model = make_classifier(load_backbone(device), run["mode"], run["rank"])
    cached = try_to_load_from_cache(cfg["model_id"], "model.safetensors", revision=cfg["model_revision"])
    base_digest = sha256_file(cached) if isinstance(cached, str) else None
    if base_digest != info["base_weights_sha256"] or (device == "cuda" and base_digest is None):
        raise ValueError("Reloaded base weights differ from training")
    restore_weights(model, trained / "selected.safetensors")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    predictions, score = predict(model, dataset, rows, device, batch_size=1)
    score["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device == "cuda" else None
    output = new_output(output)
    evaluation_config = {**run, "device": device, "evaluation_batch_size": 1}
    current = provenance(dataset, cfg["model_revision"], evaluation_config)
    write_json(output / "run.json", {**info, **current, "model_id": cfg["model_id"], "prompt": None,
            "training_source_sha": info["source_sha"], "training_gpu": info["gpu"],
            "selected_weights_sha256": hashes["selected.safetensors"]})
    write_jsonl(output / "predictions.jsonl", predictions)
    write_json(output / "metrics.json", score)
    write_json(output / "status.json", {"status": "completed", "quality_claim": "validation_only"})
    seal_output(output)
    return score
