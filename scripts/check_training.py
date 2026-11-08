"""Real Trainer/collator/checkpoint integration with random tiny weights and fixture data.

Downloads pinned tokenizer/config files only. No pretrained weights or GPU needed.
This is a compatibility check, never a wafer-accuracy benchmark.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

from wafervlm.core import LABELS, sha256_file
from wafervlm.data import prepare
from wafervlm.experiment import train, evaluate
from wafervlm.modeling import MODEL_ID, MODEL_REVISION


def main():
    torch.set_num_threads(2)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    text = config.text_config
    text.hidden_size = 32
    text.intermediate_size = 64
    text.num_hidden_layers = 2
    text.num_attention_heads = 2
    text.num_key_value_heads = 1
    text.head_dim = 16
    text.layer_types = ["linear_attention", "full_attention"]
    text.linear_num_key_heads = text.linear_num_value_heads = 2
    text.linear_key_head_dim = text.linear_value_head_dim = 16
    text.rope_parameters = dict(rope_type="default", rope_theta=10000,
                                partial_rotary_factor=0.5, mrope_section=[1, 1, 2])
    vision = config.vision_config
    vision.depth = 1
    vision.hidden_size = vision.out_hidden_size = 32
    vision.intermediate_size = 64
    vision.num_heads = 2
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    def tiny_loader(revision, device):
        assert revision == MODEL_REVISION and device == "cpu"
        return Qwen3_5ForConditionalGeneration(config), processor

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(7)
        rows = [dict(waferMap=rng.integers(0, 3, (12, 12), dtype=np.uint8),
                     lotName=f"lot-{i}", failureType=np.array([[label]]))
                for label in LABELS for i in range(100)]
        pkl = root / "fixture.pkl"
        pd.DataFrame(rows).to_pickle(pkl)
        prepare(pkl, root / "data", sha256_file(pkl), trust_pickle=True, train_cap=1, eval_cap=1)
        with patch("wafervlm.experiment.load_model", side_effect=tiny_loader):
            train(root / "data", root / "run", steps=2, accumulation=2, rank=2, device="cpu")
        with patch("wafervlm.experiment.load_model", side_effect=tiny_loader):
            train(root / "data", root / "resumed", steps=4, accumulation=2, rank=2,
                  device="cpu", resume_from=root / "run/checkpoints/checkpoint-2")
            train(root / "data", root / "full", steps=4, accumulation=2, rank=2, device="cpu")
        resumed = json.loads((root / "resumed/training.json").read_text())
        assert (resumed["start_step"], resumed["end_step"], resumed["executed_steps"]) == (2, 4, 2)
        from safetensors.torch import load_file
        continued = load_file(root / "resumed/adapter/adapter_model.safetensors")
        uninterrupted = load_file(root / "full/adapter/adapter_model.safetensors")
        assert continued.keys() == uninterrupted.keys()
        for name in continued:
            torch.testing.assert_close(continued[name], uninterrupted[name], atol=1e-6, rtol=1e-5)
        with patch("wafervlm.experiment.load_model", side_effect=tiny_loader):
            train(root/"data",root/"low-lr",steps=4,accumulation=2,rank=2,device="cpu",
                  resume_from=root/"run/checkpoints/checkpoint-2",learning_rate=.00005,lower_learning_rate=True)
            train(root/"data",root/"low-lr-resumed",steps=5,accumulation=2,rank=2,device="cpu",
                  resume_from=root/"low-lr/checkpoints/checkpoint-4",learning_rate=.00005)
        for folder,first,last in [("low-lr",3,4),("low-lr-resumed",5,5)]:
            report=json.loads((root/folder/"training.json").read_text())
            assert report['optimizer_steps_at_resume']==[first-1]
            assert [x['step'] for x in report['learning_rate_trace']]==list(range(first,last+1))
            assert all(x['rates'] and all(lr==.00005 for lr in x['rates']) for x in report['learning_rate_trace'])
            saved=torch.load(root/folder/f"checkpoints/checkpoint-{last}/optimizer.pt",weights_only=True)
            assert all(float(v['step'])==last for v in saved['state'].values())
            assert all(g['lr']==.00005 for g in saved['param_groups'])
        print("PASS: explicit lower LR survives real Trainer restore, every optimizer step, save and second resume")
        with patch("wafervlm.experiment.load_model", side_effect=tiny_loader):
            score=evaluate(root/"data",root/"validation",split="validation",adapter=root/"resumed/adapter",
                           device="cpu",include_loss=True,max_new_tokens=2)
        assert score["n"] == 8 and score["answer_tokens"] > 0
        assert 0 <= score["answer_token_loss"] < float('inf')
        print("PASS: teacher-forced answer-token validation loss and generation share the same adapter")
        print("PASS: resumed optimizer/RNG/data position reproduces uninterrupted four-step adapters")
        diagnostics = json.loads((root / "run/training.json").read_text())
        assert diagnostics["gradient_checks"] == 2
        assert diagnostics["nonzero_gradient_checks"] == 2
        assert (root / "run/adapter/adapter_model.safetensors").is_file()
        assert (root / "run/checkpoints/checkpoint-2/trainer_state.json").is_file()
        print("PASS: real Trainer, official processor, answer mask, accumulation, gradient checkpointing, adapter/checkpoint export")


if __name__ == "__main__":
    main()
