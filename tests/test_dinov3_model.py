"""Real tiny DINOv3/PEFT optimization and reload. No pretrained downloads/GPU."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from wafervlm.core import LABELS, sha256_file

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("peft") is not None


@unittest.skipUnless(HAS_TORCH, "Install .[train] for real DINOv3 compatibility tests")
class DinoModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2)

    @staticmethod
    def backbone(*args, **kwargs):
        import torch
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        with torch.random.fork_rng():
            torch.manual_seed(7)
            return DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=2, num_register_tokens=4,
                pos_embed_rescale=2.0))

    def test_preprocessing_retains_edge_pixels_and_palette(self):
        import torch
        from wafervlm.dinov3 import preprocess, recipe
        image = Image.new("RGB", (256, 256), (0, 90, 180))
        image.putpixel((255, 0), (255, 220, 0))
        tensor = preprocess(image)
        cfg = recipe()
        recovered = (tensor * torch.tensor(cfg["image_std"])[:, None, None]
                     + torch.tensor(cfg["image_mean"])[:, None, None]) * 255
        torch.testing.assert_close(recovered[:, 0, 255], torch.tensor([255., 220., 0.]), atol=0.0001, rtol=0)
        self.assertEqual(tensor.shape, (3, 256, 256))
        with self.assertRaises(ValueError):
            preprocess(image.resize((224, 224)))

    def test_both_modes_update_only_allowed_weights_and_reload_identically(self):
        import torch
        from wafervlm.dinov3 import make_classifier, assert_trainable, save_weights, restore_weights
        pixels = torch.randn(2, 3, 256, 256)
        initial_heads = []
        for mode in ("linear", "lora"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(17)
                model = make_classifier(self.backbone(), mode, 2).train()
                self.assertFalse(model.backbone.training)
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                initial_heads.append(before["head.weight"])
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
                loss = torch.nn.functional.cross_entropy(model(pixels), torch.tensor([0, 1]))
                loss.backward()
                optimizer.step()
                self.assertFalse(torch.equal(before["head.weight"], model.head.weight))
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        torch.testing.assert_close(before[name], p, rtol=0, atol=0)
                        self.assertIsNone(p.grad)
                if mode == "lora":
                    self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters() if "lora_" in n))
                self.assertEqual(assert_trainable(model)["adapter"] > 0, mode == "lora")
                model.eval()
                with torch.inference_mode():
                    expected = model(pixels)
                    self.assertFalse(torch.equal(expected, model(torch.zeros_like(pixels))))
                path = Path(tmp) / "selected.safetensors"
                save_weights(model, path)
                restored = make_classifier(self.backbone(), mode, 2).eval()
                restore_weights(restored, path)
                with torch.inference_mode():
                    torch.testing.assert_close(expected, restored(pixels), rtol=0, atol=0)
        torch.testing.assert_close(*initial_heads, rtol=0, atol=0)

    def test_pooling_excludes_register_tokens(self):
        import torch
        from types import SimpleNamespace
        from wafervlm.dinov3 import make_classifier
        backbone = self.backbone()
        model = make_classifier(backbone, "linear").eval()
        tokens = torch.zeros(1, 7, 32)
        tokens[:, 0] = 1
        tokens[:, 1:5] = 10000
        tokens[:, 5:] = 3
        with patch.object(backbone, "forward", return_value=SimpleNamespace(last_hidden_state=tokens)):
            actual = model(torch.zeros(1, 3, 256, 256))
        expected = model.head(torch.cat((torch.ones(1, 32), torch.full((1, 32), 3.)), dim=1))
        torch.testing.assert_close(actual, expected)

    def test_real_training_selection_reload_and_tamper_rejection(self):
        from wafervlm.dinov3 import train, evaluate, recipe
        from wafervlm.confirmation import verify_tree
        from wafervlm.core import metrics, read_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data"
            dataset.mkdir()
            (dataset / "dataset.json").write_text("{}")
            rows = {}
            for split in ("train", "validation"):
                rows[split] = []
                for i, label in enumerate(LABELS):
                    path = dataset / f"{split}-{i}.png"
                    image = Image.new("RGB", (256, 256), (0, 90, 180))
                    for y in range(16 * i, 16 * i + 16):
                        for x in range(256):
                            image.putpixel((x, y), (255, 220, 0))
                    image.save(path)
                    rows[split].append(dict(id=f"{split}-{i}", split=split, label=label,
                        image=path.name, image_sha256=sha256_file(path)))
            cfg = {**recipe(), "dataset_sha256": sha256_file(dataset / "dataset.json")}
            with patch("wafervlm.dinov3.check_data", return_value=rows), \
                    patch("wafervlm.dinov3.load_backbone", side_effect=self.backbone), \
                    patch("wafervlm.dinov3.recipe", return_value=cfg):
                for mode in ("linear", "lora"):
                    result = root / mode
                    train(dataset, result, mode, epochs=2, device="cpu")
                    verify_tree(result)
                    training = json.loads((result / "training.json").read_text())
                    history = json.loads((result / "history.json").read_text())
                    selected = max(history, key=lambda r: r["validation"]["macro_f1"])
                    self.assertEqual(training["selected_epoch"], selected["epoch"])
                    self.assertEqual(training["examples_seen"], 16)
                    self.assertEqual(training["optimizer_steps"], 2)
                    score = evaluate(dataset, result, root / f"{mode}-eval", "cpu")
                    self.assertEqual(score["n"], 8)
                    self.assertEqual(score["batch_size"], 1)
                    self.assertEqual(score["invalid_rate"], 0)
                    self.assertEqual(metrics(read_jsonl(result / "predictions.jsonl"))["macro_f1"], training["selected_macro_f1"])
                    (result / "selected.safetensors").write_bytes(b"tampered")
                    with self.assertRaisesRegex(ValueError, "hash coverage"):
                        evaluate(dataset, result, root / f"{mode}-rejected", "cpu")

    def test_timeout_preserves_failure_and_seal(self):
        from wafervlm.dinov3 import train
        from wafervlm.confirmation import verify_tree
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("wafervlm.dinov3.check_data", return_value={}), \
                    patch("wafervlm.dinov3.load_backbone", side_effect=TimeoutError("expired")):
                with self.assertRaises(TimeoutError):
                    train(root, root / "result", "linear", device="cpu")
            verify_tree(root / "result")
            self.assertEqual(json.loads((root / "result/status.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
