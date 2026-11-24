"""Exercise pinned HF cache lookup with local CPU fixtures at both feature widths."""
import hashlib
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from wafervlm.dinov3_backbones import BACKBONES


@unittest.skipUnless(importlib.util.find_spec("peft"), "requires training dependencies")
class SizeTests(unittest.TestCase):
    def test_offline_loading_heads_lora_reload_and_wrong_backbone(self):
        import torch
        from huggingface_hub import constants
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        from wafervlm.dinov3 import load_backbone, make_classifier, restore_weights, save_weights

        torch.set_num_threads(2)
        pixels = torch.randn(1, 3, 32, 32)
        with tempfile.TemporaryDirectory() as tmp, patch.object(constants, "HF_HUB_CACHE", tmp):
            for size, spec in BACKBONES.copy().items():
                snapshot = Path(tmp, 'models--' + spec.model_id.replace('/', '--'), 'snapshots', spec.revision)
                snapshot.mkdir(parents=True)
                config = DINOv3ViTConfig(hidden_size=spec.hidden_size, num_hidden_layers=1,
                    intermediate_size=128, num_attention_heads=spec.hidden_size // 64, num_register_tokens=4)
                fixture = DINOv3ViTModel(config)
                fixture.save_pretrained(snapshot)
                config_bytes = (snapshot / 'config.json').read_bytes()
                with (snapshot / 'model.safetensors').open('rb') as f:
                    fixture_spec = replace(spec, weights_sha256=hashlib.file_digest(f, 'sha256').hexdigest(),
                        config_blob=hashlib.sha1(b'blob %d\0' % len(config_bytes) + config_bytes).hexdigest(),
                        parameters=sum(p.numel() for p in fixture.parameters()))
                with patch.dict(BACKBONES, {size: fixture_spec}):
                    for mode in ('linear', 'lora'):
                        with self.subTest(size=size, mode=mode):
                            def classifier():
                                return make_classifier(load_backbone('cpu', True, size), mode).eval()
                            model = classifier()
                            self.assertEqual(model.head.in_features, 2 * spec.hidden_size)
                            self.assertEqual(len(model.targets), 2 if mode == 'lora' else 0)
                            model(pixels).sum().backward()
                            torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001).step()
                            expected = model(pixels).detach()
                            output = Path(tmp, f'{size}-{mode}.safetensors')
                            save_weights(model, output)
                            restored = classifier()
                            restore_weights(restored, output)
                            torch.testing.assert_close(restored(pixels), expected, atol=0, rtol=0)
                            if size == 'base':
                                with self.assertRaisesRegex(ValueError, 'tensor'):
                                    restore_weights(restored, Path(tmp, f'small-{mode}.safetensors'))
                with self.assertRaisesRegex(ValueError, 'configuration blob'):
                    load_backbone('cpu', True, size)
                with self.assertRaisesRegex(ValueError, 'weight checksum'):
                    spec.verify_file('model.safetensors', snapshot / 'model.safetensors')
