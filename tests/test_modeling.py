"""Real Transformers/PEFT tests with randomly initialized tiny Qwen3.5; no downloads."""

import importlib.util
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("peft") is not None


@unittest.skipUnless(HAS_TORCH, "Install .[train] for real model compatibility tests")
class ModelTests(unittest.TestCase):
    def test_answer_mask_rejects_wrong_prefix(self):
        import torch
        from wafervlm.modeling import mask_answer

        full = torch.tensor([[1, 2, 3, 4]])
        labels = mask_answer(full, torch.tensor([[1, 2]]))
        self.assertEqual(labels.tolist(), [[-100, -100, 3, 4]])
        with self.assertRaises(ValueError):
            mask_answer(full, torch.tensor([[1, 9]]))
        with self.assertRaises(ValueError):
            mask_answer(full, full)

    def test_multimodal_lora_forward_backward_and_reload(self):
        import torch
        from peft import PeftModel
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from wafervlm.modeling import add_lora, assert_frozen_vision

        torch.set_num_threads(2)
        config = Qwen3_5Config(
            text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                layer_types=["linear_attention", "full_attention"],
                linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                rope_parameters={"rope_type": "default", "rope_theta": 10000,
                                 "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]}),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                out_hidden_size=32, patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                num_position_embeddings=16, deepstack_visual_indexes=[]),
            image_token_id=120, video_token_id=121, vision_start_token_id=122, vision_end_token_id=123,
        )
        torch.manual_seed(7)
        base = Qwen3_5ForConditionalGeneration(config)
        model, targets = add_lora(base, rank=2)
        self.assertTrue(any("linear_attn" in t for t in targets))
        self.assertTrue(any("self_attn" in t for t in targets))
        self.assertFalse(any("visual" in t for t in targets))
        inputs = dict(input_ids=torch.tensor([[10, 122, 120, 123, 11, 12]]),
                      attention_mask=torch.ones(1, 6, dtype=torch.long),
                      pixel_values=torch.randn(4, 3 * 2 * 16 * 16),
                      image_grid_thw=torch.tensor([[1, 2, 2]]), use_cache=False)
        labels = torch.tensor([[-100, -100, -100, -100, 11, 12]])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        loss = model(**inputs, labels=labels).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if ".visual." in n))
        optimizer.step()
        assert_frozen_vision(model)
        model.eval()
        with torch.no_grad():
            expected = model(**inputs).logits
            changed = model(**{**inputs, "pixel_values": inputs["pixel_values"] * 0}).logits
        self.assertFalse(torch.equal(expected, changed), "Model must actually consume the image")
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            torch.manual_seed(7)
            fresh = Qwen3_5ForConditionalGeneration(config)
            restored = PeftModel.from_pretrained(fresh, tmp).eval()
            with torch.no_grad():
                actual = restored(**inputs).logits
            torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
