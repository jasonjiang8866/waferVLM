"""Real offline Trainer/AdamW/RNG restoration for both new seeds, using tiny random Qwen."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


class TrainerConfirmationTests(unittest.TestCase):
    def test_both_seeds_restore_rng_adam_scheduler_and_lower_lr(self):
        import torch
        from safetensors.torch import load_file
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from wafervlm.experiment import train

        torch.set_num_threads(2)
        config = Qwen3_5Config(
            text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                layer_types=['linear_attention', 'full_attention'],
                linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                rope_parameters=dict(rope_type='default', rope_theta=10000,
                                     partial_rotary_factor=.5, mrope_section=[1,1,2])),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                out_hidden_size=32, patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                num_position_embeddings=16, deepstack_visual_indexes=[]),
            image_token_id=120, video_token_id=121, vision_start_token_id=122, vision_end_token_id=123)

        class Processor:
            def save_pretrained(self, destination):
                Path(destination, 'processor.json').write_text('{}')

        def loader(*args):
            # All training seeds start with identical base weights; only LoRA/data RNG varies.
            with torch.random.fork_rng():
                torch.manual_seed(7)
                model = Qwen3_5ForConditionalGeneration(config)
            return model, Processor()

        def collate(rows):
            i = rows[0]['sample']
            return dict(input_ids=torch.tensor([[10,122,120,123,11+i%3,12]]),
                        attention_mask=torch.ones(1,6,dtype=torch.long),
                        pixel_values=torch.full((4,3*2*16*16), .01*(i+1)),
                        image_grid_thw=torch.tensor([[1,2,2]]),
                        labels=torch.tensor([[-100,-100,-100,-100,11+i%3,12]]))

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); data = root/'data';data.mkdir();(data/'dataset.json').write_text('{}')
            stack.enter_context(patch('wafervlm.experiment.load_model', side_effect=loader))
            stack.enter_context(patch('wafervlm.experiment.audit_dataset', return_value={'train':[{'sample':i} for i in range(16)]}))
            stack.enter_context(patch('wafervlm.experiment.AnswerCollator', return_value=collate))
            final = {}
            for seed in (42,123):
                common = dict(dataset=data, seed=seed, accumulation=2, rank=2, device='cpu')
                initial=root/f'{seed}-initial'
                train(output=initial, steps=2, **common)
                # Repeated restoration must reproduce RNG, optimizer and data cursor in a deterministic CPU fixture.
                for attempt in ('a','b'):
                    lower=root/f'{seed}-{attempt}-lower'
                    train(output=lower,steps=4,resume_from=initial/'checkpoints/checkpoint-2',
                          learning_rate=.00005,lower_learning_rate=True,**common)
                    last=root/f'{seed}-{attempt}-last'
                    train(output=last,steps=5,resume_from=lower/'checkpoints/checkpoint-4',
                          learning_rate=.00005,**common)
                    report=json.loads((last/'training.json').read_text())
                    self.assertEqual(report['optimizer_steps_at_resume'],[4])
                    self.assertEqual(report['scheduler_step_at_resume'],4)
                    self.assertEqual(report['learning_rate_trace'],[{'step':5,'rates':[.00005,.00005]}])
                    self.assertTrue(report['frozen_parameters_verified'])
                    opt=torch.load(last/'checkpoints/checkpoint-5/optimizer.pt',weights_only=True)
                    self.assertTrue(all(float(v['step'])==5 for v in opt['state'].values()))
                    final[seed,attempt]=load_file(last/'adapter/adapter_model.safetensors')
                for name in final[seed,'a']:
                    torch.testing.assert_close(final[seed,'a'][name],final[seed,'b'][name],atol=0,rtol=0)
            self.assertTrue(any(not torch.equal(final[42,'a'][k],final[123,'a'][k]) for k in final[42,'a']))


if __name__ == '__main__':
    unittest.main()
