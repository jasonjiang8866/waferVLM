"""Exercise all four size-study exports and the exact worker bootstrap offline."""
import ast
import base64
import json
import lzma
import shutil
import tempfile
import unittest
from pathlib import Path

from runpod import dinov3_size17 as size
from runpod.dinov3_study_controller import ROOT, overlay, check_create_payload
from wafervlm.core import metrics, write_json, write_jsonl
from wafervlm.experiment import seal_output


class SizePilotTests(unittest.TestCase):
    def test_bootstrap_and_recipe(self):
        code, _ = overlay('size17')
        tree = ast.parse(code)
        literal = next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
                       and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
        files = json.loads(lzma.decompress(base64.b64decode(literal)))
        self.assertIn('runpod/dinov3_size_worker.py',files)
        self.assertIn('src/wafervlm/dinov3_backbones.py',files)
        self.assertFalse(any('holdout' in p for p in files))
        check_create_payload(dict(cmd=[code]))
        for backbone, mode in size.ARMS:
            args = size.train_args(backbone, mode,'data','out')
            self.assertEqual(args[args.index('--backbone-size')+1],backbone)
            self.assertEqual(float(args[args.index('--learning-rate')+1]),.003 if mode=='linear' else .0001)

    def test_cpu_acceptance_rejects_wrong_backbone_and_preserves_loc_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launch = dict(controller_sha='a'*40, image='image@sha256:'+'f'*64, gpu=size.GPU, phase='size17')
            write_json(root/'hardware.json',dict(gpu=size.GPU,bf16_matmul_passed=True))
            shutil.copyfile(ROOT/'docs/showcase/dinov3-seed17/validation.jsonl',root/'validation.jsonl')
            reference = size.reference_for('validation')
            for backbone,mode in size.ARMS:
                folder = root/'models'/(backbone+'-'+mode)
                folder.mkdir(parents=True)
                cfg = size.recipe(backbone)
                info = dict(model_id=cfg['model_id'],model_revision=cfg['model_revision'],gpu=size.GPU,
                    base_weights_sha256=size.BACKBONES[backbone].weights_sha256,source_sha=launch['controller_sha'],
                    image_digest=launch['image'].split('@')[1],dataset_sha256=cfg['dataset_sha256'],
                    validation_sha256=cfg['validation_sha256'],parameter_counts={},
                    config=dict(cfg,seed=17,mode=mode,device='cuda',learning_rate=size.plan()['learning_rates'][mode]))
                rows=[]
                for row in reference:
                    pred='scratch' if backbone=='small' and row['label']=='loc' else row['label']
                    rows.append(dict(row,prediction=pred,response=pred,truncated=False,seconds=.1,forward_seconds=.01,
                        probabilities=[float(label==pred) for label in size.plan()['label_order']]))
                score = metrics(rows)
                write_jsonl(folder/'predictions.jsonl',rows)
                write_json(folder/'metrics.json',dict(score,mean_end_to_end_seconds=.1))
                write_json(folder/'training.json',dict(status='completed',examples_seen=7900,optimizer_steps=990,
                    test_evaluated=False,holdout_accessed=False,nonzero_adapter_gradient_steps=990 if mode=='lora' else 0,
                    selected_epoch=1,selected_macro_f1=score['macro_f1'],elapsed_seconds=2,peak_allocated_bytes=1000))
                write_json(folder/'history.json',[dict(epoch=i,optimizer_steps=99*i,examples_seen=790*i,validation=score)
                                                  for i in range(1,11)])
                write_json(folder/'run.json',info)
                write_json(folder/'status.json',dict(status='completed'))
                (folder/'selected.safetensors').write_bytes(b'fixture')
                seal_output(folder)
            accepted=size.verify(root,launch)
            self.assertEqual(accepted['validation_predictions'],640)
            pair=accepted['comparisons']['lora']
            self.assertEqual(sum(pair['counts'].values()),160)
            self.assertEqual(pair['loc_ids']['base_only'],[r['id'] for r in reference if r['label']=='loc'])
            info['base_weights_sha256']=size.BACKBONES['small'].weights_sha256
            write_json(folder/'run.json',info)
            seal_output(folder)
            with self.assertRaisesRegex(ValueError,'training export'):
                size.verify(root,launch)

    def test_confirmation_bootstrap_and_distinct_training_seeds(self):
        code, _ = overlay('size-confirmation')
        check_create_payload(dict(cmd=[code]))
        self.assertIn('runpod.dinov3_size_worker', code)
        arms = size.training_arms('size-confirmation')
        self.assertEqual([(s, b, m) for s, b, m, _ in arms],
                         [(s, 'base', m) for s in (42, 123) for m in ('linear', 'lora')])
        self.assertEqual(len({name for _, _, _, name in arms}), 4)
        for seed, backbone, mode, name in arms:
            args = size.train_args(backbone, mode, 'data', 'out', seed)
            self.assertEqual(name, f'seed{seed}-base-{mode}')
            self.assertEqual(args[args.index('--seed')+1], str(seed))

    def test_confirmation_report_keeps_seed_pairs_separate(self):
        from unittest.mock import patch
        from runpod import dinov3_size_confirmation as confirm
        ref = size.reference_for('validation')
        truth = [r['label'] for r in ref]
        wrong = ['center' if y != 'center' else 'loc' for y in truth]
        scores, predictions, small_scores, small_labels = {}, {}, {}, {}
        for seed, _, mode, name in size.training_arms('size-confirmation'):
            small = name.replace('-base-', '-small-')
            predictions[name], small_labels[small] = (truth,wrong) if seed == 42 else (wrong,truth)
            for key, source, destination in ((small,small_labels,small_scores),(name,predictions,scores)):
                destination[key] = metrics([dict(label=y,prediction=z) for y,z in zip(truth,source[key])])
        with patch.object(confirm, 'small_models', return_value=(small_scores,small_labels)):
            result = confirm.confirmation_report(ref, scores, predictions, None)
        self.assertEqual(result['positive_macro_f1_seeds']['lora'], [17,42])
        self.assertEqual(result['comparisons']['42']['linear']['counts']['base_only'],160)
        self.assertEqual(result['comparisons']['123']['lora']['counts']['small_only'],160)
        self.assertEqual(result['holdout_predictions'],0)
