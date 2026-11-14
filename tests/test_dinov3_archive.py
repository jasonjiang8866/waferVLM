"""Portable DINO result verification against a synthetic six-trial archive."""
import copy
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runpod.dinov3_bundle import recipe, select, trials, verify_archive
from wafervlm.core import LABELS, metrics, sha256_file, write_json, write_jsonl
from wafervlm.experiment import seal_output

SOURCE = 'a' * 40
IMAGE = 'ghcr.io/jasonjiang8866/wafervlm@sha256:' + 'b' * 64
WEIGHTS = {'files_sha256': {'model.safetensors': 'c' * 64}}


class DinoArchiveTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir()
        cfg = recipe()
        refs = [dict(id=str(i), split='validation', label=label, image_sha256=str(i)*64)
                for i, label in enumerate(LABELS)]
        rows = [{**r, 'prediction': r['label'], 'response': r['label'], 'truncated': False,
                 'probabilities': [float(l == r['label']) for l in LABELS]} for r in refs]
        write_jsonl(root / 'validation.jsonl', refs)
        write_json(root / 'dataset.json', {'synthetic': True})
        cfg['dataset_sha256'] = sha256_file(root / 'dataset.json')
        cfg['validation_sha256'] = sha256_file(root / 'validation.jsonl')
        write_json(root / 'worker.json', dict(status='accepted', source_sha=SOURCE, image=IMAGE, weights=WEIGHTS))
        write_json(root / 'hardware.json', dict(gpu='synthetic GPU', bf16=True))
        evidence = Path(__file__).resolve().parents[1] / 'docs/evidence/wm811k-lot17.json'
        (root / 'acquisition.json').write_bytes(evidence.read_bytes())
        score = metrics(rows)
        for trial in trials():
            folder = root / trial['name']
            folder.mkdir()
            info = dict(model_id=cfg['model_id'], model_revision=cfg['model_revision'],
                        dataset_sha256=cfg['dataset_sha256'], validation_sha256=cfg['validation_sha256'],
                        config={**cfg, **{k:trial[k] for k in ('mode','seed','learning_rate')}, 'device':'cuda'},
                        source_sha=SOURCE, image_digest=IMAGE.split('@')[1], gpu='synthetic GPU',
                        base_weights_sha256=WEIGHTS['files_sha256']['model.safetensors'])
            write_json(folder / 'run.json', info)
            write_json(folder / 'status.json', dict(status='completed'))
            write_jsonl(folder / 'predictions.jsonl', rows)
            write_json(folder / 'metrics.json', score)
            write_json(folder / 'training.json', dict(status='completed', selected_epoch=1,
                selected_macro_f1=1., examples_seen=7900, optimizer_steps=990,
                test_evaluated=False, holdout_accessed=False, nonzero_adapter_gradient_steps=990))
            write_json(folder / 'history.json', [dict(epoch=n, validation=score, optimizer_steps=99*n,
                       examples_seen=790*n) for n in range(1,11)])
            (folder / 'selected.safetensors').write_bytes(b'not loaded by verifier')
            seal_output(folder)
        selected = select([{**t,'macro_f1':1.} for t in trials()])
        write_json(root / 'selection.json', selected)
        for mode, trial in selected.items():
            folder = root / (mode+'-reload')
            folder.mkdir()
            original = root / trial['name']
            info = json.loads((original / 'run.json').read_text())
            info['config']['evaluation_batch_size'] = 1
            info['selected_weights_sha256'] = sha256_file(original / 'selected.safetensors')
            write_json(folder / 'run.json', info)
            write_json(folder / 'status.json', dict(status='completed'))
            write_jsonl(folder / 'predictions.jsonl', rows)
            write_json(folder / 'metrics.json', score)
            seal_output(folder)
        return cfg

    def verify(self, root, cfg):
        archive = root.parent / 'experiment.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(root, arcname='results')
        with patch('runpod.dinov3_bundle.recipe', return_value=cfg), \
                patch('wafervlm.vision_comparison.recipe', return_value=cfg):
            return verify_archive(archive, SOURCE, IMAGE, WEIGHTS)

    def test_archive_relocation_recomputes_scores_and_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'results'
            cfg = self.fixture(root)
            report = self.verify(root, cfg)
            self.assertEqual(report['status'], 'accepted')
            self.assertEqual(report['selection']['linear']['learning_rate'], .001)
            self.assertFalse(report['quality_accepted'])

    def test_resealed_recipe_exposure_holdout_and_provenance_tampering_rejected(self):
        changes = [('run.json', 'source_sha', 'd'*40),
                   ('run.json', 'base_weights_sha256', 'd'*64),
                   ('training.json', 'optimizer_steps', 989),
                   ('training.json', 'holdout_accessed', True),
                   ('training.json', 'selected_epoch', 2)]
        for filename,key,value in changes:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)/'results'
                cfg = self.fixture(root)
                folder = root/'lora-0'
                data = json.loads((folder/filename).read_text())
                data[key] = value
                write_json(folder/filename, data)
                seal_output(folder)
                with self.assertRaises(ValueError):
                    self.verify(root, cfg)

    def test_missing_trial_and_changed_selected_weights_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'results'
            cfg = self.fixture(root)
            info = json.loads((root/'linear-reload/run.json').read_text())
            info['selected_weights_sha256'] = 'd'*64
            write_json(root/'linear-reload/run.json', info)
            seal_output(root/'linear-reload')
            with self.assertRaisesRegex(ValueError, 'reload'):
                self.verify(root, cfg)


if __name__ == '__main__':
    unittest.main()
