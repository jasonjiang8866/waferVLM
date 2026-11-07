import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from runpod.provider import IMAGE, SOURCE
from runpod.verify import verify_archive


class ResultTests(unittest.TestCase):
    def fixture(self):
        root = Path(__file__).resolve().parents[2]
        evidence = json.loads((root / 'docs/evidence/wm811k-lot17.json').read_text())
        config = hashlib.sha256((root / 'configs/baseline.json').read_bytes()).hexdigest()
        values = {
            'worker.json': {'status': 'accepted', 'source_sha': SOURCE, 'controller_sha': 'a' * 40,
                            'profiles': ['smoke', 'baseline'], 'config_sha256': config},
            'acquisition.json': evidence,
        }
        for profile, steps, split in [('smoke', 8, 'validation'), ('baseline', 100, 'test')]:
            values[profile + '/status.json'] = {'status': 'completed'}
            values[profile + '/train/training.json'] = {'gradient_checks': steps,
                'nonzero_gradient_checks': steps, 'train_loss': 1.0, 'peak_allocated_bytes': 100}
            for part in ('base', 'train', 'tuned'):
                values[profile + '/' + part + '/run.json'] = {
                    'source_sha': SOURCE, 'image_digest': IMAGE.split('@')[1],
                    'dataset_sha256': evidence['dataset_sha256'], 'gpu': 'test CUDA GPU',
                    'model_revision': '2fc06364715b967f1860aea9cf38778875588b17',
                    'config': {'device': 'cuda', 'split': split}}
            values[profile + '/comparison/comparison.json'] = {
                name: {'n': 160, 'accuracy': 0.5, 'macro_f1': 0.5} for name in ('baseline', 'tuned')}
            values[profile + '/train/adapter/adapter_model.safetensors'] = 'fixture bytes'
        return values

    def archive(self, values, target, corrupt=None):
        contents = {k: (v.encode() if isinstance(v, str) else json.dumps(v).encode()) for k, v in values.items()}
        for profile in ('smoke', 'baseline'):
            for sub in ('', '/base', '/train', '/tuned', '/comparison'):
                prefix = profile + sub + '/'
                checks = {k[len(prefix):]: hashlib.sha256(v).hexdigest()
                          for k, v in contents.items() if k.startswith(prefix) and not k.endswith('checksums.json')}
                contents[prefix + 'checksums.json'] = json.dumps(checks).encode()
        if corrupt:
            contents[corrupt] += b'corrupt'
        with tarfile.open(target, 'w:gz') as tar:
            for name, data in contents.items():
                member = tarfile.TarInfo('results/' + name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))

    def test_complete_archive_and_numeric_provenance_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.tar.gz'
            self.archive(self.fixture(), path)
            self.assertEqual(verify_archive(path, 'a' * 40)['status'], 'accepted')
            cases = [
                ('baseline/train/training.json', 'gradient_checks', 99),
                ('smoke/train/training.json', 'train_loss', float('nan')),
                ('baseline/base/run.json', 'source_sha', 'b' * 40),
                ('baseline/base/run.json', 'image_digest', 'sha256:' + '0' * 64),
                ('worker.json', 'controller_sha', 'b' * 40),
                ('acquisition.json', 'dataset_sha256', 'bad'),
            ]
            for file, key, value in cases:
                with self.subTest(key=key):
                    values = self.fixture()
                    values[file][key] = value
                    self.archive(values, path)
                    with self.assertRaises(ValueError):
                        verify_archive(path, 'a' * 40)

    def test_corrupt_adapter_and_incomplete_evaluation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.tar.gz'
            self.archive(self.fixture(), path, corrupt='baseline/train/adapter/adapter_model.safetensors')
            with self.assertRaises(ValueError):
                verify_archive(path, 'a' * 40)
            values = self.fixture()
            values['baseline/comparison/comparison.json']['tuned']['n'] = 159
            self.archive(values, path)
            with self.assertRaises(ValueError):
                verify_archive(path, 'a' * 40)
