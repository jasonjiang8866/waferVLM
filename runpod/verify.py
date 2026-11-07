"""Verify untrusted archive paths, hashes, provenance, numerical and completion gates."""
import hashlib
import json
import math
import tarfile
from pathlib import Path, PurePosixPath

from runpod.provider import IMAGE, SOURCE


def verify_archive(path, controller_sha):
    with tarfile.open(path, 'r:gz') as tar:
        files = {}
        total = 0
        for member in tar:
            p = PurePosixPath(member.name)
            if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'results':
                raise ValueError('Unsafe archive path')
            if member.isdir():
                continue
            if not member.isfile() or member.name in files:
                raise ValueError('Unsafe archive member')
            total += member.size
            if total > 4 * 1024**3 or len(files) > 10000:
                raise ValueError('Archive exceeds limits')
            h = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            files[member.name] = (member, h.hexdigest())

        def read(name):
            member = files['results/' + name][0]
            if member.size > 8_000_000:
                raise ValueError('JSON exceeds limit')
            return json.load(tar.extractfile(member))

        for name in files:
            if name.endswith('/checksums.json'):
                rel = name[len('results/'):]
                checks = read(rel)
                if not checks:
                    raise ValueError('Empty checksum manifest')
                for child, digest in checks.items():
                    p = PurePosixPath(child)
                    if p.is_absolute() or '..' in p.parts:
                        raise ValueError('Unsafe checksum path')
                    if files[str(PurePosixPath(name).parent / p)][1] != digest:
                        raise ValueError('Checksum mismatch')
        worker = read('worker.json')
        if (worker['status'] != 'accepted' or worker['source_sha'] != SOURCE
                or worker['controller_sha'] != controller_sha or worker['profiles'] != ['smoke', 'baseline']):
            raise ValueError('Worker provenance mismatch')
        local = Path(__file__).resolve().parents[1]
        expected = json.loads((local / 'docs/evidence/wm811k-lot17.json').read_text())
        if read('acquisition.json') != expected:
            raise ValueError('Dataset acquisition mismatch')
        if worker['config_sha256'] != hashlib.sha256((local / 'configs/baseline.json').read_bytes()).hexdigest():
            raise ValueError('Worker config mismatch')
        for profile, steps, split in [('smoke', 8, 'validation'), ('baseline', 100, 'test')]:
            for sub in ('', '/base', '/train', '/tuned', '/comparison'):
                if not read(profile + sub + '/checksums.json'):
                    raise ValueError('Required checksum manifest missing')
            if read(profile + '/status.json')['status'] != 'completed':
                raise ValueError('Incomplete profile')
            tr = read(profile + '/train/training.json')
            if (tr['gradient_checks'] != steps or tr['nonzero_gradient_checks'] < 1
                    or not math.isfinite(tr['train_loss']) or tr['peak_allocated_bytes'] <= 0):
                raise ValueError('Training or CUDA gate failed')
            for part in ('base', 'train', 'tuned'):
                r = read(profile + '/' + part + '/run.json')
                if (r['source_sha'] != SOURCE or r['image_digest'] != IMAGE.split('@')[1]
                        or r['dataset_sha256'] != expected['dataset_sha256'] or not r['gpu']
                        or r['model_revision'] != '2fc06364715b967f1860aea9cf38778875588b17'
                        or r['config']['device'] != 'cuda'):
                    raise ValueError('Model provenance mismatch')
                if part != 'train' and r['config']['split'] != split:
                    raise ValueError('Evaluation split mismatch')
            comparison = read(profile + '/comparison/comparison.json')
            for part in ('baseline', 'tuned'):
                if comparison[part]['n'] != 160:
                    raise ValueError('Incomplete evaluation')
                for metric in ('macro_f1', 'accuracy'):
                    if not 0 <= comparison[part][metric] <= 1:
                        raise ValueError('Invalid classification metric')
            if not any(n.startswith('results/' + profile + '/train/adapter/') and n.endswith('.safetensors') for n in files):
                raise ValueError('Missing adapter weights')
        return {'status': 'accepted', 'file_count': len(files), 'unpacked_bytes': total}
