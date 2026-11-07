"""Stage only hash-verified checkpoint-100 from the successful public-data experiment."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

ARCHIVE_SHA = 'e7e4fad8a89af270278b72529db333f1698d8c027e761a6629a0464cac32633c'
PREFIX = 'results/baseline/train/'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare(archive, destination):
    if digest(archive) != ARCHIVE_SHA:
        raise ValueError('Parent archive SHA256 mismatch')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as tar:
        manifest = json.load(tar.extractfile(PREFIX + 'checksums.json'))
        selected = [m for m in tar.getmembers() if m.name == PREFIX + 'run.json'
                    or m.name.startswith(PREFIX + 'checkpoints/checkpoint-100/')]
        hashes = {}
        for member in selected:
            rel = PurePosixPath(member.name[len(PREFIX):])
            if member.isdir():
                continue
            if (not member.isfile() or rel.is_absolute() or '..' in rel.parts
                    or str(rel) in hashes or member.size > 100_000_000):
                raise ValueError('Unsafe resume archive')
            path = destination / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open('wb') as target:
                shutil.copyfileobj(source, target)
            hashes[str(rel)] = digest(path)
            if manifest.get(str(rel)) != hashes[str(rel)]:
                raise ValueError('Checkpoint checksum mismatch')
    required = {'adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt',
                'rng_state.pth','trainer_state.json','training_args.bin'}
    if not required <= {p.name for p in (destination / 'checkpoints/checkpoint-100').iterdir()}:
        raise ValueError('Incomplete checkpoint')
    state = json.loads((destination / 'checkpoints/checkpoint-100/trainer_state.json').read_text())
    if state['global_step'] != 100:
        raise ValueError('Expected checkpoint-100')
    (destination / 'checksums.json').write_text(json.dumps(hashes, indent=2))
    (destination / 'resume-origin.json').write_text(json.dumps({'run_id':42385615853,
        'artifact_id':13582826145,'archive_sha256':ARCHIVE_SHA,'global_step':100},indent=2))
    return hashes


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.archive,a.output),indent=2))
