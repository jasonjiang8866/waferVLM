"""Stage pinned DINOv3 weights/license, then verify loading without a network."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from runpod.dinov3_bundle import recipe

MODEL = 'facebook/dinov3-vits16-pretrain-lvd1689m'
REVISION = '114c1379950215c8b35dfcd4e90a5c251dde0d32'
HASHES = {
    'config.json': '9481247be9f95a134a5599402b4bfc838eecdf9a7fffbf4debd1c70ec213898b',
    'model.safetensors': '4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d',
}
PARAMETERS = 21596544
HF_ROOT = Path('/opt/dinov3-hf')
LICENSE = Path('/opt/dinov3-license/LICENSE.md')
METADATA = Path('/opt/dinov3-metadata')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def snapshot():
    cfg = recipe()
    if (cfg['model_id'], cfg['model_revision']) != (MODEL, REVISION):
        raise ValueError('Recipe differs from pinned image model')
    return HF_ROOT / 'hub' / ('models--' + MODEL.replace('/', '--')) / 'snapshots' / REVISION


def report():
    return dict(model_id=MODEL, model_revision=REVISION, files_sha256=dict(HASHES),
                parameters=PARAMETERS, status='weights-access-and-load-verified', paid_compute_started=False)


def verify_files(expected=None):
    if expected is not None and expected != report():
        raise ValueError('Embedded model differs from controller CPU fingerprint')
    root = snapshot()
    if set(p.name for p in root.iterdir()) != set(HASHES):
        raise ValueError('Unexpected embedded weight file set')
    for name, checksum in HASHES.items():
        path = root / name
        if path.is_symlink() or digest(path) != checksum:
            raise ValueError('Embedded weight checksum mismatch')
    if not LICENSE.is_file() or not 1000 < LICENSE.stat().st_size < 64000:
        raise ValueError('DINOv3 license missing or invalid')
    manifest = json.loads((METADATA / 'manifest.json').read_text())
    if manifest != dict(weights=report(), license_sha256=digest(LICENSE)):
        raise ValueError('Embedded model manifest mismatch')
    if os.environ.get('HF_HOME') != str(HF_ROOT):
        raise ValueError('Offline cache path differs from embedded weights')
    return dict(mode='embedded', verified=True, model_id=MODEL, model_revision=REVISION,
                files_sha256=dict(HASHES), license_sha256=digest(LICENSE))


def stage():
    # All download metadata/cache lives on tmpfs. Only the two reviewed files
    # and original license enter the image, never credentials or signed URLs.
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_XET_CACHE'] = '/tmp/dino-download/xet'
    from huggingface_hub import hf_hub_download
    token = Path('/run/secrets/hf_token').read_text().strip()
    if not token:
        raise ValueError('Build-only Hugging Face token required')
    destination = snapshot()
    destination.mkdir(parents=True, exist_ok=False)
    LICENSE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir='/tmp/dino-download') as temporary:
        for name in (*HASHES, 'LICENSE.md'):
            source = Path(hf_hub_download(MODEL, name, revision=REVISION,
                                         token=token, cache_dir=temporary))
            if name in HASHES and digest(source) != HASHES[name]:
                raise ValueError('Downloaded weight checksum mismatch')
            shutil.copyfile(source, LICENSE if name == 'LICENSE.md' else destination / name)
    METADATA.mkdir(parents=True, exist_ok=False)
    (METADATA / 'manifest.json').write_text(json.dumps(
        dict(weights=report(), license_sha256=digest(LICENSE)), indent=2) + '\n')


def verify_offline():
    credentials = ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN', 'RUNPOD_API_KEY')
    if any(os.environ.get(name) for name in credentials) or Path('/run/secrets/hf_token').exists():
        raise ValueError('Offline verification must have no account credentials')
    receipt = verify_files()
    from wafervlm.dinov3 import load_backbone
    model = load_backbone('cpu', local_files_only=True)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError('Embedded model parameter count mismatch')
    receipt.update(parameters=PARAMETERS, status='offline_model_load_verified', credentials_present=False)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['stage', 'verify'])
    args = parser.parse_args()
    try:
        {'stage': stage, 'verify': verify_offline}[args.mode]()
    except Exception as exc:
        # Remote exception strings may contain signed download URLs.
        raise SystemExit('DINO embedded weights failed: ' + type(exc).__name__) from None


if __name__ == '__main__':
    main()
