"""Stage pinned DINOv3 weights/license, then verify loading without a network."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from src.wafervlm.dinov3_backbones import BACKBONES

SMALL_CONFIG_SHA256 = '9481247be9f95a134a5599402b4bfc838eecdf9a7fffbf4debd1c70ec213898b'
FILES = ('config.json', 'model.safetensors')
HF_ROOT = Path('/opt/dinov3-hf')
LICENSE = Path('/opt/dinov3-license/LICENSE.md')
METADATA = Path('/opt/dinov3-metadata')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def snapshot(size='small'):
    spec = BACKBONES[size]
    return HF_ROOT / 'hub' / ('models--' + spec.model_id.replace('/', '--')) / 'snapshots' / spec.revision


def report(size='small'):
    spec = BACKBONES[size]
    # The historical controller fingerprints Small without a local snapshot.
    config_hash = SMALL_CONFIG_SHA256 if size == 'small' else spec.verify_file('config.json', snapshot(size) / 'config.json')
    return dict(model_id=spec.model_id, model_revision=spec.revision,
                files_sha256={'config.json': config_hash, 'model.safetensors': spec.weights_sha256},
                parameters=spec.parameters, status='weights-access-and-load-verified', paid_compute_started=False)


def verify_files(expected=None, size='small'):
    fingerprint = report(size)
    if expected is not None and expected != fingerprint:
        raise ValueError('Embedded model differs from controller CPU fingerprint')
    root = snapshot(size)
    if set(p.name for p in root.iterdir()) != set(FILES):
        raise ValueError('Unexpected embedded weight file set')
    for name, checksum in fingerprint['files_sha256'].items():
        path = root / name
        if path.is_symlink() or digest(path) != checksum:
            raise ValueError('Embedded weight checksum mismatch')
    if not LICENSE.is_file() or not 1000 < LICENSE.stat().st_size < 64000:
        raise ValueError('DINOv3 license missing or invalid')
    manifest = json.loads((METADATA / 'manifest.json').read_text())
    if manifest != dict(weights=fingerprint, license_sha256=digest(LICENSE)):
        raise ValueError('Embedded model manifest mismatch')
    if os.environ.get('HF_HOME') != str(HF_ROOT):
        raise ValueError('Offline cache path differs from embedded weights')
    return dict(mode='embedded', verified=True, model_id=fingerprint["model_id"], model_revision=fingerprint["model_revision"],
                files_sha256=fingerprint["files_sha256"], license_sha256=digest(LICENSE))


def stage(size="small"):
    spec = BACKBONES[size]
    # All download metadata/cache lives on tmpfs. Only the two reviewed files
    # and original license enter the image, never credentials or signed URLs.
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_XET_CACHE'] = '/tmp/dino-download/xet'
    from huggingface_hub import hf_hub_download
    token = Path('/run/secrets/hf_token').read_text().strip()
    if not token:
        raise ValueError('Build-only Hugging Face token required')
    destination = snapshot(size)
    destination.mkdir(parents=True, exist_ok=False)
    LICENSE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir='/tmp/dino-download') as temporary:
        for name in (*FILES, 'LICENSE.md'):
            source = Path(hf_hub_download(spec.model_id, name, revision=spec.revision,
                                         token=token, cache_dir=temporary))
            if name in FILES:
                spec.verify_file(name, source)
            shutil.copyfile(source, LICENSE if name == 'LICENSE.md' else destination / name)
    METADATA.mkdir(parents=True, exist_ok=False)
    (METADATA / 'manifest.json').write_text(json.dumps(
        dict(weights=report(size), license_sha256=digest(LICENSE)), indent=2) + '\n')


def verify_offline(size="small"):
    credentials = ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN', 'RUNPOD_API_KEY')
    if any(os.environ.get(name) for name in credentials) or Path('/run/secrets/hf_token').exists():
        raise ValueError('Offline verification must have no account credentials')
    receipt = verify_files(size=size)
    from wafervlm.dinov3 import load_backbone, make_classifier, save_weights, restore_weights
    import torch
    torch.set_num_threads(2)
    pixels = torch.zeros(1, 3, 256, 256)
    for mode in ('linear', 'lora'):
        model = make_classifier(load_backbone('cpu', local_files_only=True, backbone_size=size), mode).eval()
        with torch.inference_mode(), tempfile.TemporaryDirectory() as temporary:
            # Give the exported LoRA B tensors a nonzero effect on the forward pass.
            for name, param in model.named_parameters():
                if 'lora_B' in name:
                    param.fill_(0.001)
            logits = model(pixels)
            path = Path(temporary) / 'selected.safetensors'
            save_weights(model, path)
            del model
            restored = make_classifier(load_backbone('cpu', local_files_only=True, backbone_size=size), mode).eval()
            restore_weights(restored, path)
            torch.testing.assert_close(restored(pixels), logits, rtol=0, atol=0)
            del restored
    receipt.update(parameters=BACKBONES[size].parameters, reloaded_modes=['linear', 'lora'], status='offline_model_load_verified', credentials_present=False)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['stage', 'verify'])
    parser.add_argument('--backbone-size', choices=BACKBONES, default='small')
    args = parser.parse_args()
    try:
        {'stage': stage, 'verify': verify_offline}[args.mode](args.backbone_size)
    except Exception as exc:
        # Remote exception strings may contain signed download URLs.
        raise SystemExit('DINO embedded weights failed: ' + type(exc).__name__) from None


if __name__ == '__main__':
    main()
