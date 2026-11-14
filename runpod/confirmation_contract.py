"""Offline launch contract. No credentials, network calls or resource mutations."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_RE = r'ghcr\.io/jasonjiang8866/wafervlm@sha256:[0-9a-f]{64}'
GPU_CAPS = {'NVIDIA GeForce RTX 4090': .89,
            'NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb': .69,
            'NVIDIA GeForce RTX 3090': .50, 'NVIDIA GeForce RTX 5090': 1.19}


def number(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def launch_plan(spec):
    """Bind one fresh seed to an explicit ledger snapshot and immutable image.

    This validates a proposal, NOT live provider state or permission to launch.
    All prior attempts (including failed attempts) must be reconciled upstream.
    """
    cfg = json.loads((ROOT / 'configs/seed-confirmation.json').read_text())
    if type(spec.get('seed')) is not int or spec['seed'] not in cfg['new_seeds']:
        raise ValueError('Only independent training seeds 42/123 are supported')
    for key in ('source_sha', 'controller_sha'):
        if not re.fullmatch('[0-9a-f]{40}', spec.get(key) or ''):
            raise ValueError('Missing exact ' + key)
    if not re.fullmatch(IMAGE_RE, spec.get('image') or ''):
        raise ValueError('Published immutable confirmation image digest is required')
    if spec['image'].endswith('b13a9807ae05698f0813e0d1ab7f25548e782f65729d2b2aec5e66de6de24cd1'):
        raise ValueError('Old extension image does not contain confirmation code')
    ledger = spec['ledger']
    if (not number(ledger.get('usd')) or not number(ledger.get('seconds'))
            or ledger['usd'] < 5.903039 or ledger['seconds'] < 491.911 * 60):
        raise ValueError('Ledger must include all historical consumption')
    attempts = ledger.get('attempts')
    if not isinstance(attempts, list) or not attempts:
        raise ValueError('Explicit reconciled attempt inventory required')
    ids = [a.get('run_id') for a in attempts]
    if any(type(i) is not int or i <= 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Duplicate or invalid ledger run')
    for a in attempts:
        if (a.get('cleanup') != 'verified_absent'
                or type(a.get('marker_artifact_id')) is not int or a['marker_artifact_id'] <= 0
                or not re.fullmatch('[0-9a-f]{64}', a.get('evidence_sha256') or '')):
            raise ValueError('Unreconciled attempt blocks the next allocation')
    if 42682266526 not in ids:
        raise ValueError('Latest continuation attempt is missing')
    if any(a.get('seed') == spec['seed'] for a in attempts):
        raise ValueError('Existing seed attempt requires explicit recovery review')
    if spec['seed'] == 123 and not any(a.get('seed') == 42 and a.get('accepted') is True for a in attempts):
        raise ValueError('Seed 42 must be accepted and reconciled before seed 123')
    gpu, rate = spec.get('gpu'), spec.get('gpu_hourly_usd')
    if gpu not in GPU_CAPS or not number(rate) or not 0 < rate <= GPU_CAPS[gpu]:
        raise ValueError('GPU quote exceeds its authorized ceiling')
    seconds = cfg['train_work_seconds'] + cfg['cleanup_seconds']
    disk_hourly = 60 * .10 / (28 * 24)
    extra = (rate + disk_hourly) * seconds / 3600
    if ledger['usd'] + extra > 30 or ledger['seconds'] + seconds > 1100 * 60:
        raise ValueError('Insufficient aggregate budget for this phase')
    return dict(status='offline_contract_valid_not_launch_authorization', seed=spec['seed'],
                source_sha=spec['source_sha'], controller_sha=spec['controller_sha'], image=spec['image'],
                config_sha256=hashlib.sha256((ROOT/'configs/seed-confirmation.json').read_bytes()).hexdigest(),
                gpu=gpu, gpu_hourly_usd=rate, min_cuda_version='12.8', disk_gb=60,
                work_seconds=cfg['train_work_seconds'], cleanup_seconds=cfg['cleanup_seconds'],
                lifecycle_seconds=seconds, estimated_additional_usd=extra,
                estimated_total_usd=ledger['usd']+extra, total_seconds=ledger['seconds']+seconds,
                current_inventory='unverified', current_quote='unverified', holdout_access=False,
                remaining_requirements=['verify ledger against Actions inventory and artifact bytes',
                                        'verify image provenance and anonymous pull',
                                        'fresh provider quote, owned Pod inventory and account guardrails',
                                        'reviewed lifecycle workflow and independent cleanup'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    args = parser.parse_args()
    print(json.dumps(launch_plan(json.loads(args.spec.read_text())), indent=2))


if __name__ == '__main__':
    main()
