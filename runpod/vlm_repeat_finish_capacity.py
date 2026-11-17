"""Reconcile the final-two 5090 capacity rejection before any new allocation."""
import json
import re
from pathlib import Path

COOLDOWN_SECONDS = 30 * 60

PIN = dict(
    run_id=44096703333, seed='repeat',
    controller_sha='06606a81b749a5093c1428582eae62f5aa94ac84',
    result_artifact_id=14334089144,
    result_sha256='046053aed3b590ec00475d46a84a7bf75c8bdb69021493ece289ce41b7d2ce9c',
    marker_artifact_id=14334965710,
    marker_sha256='50138a7f065ca8ada90c9bd55a44fec98405d3aaed7459daefbf4907fc60bfe6',
    independent_cleanup_artifact_id=14335530465,
    independent_cleanup_sha256='f8c5d62375c7b3a6caab4d48fc569175068b0fc13a8ed3bb8006db618deb3fb7',
)


def reconcile(output, prior_usd, prior_seconds):
    from runpod import vlm_ledger as ledger, vlm_repeat_capacity as capacity
    from runpod.provider import finite
    from runpod.vlm_repeat_resume import finish_pin, RECEIPT, FINISH_RECEIPT
    output = Path(output)
    receipts = {
        'resume-verification.json': json.loads(RECEIPT.read_text()),
        'finish-verification.json': json.loads(FINISH_RECEIPT.read_text()),
        'capacity-recovery.json': json.loads((output / 'capacity-recovery.json').read_text()),
    }
    p, usd, seconds, elapsed = capacity.reconcile(
        output, prior_usd, prior_seconds, pin=PIN, source_pin=finish_pin(),
        receipts=receipts, evidence_name='finish-capacity-recovery.json')
    raw = ledger.artifact_bytes(p['result_artifact_id'], p['result_sha256'], 100000)
    preflight = ledger.zip_json(raw, 'preflight.json')
    spec = ledger.zip_json(raw, 'launch-spec.json')
    clean = ledger.zip_json(raw, 'cleanup.json')
    regions = preflight.get('regions')
    if (preflight.get('status') != 'preflight_passed'
            or preflight.get('gpu') != 'NVIDIA GeForce RTX 5090'
            or preflight.get('gpu') != spec.get('gpu')
            or preflight.get('cost', {}).get('gpu_hourly_usd') != 1.39
            or not finite(clean.get('checked_epoch'), .001)
            or not isinstance(regions, list) or not 1 <= len(regions) <= 64
            or any(not isinstance(r, str) or not re.fullmatch(r'[A-Za-z0-9-]+', r) for r in regions)
            or len(set(regions)) != len(regions)):
        raise RuntimeError('Invalid rejected-placement evidence')
    rejected = [dict(gpu=preflight['gpu'], region=r,
                     retry_after_epoch=clean['checked_epoch'] + COOLDOWN_SECONDS) for r in regions]
    return p, usd, seconds, elapsed, rejected


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # The preceding verify-finish CI step rechecks this cumulative starting ledger.
    _, usd, seconds, elapsed, rejected = reconcile(
        args.output, 9.472521519981784, 44333.58444801331)
    print(json.dumps(dict(status='verified_capacity_rejection', cumulative_usd=usd,
                         cumulative_seconds=seconds, deducted_work_seconds=elapsed,
                         excluded_placement_count=len(rejected))))
