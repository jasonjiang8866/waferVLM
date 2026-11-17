"""Secret-free, one-shot import/readback inside an attached CPU Pod."""
import json
import sys
from pathlib import Path

from scripts.collect_wafer_data import acquire, ingest, plan, verify_volume

ROOT = Path('/tmp/wafer-storage')
DATASETS = ['carinthia-s', 'silicon-wafer-pl', 'silicon-wafer-ebsd']


def main():
    mode, volume_id, mount = sys.argv[1:]
    selected, sizes = plan(DATASETS)
    try:
        if mode == 'import':
            cache = ROOT / 'cache'
            acquire(selected, cache)
            receipts = ingest(selected, cache, Path(mount), volume_id)
            result = {'status': 'mounted_readback_verified', 'receipts': receipts}
        elif mode == 'verify':
            result = verify_volume(selected, Path(mount), volume_id)
        else:
            raise ValueError('Unknown operation')
        result['sizes'] = sizes
    except Exception as exc:
        # No raw API response or credentials exist on this worker.
        result = {'status': 'failed', 'error_type': type(exc).__name__}
    target = ROOT / 'result.json'
    # Atomic writes are safe on this LOCAL ephemeral disk, not on the global mount.
    tmp = ROOT / 'result.tmp'
    tmp.write_text(json.dumps(result))
    tmp.replace(target)


if __name__ == '__main__':
    main()
