"""Run the frozen 18-pass evaluator in the existing pinned confirmation image."""
import json
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from http.server import ThreadingHTTPServer

from runpod import vlm_worker as w
from runpod.vlm_repeat_inputs import FROZEN, archives, prepare


def work():
    child = None
    result = dict(status='rejected', profiles=[], source_sha=os.environ['WAFER_SOURCE_SHA'],
                  controller_sha=os.environ['WAFER_CONTROLLER_SHA'],
                  config_sha256=os.environ['WAFER_CONFIG_SHA'],
                  input_manifest_sha256=os.environ['WAFER_INPUT_MANIFEST_SHA'])
    try:
        w.OUT.mkdir(parents=True, exist_ok=False)
        w.write('cpu-start.json', w.cpu_snapshot())
        urls = json.loads(os.environ.pop('WAFER_INPUT_URLS'))
        if (os.environ.get('WAFERVLM_SOURCE_SHA') != result['source_sha']
                or w.digest(w.ROOT / 'configs/seed-confirmation.json') != result['config_sha256']
                or w.digest(FROZEN) != result['input_manifest_sha256']):
            raise ValueError('Pinned image/config/input identity mismatch')
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError('CUDA/BF16 required')
        w.write('hardware.json', dict(gpu=torch.cuda.get_device_name(),
                total_memory=torch.cuda.get_device_properties(0).total_memory,
                torch=torch.__version__, cuda=torch.version.cuda,
                image_digest=os.environ['WAFERVLM_IMAGE_DIGEST']))
        deadline = float(os.environ['WAFER_DEADLINE'])
        env = {k: v for k, v in os.environ.items() if k not in
               {'WAFER_INPUT_URLS', 'WAFER_RESULT_TOKEN', 'RUNPOD_API_KEY', 'GH_TOKEN', 'GITHUB_TOKEN'}}
        env['PYTHONUNBUFFERED'] = '1'
        phases = []

        def run(phase, args, cap):
            nonlocal child
            w.RESULT['phase'] = phase
            remaining = min(cap, deadline - time.time() - 120)
            if remaining < 60:
                raise TimeoutError('Insufficient evaluation time')
            started = time.time()
            with (w.OUT / (phase + '.log')).open('w') as log:
                child = subprocess.Popen([sys.executable, *args], cwd=w.ROOT, env=env,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = child.wait(timeout=remaining)
            phases.append(dict(phase=phase, wall_seconds=time.time()-started, returncode=code))
            w.write('phase-timings.json', phases)
            if code:
                raise RuntimeError('Evaluation subprocess failed: ' + phase)

        dataset = Path('/workspace/wafer-data')
        run('data', ['scripts/prepare_public_data.py', '--trust-pickle',
                    '--cache', '/workspace/wafer-source', '--output', str(dataset)], 1200)
        if w.digest(dataset / 'dataset.json') != os.environ['WAFER_DATASET_SHA']:
            raise ValueError('Dataset identity mismatch')
        w.write('acquisition.json', json.loads((dataset / 'acquisition.json').read_text()))
        w.RESULT['phase'] = 'inputs'
        started = time.time()
        paths = archives('/workspace/repeat-archives', urls, deadline - 180)
        # URLs are never persisted in the result or inherited by evaluation processes.
        del urls
        rows = prepare(paths, '/workspace/repeat-inputs', dataset, json.loads(FROZEN.read_text()))
        endpoints = Path('/workspace/frozen-endpoints.json')
        endpoints.write_text(json.dumps(rows, indent=2))
        phases.append(dict(phase='inputs', wall_seconds=time.time()-started, returncode=0))
        w.write('phase-timings.json', phases)
        cap = int(deadline - time.time() - 180)
        if not 60 <= cap <= 3600:
            raise TimeoutError('Insufficient validation time')
        run('validation', ['scripts/run_seed_confirmation.py', 'repeat', '--dataset', str(dataset),
                          '--endpoints', str(endpoints), '--output', str(w.OUT / 'repeats'),
                          '--timeout-seconds', str(cap)], cap + 15)
        if json.loads((w.OUT / 'repeats/status.json').read_text()) != dict(
                status='completed', passes=18, test_accessed=False, holdout_accessed=False):
            raise ValueError('Incomplete repeat study')
        result.update(status='accepted', profiles=['validation-repeats'], quality_accepted=False)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, phase=w.RESULT.get('phase'))
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        if w.OUT.is_dir():
            w.write('cpu-end.json', w.cpu_snapshot())
            w.write('worker.json', result)
            with tarfile.open(w.ARCHIVE, 'w:gz') as tar:
                tar.add(w.OUT, arcname='results')
            if 0 < w.ARCHIVE.stat().st_size <= 2 * 1024**3:
                result['archive'] = dict(size=w.ARCHIVE.stat().st_size, sha256=w.digest(w.ARCHIVE))
        w.RESULT = result


def main():
    w.TOKEN = os.environ.pop('WAFER_RESULT_TOKEN')
    server = ThreadingHTTPServer(('0.0.0.0', 8888), w.Handler)
    threading.Thread(target=work, daemon=True).start()
    server.serve_forever()


if __name__ == '__main__':
    main()
