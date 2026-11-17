"""Run the frozen 18-pass evaluator in the existing pinned confirmation image."""
import json
import hmac
import os
import secrets
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from http.server import ThreadingHTTPServer

from runpod import vlm_worker as w
from runpod.vlm_repeat_inputs import FROZEN, archives, prepare, validated_url
from runpod.vlm_repeat_resume import enabled as resuming, transfer_pin

INPUT_LOCK = threading.Lock()
INPUT_READY = threading.Event()
INPUT_URL = None


def request_input_url(key):
    global INPUT_URL
    request = dict(key=key, nonce=secrets.token_hex(16))
    with INPUT_LOCK:
        INPUT_URL = None
        INPUT_READY.clear()
        w.RESULT['input_request'] = request
    try:
        wait = min(90, float(os.environ['WAFER_DEADLINE']) - time.time() - 60)
        if wait <= 0 or not INPUT_READY.wait(wait):
            raise TimeoutError('Fresh input URL delivery timed out')
        with INPUT_LOCK:
            return INPUT_URL
    finally:
        with INPUT_LOCK:
            INPUT_URL = None
            w.RESULT.pop('input_request', None)


class InputHandler(w.Handler):
    def do_POST(self):
        global INPUT_URL
        if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + w.TOKEN):
            self.send_error(403); return
        if self.path != '/input-url':
            self.send_error(404); return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16000:
                raise ValueError('Invalid input size')
            self.connection.settimeout(10)
            data = json.loads(self.rfile.read(length))
            if set(data) != {'key', 'nonce', 'url'}:
                raise ValueError('Unexpected input fields')
            url = validated_url(data['url'])
            with INPUT_LOCK:
                if w.RESULT.get('input_request') != {k: data[k] for k in ('key', 'nonce')}:
                    self.send_error(409); return
                if INPUT_URL is None:
                    INPUT_URL = url
                    INPUT_READY.set()
            payload = b'{"status":"accepted"}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
        except (ValueError, TypeError, KeyError, TimeoutError):
            self.send_error(400)


def work():
    child = None
    result = dict(status='rejected', profiles=[], source_sha=os.environ['WAFER_SOURCE_SHA'],
                  controller_sha=os.environ['WAFER_CONTROLLER_SHA'],
                  config_sha256=os.environ['WAFER_CONFIG_SHA'],
                  input_manifest_sha256=os.environ['WAFER_INPUT_MANIFEST_SHA'])
    try:
        w.OUT.mkdir(parents=True, exist_ok=False)
        w.write('cpu-start.json', w.cpu_snapshot())
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
        paths = archives('/workspace/repeat-archives', request_input_url, deadline - 180)
        rows = prepare(paths, '/workspace/repeat-inputs', dataset, json.loads(FROZEN.read_text()))
        partial = None
        if resuming():
            partial = archives('/workspace/repeat-archives', request_input_url, deadline - 180,
                               pins={'repeat-partial': transfer_pin()})['repeat-partial']
        endpoints = Path('/workspace/frozen-endpoints.json')
        endpoints.write_text(json.dumps(rows, indent=2))
        phases.append(dict(phase='inputs', wall_seconds=time.time()-started, returncode=0))
        w.write('phase-timings.json', phases)
        cap = int(deadline - time.time() - 180)
        if not 60 <= cap <= 3600:
            raise TimeoutError('Insufficient validation time')
        entry = (['-m', 'runpod.vlm_repeat_resume', 'resume', '--source', str(partial)] if resuming()
                 else ['scripts/run_seed_confirmation.py', 'repeat'])
        run('validation', [*entry, '--dataset', str(dataset),
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
    server = ThreadingHTTPServer(('0.0.0.0', 8888), InputHandler)
    threading.Thread(target=work, daemon=True).start()
    server.serve_forever()


if __name__ == '__main__':
    main()
