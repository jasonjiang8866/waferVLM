"""DINO study worker: image-local backbone, scoped result URLs, no account key."""
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from runpod import dinov3_worker as w
from runpod.dinov3_study import PLAN, FROZEN, plan, names, input_pin, unpack_zip, endpoints, reference_for
from runpod.provider import NoRedirect

LOCK = threading.Lock()
READY = threading.Event()
INPUT_URL = None


def validated_url(url):
    p = urllib.parse.urlsplit(url)
    if (p.scheme != 'https' or p.username or p.password or p.port not in (None, 443)
            or not (p.hostname or '').endswith(('.blob.core.windows.net', '.actions.githubusercontent.com', '.githubusercontent.com'))):
        raise ValueError('Unexpected signed artifact destination')
    return url


class InputHandler(w.Handler):
    def do_POST(self):
        global INPUT_URL
        if not self.authorized():
            self.send_error(403); return
        if self.path != '/input-url':
            self.send_error(404); return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16000:
                raise ValueError('Invalid input size')
            self.connection.settimeout(10)
            data = json.loads(self.rfile.read(size))
            if set(data) != {'key', 'nonce', 'url'}:
                raise ValueError('Invalid URL delivery fields')
            url = validated_url(data['url'])
            with LOCK:
                if w.STATE.get('input_request') != {k: data[k] for k in ('key', 'nonce')}:
                    self.send_error(409); return
                if INPUT_URL is None:
                    INPUT_URL = url
                    READY.set()
            body = b'{"status":"accepted"}'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body)
        except (ValueError, TypeError, KeyError, TimeoutError):
            self.send_error(400)


def get_input(phase):
    global INPUT_URL
    pin = input_pin(phase)
    with LOCK:
        INPUT_URL = None
        READY.clear()
        w.STATE['input_request'] = dict(key=phase, nonce=secrets.token_hex(16))
    try:
        if not READY.wait(120):
            raise TimeoutError('Scoped input URL not delivered')
        with LOCK:
            url = INPUT_URL
        path = Path('/workspace/dino-input.zip')
        count, digest = 0, hashlib.sha256()
        req = urllib.request.Request(validated_url(url), headers={'User-Agent': 'waferVLM/0.1'})
        with urllib.request.build_opener(NoRedirect).open(req, timeout=60) as response, path.open('xb') as target:
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > 32 * 1024**2 or time.time() > float(os.environ['DINO_DEADLINE']) - 180:
                    raise TimeoutError('Input archive size/deadline exceeded')
                digest.update(block); target.write(block)
        if digest.hexdigest() != pin['zip_sha256']:
            raise ValueError('Input archive digest mismatch')
        return unpack_zip(path, Path('/workspace/dino-input'), pin)
    finally:
        with LOCK:
            INPUT_URL = None
            w.STATE.pop('input_request', None)


def work():
    child = None
    phase = os.environ['DINO_PHASE']
    result = dict(status='rejected', phase=phase, controller_sha=os.environ['DINO_CONTROLLER_SHA'],
                  image_source_sha=os.environ['WAFERVLM_SOURCE_SHA'], image=os.environ['DINO_IMAGE'],
                  plan_sha256=w.digest(PLAN), overlay_sha256=os.environ['DINO_OVERLAY_SHA256'])
    try:
        w.OUT.mkdir(parents=True, exist_ok=False)
        p = plan()
        if (result['image_source_sha'] != p['image_source_sha'] or result['image'] != p['image']
                or result['plan_sha256'] != os.environ['DINO_PLAN_SHA256']):
            raise ValueError('Image/plan identity mismatch')
        if phase == 'holdout' and w.digest(FROZEN) != os.environ['DINO_FROZEN_SHA256']:
            raise ValueError('Holdout freeze differs from reviewed commit')
        w.write('weight-delivery.json', w.embedded_weights())
        w.write('hardware.json', w.hardware_smoke())
        environment = {k: v for k, v in os.environ.items() if k not in
                       {'RUNPOD_API_KEY', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN', 'DINO_RESULT_TOKEN'}}
        timings = []

        def run(label, command, cap=1200):
            nonlocal child
            w.STATE['phase'] = label
            remaining = min(cap, float(os.environ['DINO_DEADLINE']) - time.time() - 120)
            if remaining < 60:
                raise TimeoutError('Insufficient study time')
            started = time.monotonic()
            with (w.OUT / (label + '.log')).open('w') as log:
                child = subprocess.Popen([sys.executable, *command], cwd=w.ROOT, env=environment,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait(timeout=remaining)
                except BaseException:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(); raise
            timings.append(dict(phase=label, elapsed_seconds=time.monotonic() - started, returncode=code))
            w.write('timings.json', timings)
            if code:
                raise RuntimeError('DINO subprocess failed')

        w.STATE['phase'] = 'inputs'
        source = get_input(phase)
        models = w.OUT / 'models'; models.mkdir()
        if phase == 'validation':
            run('data', ['scripts/prepare_public_data.py', '--trust-pickle', '--cache', '/workspace/dino-source', '--output', str(w.DATASET)])
            for name in ('dataset.json', 'validation.jsonl', 'acquisition.json'):
                shutil.copyfile(w.DATASET / name, w.OUT / name)
            for mode in p['modes']:
                shutil.copytree(source / (mode + '-1'), models / ('seed17-' + mode))
            # Selected seed-17 exports are verified before any new training.
            from runpod.dinov3_study import verify_training
            for mode in p['modes']:
                verify_training(models / ('seed17-' + mode), reference_for('validation'), 'seed17-' + mode)
            for seed in (42, 123):
                for mode in p['modes']:
                    name = f'seed{seed}-{mode}'
                    run(name, ['scripts/run_dinov3.py', 'train', '--dataset', str(w.DATASET), '--mode', mode,
                        '--seed', str(seed), '--learning-rate', str(p['learning_rates'][mode]),
                        '--output', str(models / name), '--timeout-seconds', '3600'])
        elif phase == 'holdout':
            for name in names():
                shutil.copytree(source / 'models' / name, models / name)
            run('data', ['-m', 'runpod.vlm_holdout', 'render', '/workspace/dino-source', str(w.DATASET)])
        else:
            raise ValueError('Unknown phase')
        eps = endpoints(w.OUT, reference_for('validation'))
        if phase == 'holdout' and eps != json.loads(FROZEN.read_text())['endpoints']:
            raise ValueError('Holdout weights differ from frozen endpoints')
        w.write('endpoints.json', eps)
        for name in names():
            for repeat in range(1, (3 if phase == 'validation' else 1) + 1):
                run(f'{name}-repeat{repeat}', ['-m', 'runpod.dinov3_study', 'one', phase, name,
                    str(w.DATASET), str(models / name), str(w.OUT / phase / name / f'repeat{repeat}')], cap=300)
        result.update(status='completed', validation_passes=18 if phase == 'validation' else 0,
                      holdout_predictions=888 if phase == 'holdout' else 0)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, failed_phase=w.STATE.get('phase'))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL); child.wait()
        if w.OUT.is_dir():
            w.write('worker.json', result)
            with tarfile.open(w.ARCHIVE, 'w:gz') as tar:
                tar.add(w.OUT, arcname='results')
            if 0 < w.ARCHIVE.stat().st_size <= 24 * 1024**2:
                result['archive'] = dict(size=w.ARCHIVE.stat().st_size, sha256=w.digest(w.ARCHIVE))
        w.STATE.clear(); w.STATE.update(result)


def main():
    w.TOKEN = os.environ.pop('DINO_RESULT_TOKEN')
    w.WEIGHTS = json.loads(os.environ.pop('DINO_WEIGHTS'))
    server = ThreadingHTTPServer(('0.0.0.0', 8888), InputHandler)
    threading.Thread(target=work, daemon=True).start()
    server.serve_forever()


if __name__ == '__main__':
    main()
