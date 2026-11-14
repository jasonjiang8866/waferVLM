"""One GPU, six declared DINO development trials. No account credentials."""
import hashlib
import hmac
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from runpod.dinov3_bundle import FILES, UPLOAD_CHUNK_BYTES, recipe, select, trials

ROOT = Path('/app')
OUT = Path('/workspace/dino-results')
ARCHIVE = Path('/workspace/dino-results.tar.gz')
DATASET = Path('/workspace/dino-data')
TOKEN = ''
WEIGHTS = {}
STATE = {'status': 'running', 'phase': 'awaiting_weights'}
READY = threading.Event()
LOCK = threading.Lock()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def cache_root():
    cfg = recipe()
    return Path(os.environ['HF_HOME']) / 'hub' / ('models--' + cfg['model_id'].replace('/', '--')) / 'snapshots' / cfg['model_revision']


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def authorized(self):
        return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + TOKEN)

    def do_PUT(self):
        if not self.authorized():
            self.send_error(403)
            return
        name = self.path.removeprefix('/weights/')
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            size = 0
        span = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', self.headers.get('Content-Range', ''))
        if self.path != '/weights/' + name or name not in FILES or not span:
            self.send_error(400)
            return
        start, end, total = map(int, span.groups())
        if not (0 <= start <= end < total <= FILES[name] and end-start+1 == size
                and 0 < size <= UPLOAD_CHUNK_BYTES):
            self.send_error(400)
            return
        # Read at most one bounded chunk. Commit only authenticated, hash-checked
        # bytes. Replaying an acknowledged chunk is safe after a lost response.
        with LOCK:
            target = cache_root() / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.upload')
            self.connection.settimeout(45)
            try:
                block = self.rfile.read(size)
                if len(block) != size or hashlib.sha256(block).hexdigest() != self.headers.get('X-Chunk-SHA256'):
                    raise ValueError('chunk_hash_mismatch')
                existing = target if target.exists() else temporary
                offset = existing.stat().st_size if existing.exists() else 0
                if start < offset:
                    with existing.open('rb') as f:
                        f.seek(start)
                        same = end < offset and f.read(size) == block
                    if not same or (target.exists() and offset != total):
                        self.send_error(409)
                        return
                elif start == offset and not target.exists():
                    with temporary.open('ab') as f:
                        f.write(block)
                else:
                    self.send_error(409)
                    return
                if not target.exists() and temporary.stat().st_size == total:
                    if digest(temporary) != WEIGHTS['files_sha256'][name]:
                        temporary.unlink()
                        raise ValueError('weight_hash_mismatch')
                    temporary.replace(target)
                    STATE['received'] = [n for n in FILES if (cache_root() / n).is_file()]
                    if len(STATE['received']) == len(FILES):
                        READY.set()
            except Exception:
                self.send_error(400)
                return
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if not self.authorized():
            self.send_error(403)
            return
        if self.path == '/result':
            data = json.dumps(STATE, allow_nan=False).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/archive' and STATE.get('archive'):
            self.send_response(200)
            self.send_header('Content-Length', str(ARCHIVE.stat().st_size))
            self.end_headers()
            with ARCHIVE.open('rb') as f:
                shutil.copyfileobj(f, self.wfile, 1024 * 1024)
        else:
            self.send_error(404)


def work():
    child = None
    result = {'status': 'rejected', 'source_sha': os.environ['WAFERVLM_SOURCE_SHA'],
              'image': os.environ['DINO_IMAGE'], 'weights': WEIGHTS}
    try:
        OUT.mkdir(parents=True, exist_ok=False)
        if not READY.wait(timeout=min(1200, max(0, float(os.environ['DINO_DEADLINE']) - time.time() - 180))):
            raise TimeoutError('weights_delivery_timeout')
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError('CUDA_BF16_required')
        write('hardware.json', dict(gpu=torch.cuda.get_device_name(), bf16=True,
              torch=torch.__version__, cuda=torch.version.cuda))
        environment = {k: v for k, v in os.environ.items() if k not in
                       {'RUNPOD_API_KEY', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN', 'DINO_RESULT_TOKEN'}}
        environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
        timings = []
        def run(name, command, cap=3600):
            nonlocal child
            STATE['phase'] = name
            remaining = min(cap, int(float(os.environ['DINO_DEADLINE']) - time.time() - 180))
            if remaining < 60:
                raise TimeoutError('insufficient_worker_time')
            started = time.monotonic()
            with (OUT / (name + '-worker.log')).open('w') as log:
                child = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=environment,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait(timeout=remaining)
                except BaseException:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    raise
            timings.append(dict(phase=name, seconds=time.monotonic() - started, returncode=code))
            write('timings.json', timings)
            if code:
                raise RuntimeError('child_failed')
        dataset = DATASET
        run('data', ['scripts/prepare_public_data.py', '--trust-pickle', '--cache', '/workspace/dino-source',
                     '--output', str(dataset)], cap=1200)
        for name in ('dataset.json', 'validation.jsonl', 'acquisition.json'):
            shutil.copyfile(dataset / name, OUT / name)
        cfg = recipe()
        if digest(dataset / 'dataset.json') != cfg['dataset_sha256']:
            raise ValueError('dataset_identity_mismatch')
        rows = []
        for trial in trials():
            name = trial['name']
            run(name, ['scripts/run_dinov3.py', 'train', '--dataset', str(dataset),
                       '--mode', trial['mode'], '--seed', '17', '--learning-rate', str(trial['learning_rate']),
                       '--output', str(OUT / name), '--timeout-seconds', '3600'])
            score = json.loads((OUT / name / 'metrics.json').read_text())['macro_f1']
            rows.append({**trial, 'macro_f1': score})
        selected = select(rows)
        write('selection.json', selected)
        for mode, chosen in selected.items():
            run(mode + '-reload', ['scripts/run_dinov3.py', 'evaluate', '--dataset', str(dataset),
                '--trained', str(OUT / chosen['name']), '--output', str(OUT / (mode + '-reload'))])
            run(mode + '-compare', ['scripts/compare_dinov3.py', 'compare', '--dataset', str(dataset),
                '--dinov3-result', str(OUT / chosen['name']), '--output', str(OUT / (mode + '-comparison'))], cap=300)
        result['status'] = 'accepted'
    except Exception as exc:
        result.update(error_type=type(exc).__name__, phase=STATE.get('phase'))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        if OUT.is_dir():
            write('worker.json', result)
            # This directory contains adapters/results only, never the HF cache.
            with tarfile.open(ARCHIVE, 'w:gz') as tar:
                tar.add(OUT, arcname='results')
            if 0 < ARCHIVE.stat().st_size <= 2 * 1024**3:
                result['archive'] = dict(size=ARCHIVE.stat().st_size, sha256=digest(ARCHIVE))
        STATE.clear()
        STATE.update(result)


def main():
    global TOKEN, WEIGHTS
    TOKEN = os.environ.pop('DINO_RESULT_TOKEN')
    WEIGHTS = json.loads(os.environ.pop('DINO_WEIGHTS'))
    os.environ['HF_HOME'] = '/workspace/dino-hf'
    os.environ['HF_HUB_OFFLINE'] = '1'
    threading.Thread(target=work, daemon=True).start()
    ThreadingHTTPServer(('0.0.0.0', 8888), Handler).serve_forever()


if __name__ == '__main__':
    main()
