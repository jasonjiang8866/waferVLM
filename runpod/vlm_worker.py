"""Injected into the pinned image. No RunPod/GitHub account credentials."""
import hashlib
import hmac
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path('/app')
OUT = Path('/workspace/wafer-results')
ARCHIVE = Path('/workspace/wafer-results.tar.gz')
RESULT = {'status': 'running', 'phase': 'starting'}
TOKEN = ''


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False))


def progress_snapshot():
    """Read only bounded progress, never serve subprocess logs or credentials."""
    snapshot = dict(RESULT)
    if snapshot.get('status') != 'running' or snapshot.get('phase') != 'training':
        return snapshot
    for total in (250, 200, 150, 100):
        path = OUT / 'training' / f'train-{total}.log'
        try:
            with path.open('rb') as log:
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 65536))
                tail = log.read(65536).decode('utf-8', errors='replace')
            matches = re.findall(r'(?<![0-9])([0-9]+)/' + str(total) + r'(?=\s|\[)', tail)
            steps = [int(x) for x in matches if 0 <= int(x) <= total]
            if steps:
                snapshot['progress'] = {'optimizer_step': max(steps), 'optimizer_steps': total}
            break
        except OSError:
            pass
    return snapshot


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + TOKEN):
            self.send_error(403)
            return
        if self.path == '/result':
            data = json.dumps(progress_snapshot(), allow_nan=False).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/archive' and RESULT.get('archive'):
            self.send_response(200)
            self.send_header('Content-Length', str(ARCHIVE.stat().st_size))
            self.end_headers()
            with ARCHIVE.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    self.wfile.write(block)
        else:
            self.send_error(404)


def work():
    global RESULT
    child = None
    result = {'status': 'rejected', 'profiles': [],
              'source_sha': os.environ['WAFER_SOURCE_SHA'],
              'controller_sha': os.environ['WAFER_CONTROLLER_SHA'],
              'config_sha256': os.environ['WAFER_CONFIG_SHA'], 'seed': None}
    try:
        OUT.mkdir(parents=True, exist_ok=False)  # restart must not rerun training
        seed = os.environ.get('WAFER_SEED', '')
        if seed not in {'42', '123'}:
            raise ValueError('unsupported_training_seed')
        result['seed'] = int(seed)
        if os.environ.get('WAFERVLM_SOURCE_SHA') != result['source_sha']:
            raise ValueError('image_source_mismatch')
        if digest(ROOT / 'configs/seed-confirmation.json') != result['config_sha256']:
            raise ValueError('configuration_mismatch')
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError('CUDA_BF16_required')
        write('hardware.json', {'gpu': torch.cuda.get_device_name(),
              'total_memory': torch.cuda.get_device_properties(0).total_memory,
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'image_digest': os.environ['WAFERVLM_IMAGE_DIGEST']})
        environment = {k: v for k, v in os.environ.items()
                       if k not in {'WAFER_RESULT_TOKEN', 'RUNPOD_API_KEY', 'GH_TOKEN', 'GITHUB_TOKEN'}}
        environment['PYTHONUNBUFFERED'] = '1'
        phases = []
        def run(phase, args, cap):
            nonlocal child
            RESULT['phase'] = phase
            phase_started = time.time()
            remaining = min(cap, float(os.environ['WAFER_DEADLINE']) - time.time() - 120)
            if remaining < 60:
                raise TimeoutError('insufficient_worker_time')
            with (OUT / (phase + '.log')).open('w') as log:
                child = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise
                phases.append({'phase': phase, 'wall_seconds': time.time() - phase_started, 'returncode': code})
                write('phase-timings.json', phases)
                if code:
                    raise RuntimeError('child_failed_' + phase)
        dataset = '/workspace/wafer-data'
        run('data', ['scripts/prepare_public_data.py', '--trust-pickle',
                    '--cache', '/workspace/wafer-source', '--output', dataset], 1200)
        if digest(Path(dataset) / 'dataset.json') != os.environ['WAFER_DATASET_SHA']:
            raise ValueError('dataset_protocol_mismatch')
        write('acquisition.json', json.loads((Path(dataset) / 'acquisition.json').read_text()))
        cap = int(float(os.environ['WAFER_DEADLINE']) - time.time() - 180)
        if not 60 <= cap <= 11100:
            raise TimeoutError('insufficient_training_time')
        run('training', ['scripts/run_seed_confirmation.py', 'train', '--seed', seed,
            '--dataset', dataset, '--output', str(OUT / 'training'), '--timeout-seconds', str(cap)], cap + 15)
        state = json.loads((OUT / 'training/status.json').read_text())
        if state != dict(status='completed', seed=int(seed), end_step=250,
                         test_accessed=False, holdout_accessed=False):
            raise ValueError('confirmation_acceptance_failed')
        result['profiles'] = ['seed-confirmation']
        # Accepted means execution/integrity, not a claim of improved quality.
        result.update(status='accepted', quality_accepted=False)
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        result['phase'] = RESULT.get('phase')
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        if OUT.is_dir():
            write('worker.json', result)
            # Keep checkpoints and their existing checksums for exact independent verification.
            with tarfile.open(ARCHIVE, 'w:gz') as tar:
                tar.add(OUT, arcname='results')
            if 0 < ARCHIVE.stat().st_size <= 2 * 1024**3:
                result['archive'] = {'size': ARCHIVE.stat().st_size, 'sha256': digest(ARCHIVE)}
        RESULT = result


def main():
    global TOKEN
    TOKEN = os.environ.pop('WAFER_RESULT_TOKEN')
    server = ThreadingHTTPServer(('0.0.0.0', 8888), Handler)
    threading.Thread(target=work, daemon=True).start()
    server.serve_forever()  # controller and independent cleanup remove billable compute


if __name__ == '__main__':
    main()
