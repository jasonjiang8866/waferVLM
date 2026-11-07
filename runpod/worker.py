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
PROFILE_LIMITS = [('smoke', 8, 1800), ('baseline', 100, 7200)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False))


def progress_snapshot():
    """Bounded numeric telemetry only; never expose raw subprocess logs."""
    snapshot = dict(RESULT)
    phase = snapshot.get('phase')
    if snapshot.get('status') != 'running' or phase not in {'smoke', 'baseline'}:
        return snapshot
    total = 8 if phase == 'smoke' else 100
    try:
        with (OUT / (phase + '.log')).open('rb') as log:
            log.seek(0, 2)
            log.seek(max(0, log.tell() - 65536))
            tail = log.read(65536).decode('utf-8', errors='replace')
        # Trainer tqdm uses the optimizer-step total; eval has 160 samples.
        matches = re.findall(r'(?<![0-9])([0-9]+)/' + str(total) + r'(?=\s|\[)', tail)
        steps = [int(x) for x in matches if 0 <= int(x) <= total]
        if steps:
            snapshot['progress'] = {'optimizer_step': max(steps), 'optimizer_steps': total}
    except OSError:
        pass  # Missing telemetry must never terminate a healthy workload.
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
              'config_sha256': os.environ['WAFER_CONFIG_SHA']}
    try:
        OUT.mkdir(parents=True, exist_ok=False)  # restart must not rerun training
        if os.environ.get('WAFERVLM_SOURCE_SHA') != result['source_sha']:
            raise ValueError('image_source_mismatch')
        if digest(ROOT / 'configs/baseline.json') != result['config_sha256']:
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
        def run(phase, args, cap):
            nonlocal child
            RESULT['phase'] = phase
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
                if code:
                    raise RuntimeError('child_failed_' + phase)
        dataset = '/workspace/wafer-data'
        run('data', ['scripts/prepare_public_data.py', '--trust-pickle',
                    '--cache', '/workspace/wafer-source', '--output', dataset], 1200)
        if digest(Path(dataset) / 'dataset.json') != os.environ['WAFER_DATASET_SHA']:
            raise ValueError('dataset_protocol_mismatch')
        write('acquisition.json', json.loads((Path(dataset) / 'acquisition.json').read_text()))
        for profile, steps, seconds in PROFILE_LIMITS:
            run(profile, ['scripts/run_experiment.py', '--dataset', dataset,
                '--output', str(OUT / profile), '--profile', profile,
                '--timeout-seconds', str(seconds)], seconds + 30)
            state = json.loads((OUT / profile / 'status.json').read_text())
            training = json.loads((OUT / profile / 'train/training.json').read_text())
            if state['status'] != 'completed' or training['gradient_checks'] != steps or training['nonzero_gradient_checks'] < 1:
                raise ValueError('training_acceptance_failed')
            result['profiles'].append(profile)
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
