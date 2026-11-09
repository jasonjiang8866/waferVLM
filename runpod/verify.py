"""Verify untrusted archive paths, hashes, provenance, numerical and completion gates."""
import hashlib
import json
import math
import tarfile
from pathlib import Path, PurePosixPath

from runpod.provider import IMAGE, SOURCE


def verify_archive(path, controller_sha, continuation=False, profile=False):
    with tarfile.open(path, 'r:gz') as tar:
        files = {}
        total = 0
        for member in tar:
            p = PurePosixPath(member.name)
            if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'results':
                raise ValueError('Unsafe archive path')
            if member.isdir():
                continue
            if not member.isfile() or member.name in files:
                raise ValueError('Unsafe archive member')
            total += member.size
            if total > 4 * 1024**3 or len(files) > 10000:
                raise ValueError('Archive exceeds limits')
            h = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            files[member.name] = (member, h.hexdigest())

        def read(name):
            member = files['results/' + name][0]
            if member.size > 8_000_000:
                raise ValueError('JSON exceeds limit')
            return json.load(tar.extractfile(member))

        for name in files:
            if name.endswith('/checksums.json'):
                rel = name[len('results/'):]
                checks = read(rel)
                if not checks:
                    raise ValueError('Empty checksum manifest')
                for child, digest in checks.items():
                    p = PurePosixPath(child)
                    if p.is_absolute() or '..' in p.parts:
                        raise ValueError('Unsafe checksum path')
                    if files[str(PurePosixPath(name).parent / p)][1] != digest:
                        raise ValueError('Checksum mismatch')
        if profile:
            _verify_profile(read, files, controller_sha)
            return {'status': 'accepted', 'file_count': len(files), 'unpacked_bytes': total}
        if continuation:
            _verify_continuation(read, files, controller_sha)
            return {'status': 'accepted', 'file_count': len(files), 'unpacked_bytes': total}
        worker = read('worker.json')
        if (worker['status'] != 'accepted' or worker['source_sha'] != SOURCE
                or worker['controller_sha'] != controller_sha or worker['profiles'] != ['smoke', 'baseline']):
            raise ValueError('Worker provenance mismatch')
        local = Path(__file__).resolve().parents[1]
        expected = json.loads((local / 'docs/evidence/wm811k-lot17.json').read_text())
        if read('acquisition.json') != expected:
            raise ValueError('Dataset acquisition mismatch')
        if worker['config_sha256'] != hashlib.sha256((local / 'configs/baseline.json').read_bytes()).hexdigest():
            raise ValueError('Worker config mismatch')
        for profile, steps, split in [('smoke', 8, 'validation'), ('baseline', 100, 'test')]:
            for sub in ('', '/base', '/train', '/tuned', '/comparison'):
                if not read(profile + sub + '/checksums.json'):
                    raise ValueError('Required checksum manifest missing')
            if read(profile + '/status.json')['status'] != 'completed':
                raise ValueError('Incomplete profile')
            tr = read(profile + '/train/training.json')
            if (tr['gradient_checks'] != steps or tr['nonzero_gradient_checks'] < 1
                    or not math.isfinite(tr['train_loss']) or tr['peak_allocated_bytes'] <= 0):
                raise ValueError('Training or CUDA gate failed')
            for part in ('base', 'train', 'tuned'):
                r = read(profile + '/' + part + '/run.json')
                if (r['source_sha'] != SOURCE or r['image_digest'] != IMAGE.split('@')[1]
                        or r['dataset_sha256'] != expected['dataset_sha256'] or not r['gpu']
                        or r['model_revision'] != '2fc06364715b967f1860aea9cf38778875588b17'
                        or r['config']['device'] != 'cuda'):
                    raise ValueError('Model provenance mismatch')
                if part != 'train' and r['config']['split'] != split:
                    raise ValueError('Evaluation split mismatch')
            comparison = read(profile + '/comparison/comparison.json')
            for part in ('baseline', 'tuned'):
                if comparison[part]['n'] != 160:
                    raise ValueError('Incomplete evaluation')
                for metric in ('macro_f1', 'accuracy'):
                    if not 0 <= comparison[part][metric] <= 1:
                        raise ValueError('Invalid classification metric')
            if not any(n.startswith('results/' + profile + '/train/adapter/') and n.endswith('.safetensors') for n in files):
                raise ValueError('Missing adapter weights')
        return {'status': 'accepted', 'file_count': len(files), 'unpacked_bytes': total}


def _verify_profile(read, files, controller_sha):
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'configs/profile.json').read_text())
    expected = json.loads((root/'docs/evidence/wm811k-lot17.json').read_text())
    worker = read('worker.json')
    if (worker['status'] != 'accepted' or worker['profiles'] != ['profile']
            or worker['source_sha'] != SOURCE or worker['controller_sha'] != controller_sha
            or worker['config_sha256'] != hashlib.sha256((root/'configs/profile.json').read_bytes()).hexdigest()
            or read('acquisition.json') != expected or read('profile/workload.json') != cfg):
        raise ValueError('Profile provenance mismatch')
    origin = read('profile/resume-origin.json')
    state = read('profile/status.json')
    if (origin['run_id'] != cfg['parent_run_id'] or origin['global_step'] != 200
            or origin['adapter_sha256'] != cfg['parent_adapter_sha256']
            or state['status'] != 'completed' or state['diagnostic_only'] is not True
            or state['test_accessed'] is not False or not 0 < state['elapsed_seconds'] <= 1230):
        raise ValueError('Profile parent/completion mismatch')
    for arm in cfg['arms']:
        tr = read(f'profile/{arm}/training.json')
        run = read(f'profile/{arm}/run.json')
        if (tr['start_step'] != 200 or tr['end_step'] != 204 or tr['executed_steps'] != 4
                or tr['gradient_checks'] != 4 or tr['nonzero_gradient_checks'] != 4
                or tr['optimizer_steps_at_resume'] != [200] or not math.isfinite(tr['train_loss'])
                or tr['peak_allocated_bytes'] <= 0 or run['source_sha'] != SOURCE
                or run['image_digest'] != IMAGE.split('@')[1]
                or run['dataset_sha256'] != expected['dataset_sha256']
                or run['model_revision'] != '2fc06364715b967f1860aea9cf38778875588b17'
                or run['config']['device'] != 'cuda'
                or run['resume']['adapter_sha256'] != cfg['parent_adapter_sha256']
                or run['resume']['optimizer_sha256'] != cfg['parent_optimizer_sha256']):
            raise ValueError('Profile training/restore mismatch')
        if any(run['config'].get(k) != cfg[k] for k in ('seed','accumulation','rank','learning_rate','max_length','full_determinism')):
            raise ValueError('Profile hyperparameter mismatch')
        trace = tr['learning_rate_trace']
        if ([r['step'] for r in trace] != [201,202,203,204] or
                any(not r['rates'] or any(lr != cfg['learning_rate'] for lr in r['rates']) for r in trace)):
            raise ValueError('Profile LR mismatch')
        if any(n.startswith(f'results/profile/{arm}/checkpoints/') for n in files):
            raise ValueError('Diagnostic weights must not be promoted as checkpoints')
    compare = read('profile/comparison.json')
    if (compare['adapter_allclose'] is not True or compare.get('full_determinism') is not True
            or not math.isfinite(compare['max_abs_adapter_difference'])
            or any(not math.isfinite(v) or v <= 0 for v in compare['arm_wall_seconds'].values())):
        raise ValueError('Invalid profile comparison')
    timing = read('profile/profiled/timings.json')
    required = {'image_verify_decode','image_preprocess','tokenize','host_to_device',
                'forward_and_loss','backward','optimizer_step','gradient_check','checkpoint_save','adapter_export'}
    if (timing['device'] != 'cuda' or not timing['synchronized'] or
            not required <= {x['phase'] for x in timing['events']} or
            any(not math.isfinite(x['wall_seconds']) or x['wall_seconds'] < 0 for x in timing['events'])):
        raise ValueError('Incomplete phase timings')
    execution = read('profile/profiled/execution.json')
    if (execution.get('deterministic_algorithms') is not True or not execution['parameter_devices'] or
            any(not d.startswith('cuda:') for d in execution['parameter_devices']) or
            read('profile/profiled/operators.json')['cuda_event_count'] <= 0):
        raise ValueError('Missing actual CUDA execution evidence')
    probe = read('profile/evaluation-probe.json')
    if probe['split'] != 'validation' or probe['n'] != 8 or len(probe['samples']) != 8:
        raise ValueError('Invalid validation timing probe')
    analysis = read('validation-analysis/metrics.json')
    if analysis['split'] != 'validation' or any(m['n'] != 160 for m in analysis['checkpoints'].values()):
        raise ValueError('Incomplete released validation analysis')


def _verify_continuation(read, files, controller_sha):
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root/'docs/evidence/wm811k-lot17.json').read_text())
    cfg = json.loads((root/'configs/continuation.json').read_text())
    worker = read('worker.json')
    if (worker['status'] != 'accepted' or worker['profiles'] != ['continuation']
            or worker['source_sha'] != SOURCE or worker['controller_sha'] != controller_sha
            or worker['config_sha256'] != hashlib.sha256((root/'configs/continuation.json').read_bytes()).hexdigest()
            or read('acquisition.json') != expected):
        raise ValueError('Continuation provenance mismatch')
    base = 'continuation/'
    for part in ('', 'step100/', 'step150/', 'step200/', 'validation-100/', 'validation-150/', 'validation-200/', 'selected-test/'):
        if not read(base+part+'checksums.json'):
            raise ValueError('Missing continuation checksums')
    origin = read(base+'resume-origin.json')
    if (origin['run_id'] != cfg['parent_run_id'] or origin['artifact_id'] != cfg['parent_artifact_id']
            or origin['archive_sha256'] != cfg['parent_archive_sha256'] or origin['global_step'] != 100):
        raise ValueError('Wrong parent checkpoint')
    status = read(base+'status.json')
    if status['status'] != 'completed' or (status['start_step'],status['end_step'],status['additional_steps']) != (100,200,100):
        raise ValueError('Incomplete continuation')
    for step in (150,200):
        tr = read(base+f'step{step}/training.json')
        run = read(base+f'step{step}/run.json')
        resume = run['resume']
        if (tr['start_step'] != step-50 or tr['end_step'] != step or tr['gradient_checks'] != 50
                or tr['executed_steps'] != 50 or tr['nonzero_gradient_checks'] < 1
                or not math.isfinite(tr['train_loss']) or tr['peak_allocated_bytes'] <= 0
                or resume['global_step'] != step-50 or run['source_sha'] != SOURCE
                or run['image_digest'] != IMAGE.split('@')[1]
                or run['dataset_sha256'] != expected['dataset_sha256']
                or run['config']['device'] != 'cuda'):
            raise ValueError('Resume training gate failed')
        if any(run['config'].get(k) != cfg[k] for k in
               ('seed', 'accumulation', 'rank', 'learning_rate', 'max_length')):
            raise ValueError('Continuation hyperparameter mismatch')
        trace = tr.get('learning_rate_trace', [])
        if (len(trace) != 50 or [row['step'] for row in trace] != list(range(step-49, step+1))
                or any(not row['rates'] or any(not math.isfinite(lr) or not math.isclose(
                    lr, cfg['learning_rate'], rel_tol=1e-12) for lr in row['rates']) for row in trace)
                or tr.get('optimizer_steps_at_resume') != [step-50]):
            raise ValueError('Effective learning rate or restored Adam state mismatch')
        intervention = tr.get('learning_rate_intervention')
        if step == 150:
            if (resume.get('lower_learning_rate') is not True
                    or resume.get('parent_learning_rate') != .0002
                    or intervention != {'from': .0002, 'to': cfg['learning_rate'], 'scheduler_last_epoch': 100}):
                raise ValueError('Missing initial learning rate intervention')
        elif (resume.get('lower_learning_rate') is not False
                or resume.get('parent_learning_rate') != cfg['learning_rate'] or intervention is not None):
            raise ValueError('Second segment must preserve the lowered learning rate')
        parent = base+'step100/adapter/adapter_model.safetensors' if step==150 else base+'step150/checkpoints/checkpoint-150/adapter_model.safetensors'
        if resume['adapter_sha256'] != files['results/'+parent][1]:
            raise ValueError('Resume adapter chain mismatch')
        if step==150 and resume['optimizer_sha256'] != 'd1168af8d924c1e5654ad7854dd9d2344763aa68c119f8eae4716a8aa32e7fbe':
            raise ValueError('Original optimizer mismatch')
        if step==200:
            for key,name in [('optimizer_sha256','optimizer.pt'),('scheduler_sha256','scheduler.pt'),('rng_sha256','rng_state.pth')]:
                if resume[key] != files['results/'+base+'step150/checkpoints/checkpoint-150/'+name][1]:
                    raise ValueError('Resume optimizer/RNG chain mismatch')
    scores=[]
    for step in (100,150,200):
        m=read(base+f'validation-{step}/metrics.json')
        run=read(base+f'validation-{step}/run.json')
        if (m['n'] != 160 or not math.isfinite(m['answer_token_loss']) or m['answer_token_loss'] < 0
                or m['answer_tokens'] <= 0 or not 0 <= m['macro_f1'] <= 1
                or run['config']['split'] != 'validation' or run['config']['device'] != 'cuda'
                or run['dataset_sha256'] != expected['dataset_sha256']
                or run['source_sha'] != SOURCE or run['image_digest'] != IMAGE.split('@')[1]):
            raise ValueError('Validation gate failed')
        scores.append({'step':step,**m})
    selection=read(base+'selection.json')
    best=max(scores,key=lambda x:(x['macro_f1'],-x['step']))['step']
    if (selection['validation'] != scores or selection['best_step'] != best
            or selection['test_used_for_selection'] is not False or status['best_step'] != best
            or selection['criterion'] != cfg['selection']):
        raise ValueError('Selection gate failed')
    test=read(base+'selected-test/metrics.json')
    run=read(base+'selected-test/run.json')
    if (test['n'] != 160 or not 0 <= test['macro_f1'] <= 1 or run['config']['split'] != 'test'
            or run['adapter_checksums_sha256'] != files['results/'+base+f'step{best}/checksums.json'][1]
            or run['source_sha'] != SOURCE or run['image_digest'] != IMAGE.split('@')[1]
            or run['dataset_sha256'] != expected['dataset_sha256'] or run['config']['device'] != 'cuda'):
        raise ValueError('Selected test gate failed')
