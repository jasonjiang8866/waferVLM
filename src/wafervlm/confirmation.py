"""Frozen seed confirmation; validation only, no provider access or holdout entrypoint."""
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .core import LABELS, PROMPT, audit_dataset, metrics, parse_label, read_jsonl, sha256_file, write_json
from .experiment import seal_output
from .modeling import MODEL_ID, MODEL_REVISION
from .resume import check_resume, REQUIRED

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/seed-confirmation.json'


def recipe():
    return json.loads(CONFIG.read_text())


def training_config(seed, step):
    cfg = recipe()
    if type(seed) is not int or seed not in cfg['seeds'] or not 0 < step <= 251:
        raise ValueError('Unexpected seed or step')
    return dict(seed=seed, steps=step, device='cuda',
                accumulation=cfg['accumulation'], rank=cfg['rank'], max_length=cfg['max_length'],
                learning_rate=cfg['initial_learning_rate'] if step <= 100
                else cfg['continuation_learning_rate'])


def check_dataset(dataset):
    cfg = recipe()
    dataset = Path(dataset)
    if (sha256_file(dataset / 'dataset.json') != cfg['dataset_sha256']
            or sha256_file(dataset / 'validation.jsonl') != cfg['validation_sha256']):
        raise ValueError('Frozen dataset/split identity mismatch')
    rows = audit_dataset(dataset)
    if {k: len(v) for k, v in rows.items()} != dict(train=790, validation=160, test=160):
        raise ValueError('Frozen split counts mismatch')
    return rows['validation']  # test is integrity-audited, never inferred or scored


def verify_tree(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'checksums.json').read_text())
    paths = list(root.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('Symlinks are not valid result members')
    actual = {str(p.relative_to(root)): sha256_file(p) for p in paths
              if p.is_file() and p.name != 'checksums.json'}
    if not actual or actual != manifest:
        raise ValueError('Incomplete result hash coverage or changed result')
    return manifest


def inspect_checkpoint(checkpoint, dataset, seed, step):
    checkpoint = Path(checkpoint).resolve()
    root = checkpoint.parent.parent
    check_resume(checkpoint, dataset, MODEL_REVISION, training_config(seed, step + 1),
                 lower_learning_rate=(step == 100))
    run = json.loads((root / 'run.json').read_text())
    if run['config']['learning_rate'] != training_config(seed, step)['learning_rate']:
        raise ValueError('Checkpoint is outside the frozen LR schedule')
    if json.loads((checkpoint / 'trainer_state.json').read_text())['global_step'] != step:
        raise ValueError('Checkpoint endpoint does not match its declared step')
    if checkpoint.name != f'checkpoint-{step}':
        raise ValueError('Checkpoint path/step mismatch')
    if seed == 17 and step in (200, 250):
        if sha256_file(checkpoint / 'adapter_model.safetensors') != recipe()['seed17_adapter_sha256'][str(step)]:
            raise ValueError('Seed 17 must reuse the released weights')
    return {'seed': seed, 'step': step, 'checkpoint': str(checkpoint),
            'files': {name: sha256_file(checkpoint / name) for name in sorted(REQUIRED)},
            'run_sha256': sha256_file(root / 'run.json')}


def verify_training(root, dataset, seed, start, end, parent=None):
    root = Path(root)
    verify_tree(root)
    endpoint = inspect_checkpoint(root / f'checkpoints/checkpoint-{end}', dataset, seed, end)
    run = json.loads((root / 'run.json').read_text())
    if run['config'] != training_config(seed, end):
        raise ValueError('Training configuration drift')
    training = json.loads((root / 'training.json').read_text())
    for key, expected in {'start_step': start, 'end_step': end, 'executed_steps': end-start,
                          'gradient_checks': end-start, 'nonzero_gradient_checks': end-start}.items():
        if training[key] != expected:
            raise ValueError('Training update/gradient count mismatch: ' + key)
    trace = training['learning_rate_trace']
    if [r['step'] for r in trace] != list(range(start+1, end+1)):
        raise ValueError('Incomplete LR trace')
    lr = training_config(seed, end)['learning_rate']
    if any(not r['rates'] or any(not math.isfinite(x) or x != lr for x in r['rates']) for r in trace):
        raise ValueError('Actual optimizer LR differs from frozen schedule')
    if not math.isfinite(training['train_loss']) or training['peak_allocated_bytes'] <= 0:
        raise ValueError('Invalid loss or missing CUDA memory evidence')
    if training.get('frozen_parameters_verified') is not True:
        raise ValueError('Missing frozen parameter check')
    if parent is None:
        if run['resume'] is not None or training['learning_rate_intervention'] is not None:
            raise ValueError('New seed must initialize independently')
    else:
        resumed = run['resume']
        if (resumed['global_step'] != start or training['optimizer_steps_at_resume'] != [start]
                or training['scheduler_step_at_resume'] != start
                or resumed['parent_learning_rate'] != training_config(seed, start)['learning_rate']
                or resumed['lower_learning_rate'] != (start == 100)):
            raise ValueError('Invalid full-state restore or LR transition')
        if (resumed['parent_run_sha256'] != sha256_file(Path(parent).parent.parent / 'run.json')
                or resumed['checkpoint_state_sha256'] != sha256_file(Path(parent) / 'trainer_state.json')):
            raise ValueError('Broken parent metadata hash chain')
        for key, name in [('adapter', 'adapter_model.safetensors'), ('optimizer', 'optimizer.pt'),
                          ('scheduler', 'scheduler.pt'), ('rng', 'rng_state.pth')]:
            if resumed[key + '_sha256'] != sha256_file(Path(parent) / name):
                raise ValueError('Broken checkpoint state hash chain')
        intervention = training['learning_rate_intervention']
        if start == 100:
            if intervention != {'from': .0002, 'to': .00005, 'scheduler_last_epoch': 100}:
                raise ValueError('Incorrect LR intervention')
        elif intervention is not None:
            raise ValueError('Unexpected additional LR intervention')
    return endpoint


def run_process(args, deadline, log):
    """One fresh process per stage/pass; reap its whole process group on any exit."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Confirmation work deadline exhausted')
    environment = {k: v for k, v in os.environ.items()
                   if k not in {'RUNPOD_API_KEY', 'GH_TOKEN', 'GITHUB_TOKEN', 'WAFER_RESULT_TOKEN'}}
    environment['PYTHONHASHSEED'] = str(recipe()['evaluation_seed'])
    with Path(log).open('w') as stream:
        child = subprocess.Popen(args, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code = child.wait(timeout=remaining)
            if code:
                raise subprocess.CalledProcessError(code, args)
        finally:
            # Also remove descendants after an early/failed parent exit.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def train_seed(dataset, output, seed, resume_from=None, timeout_seconds=None):
    cfg = recipe()
    if type(seed) is not int or seed not in cfg['new_seeds']:
        raise ValueError('Only seeds 42/123 may be newly trained')
    cap = cfg['train_work_seconds'] if timeout_seconds is None else timeout_seconds
    if type(cap) is not int or not 1 <= cap <= cfg['train_work_seconds']:
        raise ValueError('Invalid training deadline')
    check_dataset(dataset)
    parent, start = None, 0
    if resume_from:
        parent = Path(resume_from).resolve()
        start = json.loads((parent / 'trainer_state.json').read_text())['global_step']
        if start not in (50, 100, 150, 200):
            raise ValueError('Recovery requires a saved step 50/100/150/200 checkpoint')
        inspect_checkpoint(parent, dataset, seed, start)
        verify_tree(parent.parent.parent)
        if Path(output).resolve().is_relative_to(parent.parent.parent):
            raise ValueError('Recovery output must be outside the immutable parent directory')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + cap
    write_json(root / 'workload.json', cfg)
    endpoints = []
    try:
        if parent:
            import shutil
            copied = root / 'recovery-parent'
            shutil.copytree(parent.parent.parent, copied)
            parent = copied / 'checkpoints' / parent.name
            inspect_checkpoint(parent, dataset, seed, start)
            if start == 200:
                endpoints.append(inspect_checkpoint(parent, dataset, seed, 200))
        for end in cfg['stages']:
            if end <= start:
                continue
            config = training_config(seed, end)
            out = root / f'step{end}'
            args = [sys.executable, '-m', 'wafervlm.cli', 'train', '--dataset', str(dataset),
                    '--output', str(out)]
            for key, value in config.items():
                args += ['--' + key.replace('_', '-'), str(value)]
            if parent:
                args += ['--resume-from', str(parent)]
                if start == 100:
                    args += ['--lower-learning-rate']
            try:
                run_process(args, deadline, root / f'train-{end}.log')
            finally:
                # Preserve fully written intermediate checkpoints after a timeout.
                # This seal records bytes, NOT acceptance; inspect_checkpoint still rejects incomplete states.
                if out.exists():
                    seal_output(out)
            result = verify_training(out, dataset, seed, start, end, parent)
            if end in cfg['endpoints']:
                endpoints.append(result)
            start, parent = end, out / f'checkpoints/checkpoint-{end}'
        write_json(root / 'endpoints.json', endpoints)
        write_json(root / 'status.json', {'status': 'completed', 'seed': seed, 'end_step': 250,
                                         'test_accessed': False, 'holdout_accessed': False})
    except BaseException as exc:
        write_json(root / 'status.json', {'status': 'failed', 'error_type': type(exc).__name__,
                                         'seed': seed, 'last_accepted_step': start})
        raise
    finally:
        seal_output(root)
    return endpoints


def prepare_endpoints(specification, dataset):
    """Require all six endpoints before starting even the first validation process."""
    cfg = recipe()
    expected = {(seed, step) for seed in cfg['seeds'] for step in cfg['endpoints']}
    pairs = [(r['seed'], r['step']) for r in specification]
    if len(pairs) != 6 or set(pairs) != expected:
        raise ValueError('Exactly six unique seed/checkpoint pairs are required')
    results = []
    for row in sorted(specification, key=lambda r: (r['seed'], r['step'])):
        actual = inspect_checkpoint(row['checkpoint'], dataset, row['seed'], row['step'])
        if actual['files'] != row['files'] or actual['run_sha256'] != row['run_sha256']:
            raise ValueError('Endpoint changed since the matrix was frozen')
        results.append(actual)
    return results


def summarize_repeats(records, reference, base=None):
    """Recompute every score; first pass is headline, including if a later pass is better."""
    cfg = recipe()
    expected = {(s, t, r) for s in cfg['seeds'] for t in cfg['endpoints'] for r in range(1, 4)}
    keys = [(r['seed'], r['step'], r['repeat']) for r in records]
    if len(keys) != 18 or set(keys) != expected:
        raise ValueError('All 18 uniquely identified passes are required')
    identities = [(r['id'], r['label'], r['image_sha256']) for r in reference]
    if len(identities) != 160 or len({r[0] for r in identities}) != 160:
        raise ValueError('Expected the 160 unique frozen validation samples')
    instances = set()
    by_pair = {}
    environment_keys = ('model_id', 'model_revision', 'dataset_sha256', 'source_sha', 'image_digest',
                        'packages', 'gpu', 'cuda', 'driver', 'processor_identity', 'host',
                        'base_weights_sha256', 'generation')
    environments = []
    for record in sorted(records, key=lambda r: (r['seed'], r['step'], r['repeat'])):
        root = Path(record['output']) if base is None else Path(base) / record['output']
        if base is not None and (Path(record['output']).is_absolute()
                                 or not root.resolve().is_relative_to(Path(base).resolve())):
            raise ValueError('Evaluation path escapes result package')
        verify_tree(root)
        run = json.loads((root / 'run.json').read_text())
        pred = read_jsonl(root / 'predictions.jsonl')
        if [(r['id'], r['label'], r['image_sha256']) for r in pred] != identities:
            raise ValueError('Validation order or identity changed')
        if any(r['prediction'] != parse_label(r['response'], r['truncated']) for r in pred):
            raise ValueError('Predictions do not match raw model responses')
        config = run['config']
        required_config = dict(split='validation', device='cuda', max_new_tokens=16,
                               seed=17, do_sample=False, enable_thinking=False, include_loss=True, adapter=True)
        if config != required_config or run['prompt'] != PROMPT or run['labels'] != list(LABELS):
            raise ValueError('Fixed inference settings changed')
        if any(run['generation'].get(k) != v for k, v in
               {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 16, 'use_cache': True}.items()):
            raise ValueError('Actual generation configuration changed')
        if (run['model_id'] != MODEL_ID or run['model_revision'] != MODEL_REVISION
                or run['dataset_sha256'] != cfg['dataset_sha256']
                or run['validation_sha256'] != cfg['validation_sha256']
                or run['adapter_sha256'] != record['adapter_sha256']):
            raise ValueError('Evaluation weights or dataset identity mismatch')
        instance = run['process_instance']
        if not instance or instance in instances:
            raise ValueError('Evaluation process was reused')
        instances.add(instance)
        score = metrics(pred)
        saved = json.loads((root / 'metrics.json').read_text())
        if any(saved[k] != v for k, v in score.items()):
            raise ValueError('Saved metrics do not match predictions')
        if (not math.isfinite(saved['answer_token_loss']) or saved['answer_tokens'] <= 0
                or not math.isfinite(saved['mean_generation_seconds'])
                or saved['mean_generation_seconds'] < 0 or saved['peak_allocated_bytes'] <= 0):
            raise ValueError('Missing finite loss, timing or CUDA memory evidence')
        if any(not run.get(k) for k in environment_keys):
            raise ValueError('Incomplete environment provenance')
        environments.append({k: run[k] for k in environment_keys})
        locked = ('model_id', 'model_revision', 'dataset_sha256', 'source_sha', 'image_digest',
                  'packages', 'cuda', 'processor_identity', 'base_weights_sha256', 'generation')
        if any(environments[-1][k] != environments[0][k] for k in locked):
            raise ValueError('Locked model, processor or runtime changed between passes')
        pair = (record['seed'], record['step'])
        passes = by_pair.setdefault(pair, [])
        if passes and passes[0]['adapter_sha256'] != record['adapter_sha256']:
            raise ValueError('Fixed weights changed between repeats')
        first = passes[0]['predictions'] if passes else pred
        passes.append({**record, 'score': score, 'predictions': pred,
                       'predictions_sha256': sha256_file(root / 'predictions.jsonl'),
                       'changed_prediction_ids': [r['id'] for r, old in zip(pred, first)
                                                  if r['prediction'] != old['prediction']],
                       'changed_response_ids': [r['id'] for r, old in zip(pred, first)
                                                if r['response'] != old['response']]})
    pairs = []
    for (seed, step), passes in sorted(by_pair.items()):
        values = [p['score']['macro_f1'] for p in passes]
        pairs.append({'seed': seed, 'step': step, 'headline': passes[0]['score'],
                      'macro_f1_range': max(values)-min(values),
                      'passes': [{k: v for k, v in p.items() if k != 'predictions'} for p in passes]})
    deltas = [{'seed': seed, 'role': 'exploratory' if seed == 17 else 'confirmation',
               **{key: by_pair[seed, 250][0]['score'][key]-by_pair[seed, 200][0]['score'][key]
                  for key in ('accuracy', 'macro_f1')}} for seed in cfg['seeds']]
    def dispersion(rows):
        return {key: {'mean': statistics.mean(r[key] for r in rows),
                      'sample_sd': statistics.stdev(r[key] for r in rows)} for key in ('accuracy', 'macro_f1')}
    return {'headline_rule': 'first_pass_not_best_repeat', 'pairs': pairs, 'paired_deltas': deltas,
            'all_seed_deltas': dispersion(deltas), 'confirmation_seed_deltas': dispersion(deltas[1:]),
            'same_environment': all(e == environments[0] for e in environments),
            'environments': environments, 'test_accessed': False, 'holdout_accessed': False,
            'interpretation': 'Validation repeatability and seed variation only; no generalization claim.'}


def repeat_validation(dataset, output, endpoints, timeout_seconds=None):
    cfg = recipe()
    cap = cfg['evaluation_work_seconds'] if timeout_seconds is None else timeout_seconds
    if type(cap) is not int or not 1 <= cap <= cfg['evaluation_work_seconds']:
        raise ValueError('Invalid evaluation deadline')
    reference = check_dataset(dataset)
    endpoints = prepare_endpoints(endpoints, dataset)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'endpoints.json', endpoints)
    write_json(root / 'workload.json', cfg)
    # Preserve the original manifest bytes, including their predeclared digest.
    import shutil
    shutil.copyfile(Path(dataset) / 'validation.jsonl', root / 'validation-reference.jsonl')
    records = []
    deadline = time.monotonic() + cap
    try:
        for endpoint in endpoints:
            seed, step = endpoint['seed'], endpoint['step']
            cp = Path(endpoint['checkpoint'])
            origin = root / f'weights-{seed}-{step}'
            (origin / 'adapter').mkdir(parents=True)
            for name in ('adapter_config.json', 'adapter_model.safetensors'):
                shutil.copyfile(cp / name, origin / 'adapter' / name)
            if sha256_file(origin / 'adapter/adapter_model.safetensors') != endpoint['files']['adapter_model.safetensors']:
                raise ValueError('Endpoint weights changed after matrix verification')
            shutil.copyfile(cp.parent.parent / 'run.json', origin / 'run.json')
            seal_output(origin)
            for repeat in range(1, 4):
                dest = root / f'seed{seed}-step{step}-repeat{repeat}'
                run_process([sys.executable, '-m', 'wafervlm.cli', 'evaluate', '--dataset', str(dataset),
                             '--output', str(dest), '--adapter', str(origin / 'adapter'),
                             '--split', 'validation', '--device', 'cuda', '--seed', '17',
                             '--include-loss', '--max-new-tokens', '16'], deadline,
                            root / f'{dest.name}.log')
                records.append({'seed': seed, 'step': step, 'repeat': repeat, 'output': dest.name,
                                'adapter_sha256': endpoint['files']['adapter_model.safetensors']})
                write_json(root / 'passes.json', records)
        report = summarize_repeats(records, reference, base=root)
        write_json(root / 'summary.json', report)
        write_json(root / 'status.json', {'status': 'completed', 'passes': 18,
                                         'test_accessed': False, 'holdout_accessed': False})
    except BaseException as exc:
        write_json(root / 'status.json', {'status': 'failed', 'error_type': type(exc).__name__,
                                         'completed_passes': len(records)})
        raise
    finally:
        seal_output(root)
    return report
