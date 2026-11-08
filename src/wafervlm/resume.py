"""Validate a complete Trainer checkpoint before any optimizer/RNG deserialization."""
import json
import math
from pathlib import Path
from .core import LABELS, PROMPT, sha256_file

REQUIRED = {'adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
            'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'training_args.bin'}


def check_resume(checkpoint, dataset, revision, config, *, lower_learning_rate=False):
    checkpoint = Path(checkpoint).resolve()
    root = checkpoint.parent.parent
    manifest = json.loads((root / 'checksums.json').read_text())
    run = json.loads((root / 'run.json').read_text())
    paths = [root / 'run.json', *checkpoint.iterdir()]
    if not REQUIRED <= {p.name for p in checkpoint.iterdir()}:
        raise ValueError('Incomplete resume checkpoint')
    for path in paths:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('Unsafe checkpoint member')
        rel = str(path.relative_to(root))
        if manifest.get(rel) != sha256_file(path):
            raise ValueError('Resume checksum mismatch')
    from .modeling import MODEL_ID
    if (run['model_id'] != MODEL_ID or run['model_revision'] != revision
            or run['dataset_sha256'] != sha256_file(Path(dataset) / 'dataset.json')
            or run['labels'] != list(LABELS) or run['prompt'] != PROMPT):
        raise ValueError('Resume data/model/prompt mismatch')
    for key in ('seed', 'accumulation', 'rank', 'max_length', 'device'):
        if run['config'][key] != config[key]:
            raise ValueError('Resume hyperparameter mismatch: ' + key)
    old_lr, new_lr = run['config']['learning_rate'], config['learning_rate']
    if lower_learning_rate:
        if not math.isfinite(new_lr) or not 0 < new_lr < old_lr:
            raise ValueError('Explicit LR intervention must lower the learning rate')
    elif old_lr != new_lr:
        raise ValueError('Resume hyperparameter mismatch: learning_rate')
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    step = state['global_step']
    if type(step) is not int or not 0 < step < config['steps']:
        raise ValueError('Resume step must precede requested total steps')
    return {'global_step': step, 'checkpoint': checkpoint.name,
            'parent_learning_rate': old_lr, 'lower_learning_rate': lower_learning_rate,
            'checkpoint_state_sha256': sha256_file(checkpoint / 'trainer_state.json'),
            'adapter_sha256': sha256_file(checkpoint / 'adapter_model.safetensors'),
            'parent_run_sha256': sha256_file(root / 'run.json'),
            'optimizer_sha256': sha256_file(checkpoint / 'optimizer.pt'),
            'scheduler_sha256': sha256_file(checkpoint / 'scheduler.pt'),
            'rng_sha256': sha256_file(checkpoint / 'rng_state.pth')}


def lower_constant_learning_rate(optimizer, scheduler, old_lr, new_lr):
    """Change LR after Trainer restore without resetting Adam moments or step counts."""
    from torch.optim.lr_scheduler import LambdaLR

    scheduler = getattr(scheduler, 'scheduler', scheduler)
    if (not isinstance(scheduler, LambdaLR) or not math.isfinite(new_lr)
            or not 0 < new_lr < old_lr):
        raise ValueError('Expected a lower LR and constant LambdaLR scheduler')
    groups = optimizer.param_groups
    if (len(groups) != len(scheduler.base_lrs)
            or len(groups) != len(scheduler.lr_lambdas)
            or any(not math.isclose(g['lr'], old_lr, rel_tol=1e-12) for g in groups)
            or any(not math.isclose(lr, old_lr, rel_tol=1e-12) for lr in scheduler.base_lrs)
            or any(fn(scheduler.last_epoch) != 1 or fn(scheduler.last_epoch + 1) != 1
                   for fn in scheduler.lr_lambdas)):
        raise ValueError('Restored optimizer/scheduler LR is not the expected constant LR')
    for group in groups:
        group['lr'] = group['initial_lr'] = new_lr
    scheduler.base_lrs = [new_lr] * len(groups)
    scheduler._last_lr = [new_lr] * len(groups)
    return {'from': old_lr, 'to': new_lr, 'scheduler_last_epoch': scheduler.last_epoch}
