"""Validate a complete Trainer checkpoint before any optimizer/RNG deserialization."""
import json
from pathlib import Path
from .core import LABELS, PROMPT, sha256_file

REQUIRED = {'adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
            'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'training_args.bin'}


def check_resume(checkpoint, dataset, revision, config):
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
    for key in ('seed', 'accumulation', 'rank', 'learning_rate', 'max_length', 'device'):
        if run['config'][key] != config[key]:
            raise ValueError('Resume hyperparameter mismatch: ' + key)
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    step = state['global_step']
    if type(step) is not int or not 0 < step < config['steps']:
        raise ValueError('Resume step must precede requested total steps')
    return {'global_step': step, 'checkpoint': checkpoint.name,
            'checkpoint_state_sha256': sha256_file(checkpoint / 'trainer_state.json'),
            'adapter_sha256': sha256_file(checkpoint / 'adapter_model.safetensors'),
            'parent_run_sha256': sha256_file(root / 'run.json'),
            'optimizer_sha256': sha256_file(checkpoint / 'optimizer.pt'),
            'scheduler_sha256': sha256_file(checkpoint / 'scheduler.pt'),
            'rng_sha256': sha256_file(checkpoint / 'rng_state.pth')}
