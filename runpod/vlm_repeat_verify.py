"""Recompute the complete validation study from retrieved, hash-bound evidence."""
import json
import tempfile
from pathlib import Path

from runpod.confirmation_verify import unpack
from runpod.vlm_repeat_inputs import FROZEN, portable
from wafervlm.confirmation import recipe, summarize_repeats, verify_tree
from wafervlm.core import read_jsonl, sha256_file

ACQUISITION = Path(__file__).resolve().parents[1] / 'docs/evidence/wm811k-lot17.json'

def verify_repeat_archive(archive, spec):
    cfg = recipe()
    with tempfile.TemporaryDirectory() as tmp:
        root = unpack(archive, Path(tmp) / 'unpacked')
        if any(p.name in {'test', 'holdout', 'selected-test', 'training'} for p in root.rglob('*')):
            raise ValueError('Unexpected training/test/holdout output')
        worker = json.loads((root / 'worker.json').read_text())
        expected = dict(status='accepted', profiles=['validation-repeats'], quality_accepted=False,
                        source_sha=spec['source_sha'], controller_sha=spec['controller_sha'],
                        config_sha256=spec['config_sha256'], input_manifest_sha256=sha256_file(FROZEN))
        if spec['input_manifest_sha256'] != expected['input_manifest_sha256']:
            raise ValueError('Launch input manifest changed')
        if any(worker.get(k) != v for k, v in expected.items()):
            raise ValueError('Worker repeat provenance mismatch')
        hardware = json.loads((root / 'hardware.json').read_text())
        if hardware.get('gpu') != spec['gpu'] or hardware.get('image_digest') != spec['image'].split('@')[1]:
            raise ValueError('Repeat hardware/image mismatch')
        if json.loads((root / 'acquisition.json').read_text()) != json.loads(ACQUISITION.read_text()):
            raise ValueError('Acquisition evidence mismatch')
        repeats = root / 'repeats'
        verify_tree(repeats)
        if json.loads((repeats / 'workload.json').read_text()) != cfg:
            raise ValueError('Repeat recipe changed')
        if json.loads((repeats / 'status.json').read_text()) != dict(
                status='completed', passes=18, test_accessed=False, holdout_accessed=False):
            raise ValueError('Incomplete repeat study')
        frozen = json.loads(FROZEN.read_text())
        if portable(json.loads((repeats / 'endpoints.json').read_text())) != frozen:
            raise ValueError('Endpoint matrix changed')
        if sha256_file(repeats / 'validation-reference.jsonl') != cfg['validation_sha256']:
            raise ValueError('Validation reference changed')
        weights = {(e['seed'], e['step']): e['files']['adapter_model.safetensors'] for e in frozen}
        records = json.loads((repeats / 'passes.json').read_text())
        for r in records:
            if r.get('adapter_sha256') != weights.get((r.get('seed'), r.get('step'))):
                raise ValueError('Pass does not use frozen weights')
        report = summarize_repeats(records, read_jsonl(repeats / 'validation-reference.jsonl'), base=repeats)
        for env in report['environments']:
            if (env['source_sha'] != spec['source_sha'] or env['image_digest'] != spec['image'].split('@')[1]
                    or env['gpu'] != hardware['gpu']):
                raise ValueError('Evaluation process provenance mismatch')
        if report != json.loads((repeats / 'summary.json').read_text()):
            raise ValueError('Saved summary differs from recomputed predictions')
        return dict(status='accepted', passes=18, input_manifest_sha256=sha256_file(FROZEN),
                    archive_sha256=sha256_file(archive), summary=report,
                    quality_accepted=False, holdout_accessed=False)
