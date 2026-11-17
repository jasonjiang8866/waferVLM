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
        reused_outputs = set()
        if 'resume_source_run_id' in spec:
            from runpod.vlm_repeat_resume import PIN, RECEIPT, GPU as original_gpu, FINISH_RECEIPT, finish_pin
            source = PIN if spec['resume_source_run_id'] == PIN['run_id'] else finish_pin()
            receipt_path = RECEIPT if source == PIN else FINISH_RECEIPT
            original_outputs = {r['output'] for r in json.loads(RECEIPT.read_text())['reused']}
            receipt = json.loads(receipt_path.read_text())
            if (spec['resume_source_run_id'] != source['run_id']
                    or json.loads((repeats / 'resume.json').read_text()) != receipt):
                raise ValueError('Continuation source receipt mismatch')
            for old in receipt['reused']:
                reused_outputs.add(old['output'])
                folder = repeats / old['output']
                if (sha256_file(folder / 'checksums.json') != old['checksums_sha256']
                        or sha256_file(folder / 'predictions.jsonl') != old['predictions_sha256']):
                    raise ValueError('Previously completed pass was altered')
        elif (repeats / 'resume.json').exists():
            raise ValueError('Unexpected continuation output')
        report = summarize_repeats(records, read_jsonl(repeats / 'validation-reference.jsonl'), base=repeats)
        ordered = sorted(records, key=lambda r: (r['seed'], r['step'], r['repeat']))
        for record, env in zip(ordered, report['environments']):
            if record['output'] in reused_outputs:
                expected_gpu = original_gpu if record['output'] in original_outputs else source['gpu']
            else:
                expected_gpu = hardware['gpu']
            if (env['source_sha'] != spec['source_sha'] or env['image_digest'] != spec['image'].split('@')[1]
                    or env['gpu'] != expected_gpu):
                raise ValueError('Evaluation process provenance mismatch')
        if report != json.loads((repeats / 'summary.json').read_text()):
            raise ValueError('Saved summary differs from recomputed predictions')
        return dict(status='accepted', passes=18, input_manifest_sha256=sha256_file(FROZEN),
                    archive_sha256=sha256_file(archive), summary=report,
                    quality_accepted=False, holdout_accessed=False)
