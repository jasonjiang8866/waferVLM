import json
import tempfile
import unittest
from pathlib import Path

from runpod.vlm_holdout import DIGEST, FROZEN, MODELS, analyze, bootstrap, manifest
from wafervlm.core import LABELS, PROMPT, write_json, write_jsonl
from wafervlm.modeling import MODEL_ID, MODEL_REVISION


class HoldoutTest(unittest.TestCase):
    def test_paired_bootstrap_keeps_maps_in_same_group_together(self):
        # One group with two maps; independent image resampling would widen the interval.
        rows = [dict(lot_group='same', label=LABELS[0], prediction=LABELS[0]) for _ in range(2)]
        other = [rows[0], dict(rows[1], prediction=None)]
        self.assertEqual(bootstrap(rows, rows, other)['accuracy'], [-0.5, -0.5])

    def test_analysis_requires_exact_matrix_and_recomputes_predictions(self):
        reference = manifest()
        self.assertEqual((len(reference),len({r['lot_group'] for r in reference})),(148,143))
        endpoints=json.loads(FROZEN.read_text())
        weights={f"seed{e['seed']}-step{e['step']}":e['files']['adapter_model.safetensors'] for e in endpoints}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in MODELS:
                path=root/name
                path.mkdir()
                write_json(path/'run.json',dict(endpoint=name,model=MODEL_ID,revision=MODEL_REVISION,
                    adapter_sha256=weights.get(name),prompt=PROMPT,gpu='gpu',image='image',
                    controller='sha',manifest_sha256=DIGEST,peak_allocated_bytes=1))
                rows=[dict(id=r['id'],label=r['label'],image_sha256=r['image_sha256'],
                           prediction=r['label'],response=r['label'],truncated=False,seconds=.1) for r in reference]
                write_jsonl(path/'predictions.jsonl',rows)
            result=analyze(root,'gpu','image','sha')
            self.assertEqual(result['rows'],1036)
            self.assertEqual(result['models']['original']['macro_f1'],1.)
            self.assertEqual(result['paired_step250_minus_200']['42']['percentile_95']['macro_f1'],[0.,0.])
            write_jsonl(path/'predictions.jsonl',rows[:-1])
            with self.assertRaises(ValueError):
                analyze(root,'gpu','image','sha')
