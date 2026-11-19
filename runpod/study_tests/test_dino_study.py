"""Offline checks for identity, budget, one-create and scoped-input boundaries."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from runpod import dinov3_study as study
from runpod import dinov3_study_controller as controller
from runpod.dinov3_study_ledger import check_history
from runpod.dinov3_study_worker import validated_url
from wafervlm.core import read_jsonl


class StudyTests(unittest.TestCase):
    def test_frozen_grid_and_probability_identity(self):
        self.assertEqual(len(study.names()), 6)
        reference = study.reference_for('validation')
        rows = read_jsonl(study.ROOT / 'docs/showcase/dinov3-seed17/predictions.jsonl')
        self.assertEqual(study.verify_predictions(rows, reference)['accuracy'], 139/160)
        bad = copy.deepcopy(rows); bad[0]['probabilities'] = [.125] * 8
        with self.assertRaises(ValueError): study.verify_predictions(bad, reference)
        for bad in (rows[:-1], rows[:-1]+[rows[0]], list(reversed(rows))):
            with self.assertRaises(ValueError): study.verify_predictions(bad, reference)
        bad = copy.deepcopy(rows); bad[0]['image_sha256'] = '0'*64
        with self.assertRaises(ValueError): study.verify_predictions(bad, reference)

    def test_holdout_fails_without_committed_six_weights(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(study, 'FROZEN', Path(tmp)/'missing.json'):
            with self.assertRaises(FileNotFoundError): study.input_pin('holdout')

    def test_cumulative_caps_price_and_reserve(self):
        ledger = dict(usd=.6481217168215011, seconds=1765.993593454361)
        q = controller.quote(1.39, 'NVIDIA GeForce RTX 5090', ledger)
        self.assertLess(q['maximum_estimated_usd'], 1.29)
        for gpu, rate, prior in [('NVIDIA GeForce RTX 4090', .90, ledger),
                                 ('NVIDIA GeForce RTX 5090', 1.39, dict(usd=8,seconds=100)),
                                 ('NVIDIA GeForce RTX 5090', 1.39, dict(usd=1,seconds=28000))]:
            with self.assertRaises(ValueError): controller.quote(rate,gpu,prior)

    def test_unknown_run_and_repeated_history_rejected(self):
        check_history(dict(total_count=1,workflow_runs=[dict(id=1)]), {'1'})
        for rows in ([dict(id=2)], [dict(id=1),dict(id=1)]):
            with self.assertRaises(ValueError): check_history(dict(total_count=len(rows),workflow_runs=rows), {'1'})

    def test_overlay_keeps_original_training_code_and_no_holdout_in_validation(self):
        import base64, gzip, ast
        code, digest = controller.overlay('validation')
        compile(code,'worker-overlay','exec')
        tree = ast.parse(code)
        packed = next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
                      and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
        raw = gzip.decompress(base64.b64decode(packed))
        self.assertEqual(hashlib.sha256(raw).hexdigest(),digest)
        files = json.loads(raw)
        self.assertFalse(any(n.startswith(('src/','scripts/')) or 'holdout' in n for n in files))
        self.assertNotIn('configs/dinov3.json',files)
        self.assertEqual(json.loads(files['configs/dinov3-confirmation.json']),study.plan())

    def test_guard_failure_never_creates_and_cleanup_uses_exact_owner(self):
        api=Mock()
        with patch.object(controller,'preflight',side_effect=ValueError('ledger')):
            with self.assertRaises(ValueError): controller.launch(api)
        api.call.assert_not_called()
        pod=dict(name=controller.AUTH+'-123',env=dict(WAFER_MANAGED_BY=controller.REPOSITORY,DINO_RUN_ID='123',DINO_AUTH=controller.AUTH))
        self.assertTrue(controller.owned(pod,'123'))
        pod['env']['DINO_RUN_ID']='124'
        self.assertFalse(controller.owned(pod,'123'))

    def test_scoped_url_rejects_credentials_localhost_and_bad_host(self):
        self.assertEqual(validated_url('https://example.blob.core.windows.net/a?sig=test'),'https://example.blob.core.windows.net/a?sig=test')
        for url in ('http://example.blob.core.windows.net/a','https://localhost/a',
                    'https://user:password@example.blob.core.windows.net/a',
                    'https://example.blob.core.windows.net.evil.test/a'):
            with self.assertRaises(ValueError): validated_url(url)

    def test_cleanup_requires_consecutive_inventory_absence(self):
        api=Mock();api.listing.return_value=[]
        with tempfile.TemporaryDirectory() as tmp, patch.object(controller,'OUT',Path(tmp)), \
             patch.object(controller,'identity',return_value='123'), patch.object(controller.time,'sleep'):
            controller.cleanup(api)
            record=json.loads((Path(tmp)/'cleanup.json').read_text())
            self.assertEqual(record['checks'],12)
            self.assertEqual(record['status'],'verified_absent')
            self.assertEqual(api.listing.call_count,12)
            api.call.assert_not_called()

    def test_acceptance_counts_processes_and_reports_repeat_changes(self):
        from wafervlm.core import write_json, write_jsonl, metrics, sha256_file
        from wafervlm.experiment import seal_output
        p=study.plan()
        original=read_jsonl(study.ROOT/'docs/showcase/dinov3-seed17/predictions.jsonl')
        eps=[dict(name=n,weight_sha256='a'*64) for n in study.names()]
        spec=dict(controller_sha='b'*40,gpu='synthetic-test-gpu')
        with tempfile.TemporaryDirectory() as tmp, patch.object(study,'endpoints',return_value=eps):
            root=Path(tmp);write_json(root/'endpoints.json',eps)
            for i,ep in enumerate(eps):
                for repeat in (1,2,3):
                    folder=root/'validation'/ep['name']/f'repeat{repeat}';folder.mkdir(parents=True)
                    rows=copy.deepcopy(original)
                    if repeat==2:
                        target=(study.LABELS.index(rows[0]['prediction'])+1)%8
                        rows[0].update(prediction=study.LABELS[target],response=study.LABELS[target],probabilities=[float(j==target) for j in range(8)])
                    write_jsonl(folder/'predictions.jsonl',rows)
                    write_json(folder/'metrics.json',metrics(rows))
                    write_json(folder/'status.json',dict(status='completed',rows=160))
                    write_json(folder/'run.json',dict(endpoint=ep['name'],phase='validation',pid=100+i*3+repeat,
                        plan_sha256=sha256_file(study.PLAN),weight_sha256=ep['weight_sha256'],
                        base_weights_sha256=p['base_weights_sha256'],batch_size=8,
                        manifest_sha256=study.recipe()['validation_sha256'],image=p['image'],
                        image_source_sha=p['image_source_sha'],**spec,peak_allocated_bytes=1,
                        precision='bf16_autocast_fp32_parameters'))
                    seal_output(folder)
            report=study.verify_results(root,'validation',spec)
            self.assertEqual(report['validation_passes'],18)
            self.assertEqual(report['holdout_predictions'],0)
            self.assertEqual(report['passes']['seed17-linear'][1]['label_changes'],[original[0]['id']])
            self.assertEqual(report['passes']['seed17-linear'][2]['label_change_count'],0)
            folder=root/'validation'/'seed123-lora'/'repeat3'
            data=json.loads((folder/'run.json').read_text());data['pid']=101
            write_json(folder/'run.json',data);seal_output(folder)
            with self.assertRaisesRegex(ValueError,'process'):
                study.verify_results(root,'validation',spec)


if __name__=='__main__':
    unittest.main()
