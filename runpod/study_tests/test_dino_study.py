"""Offline checks for identity, budget, one-create and scoped-input boundaries."""
import copy
import hashlib
import json
import tempfile
import unittest
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch, Mock

from runpod import dinov3_study as study
from runpod import dinov3_study_controller as controller
from runpod.dinov3_study_ledger import check_history
from runpod.dinov3_study_worker import validated_url
from wafervlm.core import read_jsonl


class StudyTests(unittest.TestCase):
    def test_weight_input_is_original_fingerprint_not_verification_receipt(self):
        from runpod import dinov3_image_weights as weights
        published=json.loads((controller.ROOT/'docs/evidence/dinov3-seed17-43716864164.json').read_text())['weights']
        # The previous wiring is rejected before touching the actual files.
        with patch.object(weights,'snapshot',side_effect=RuntimeError('reached file verification')) as snapshot:
            with self.assertRaisesRegex(ValueError,'CPU fingerprint'):
                weights.verify_files(published)
            snapshot.assert_not_called()
            with self.assertRaisesRegex(RuntimeError,'reached file verification'):
                weights.verify_files(controller.weight_fingerprint())
            snapshot.assert_called_once()

    def test_archive_download_keeps_working_proxy_headers_and_checks_hash(self):
        payload=b'controlled archive bytes'
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                allowed=(self.path=='/archive' and self.headers.get('User-Agent')=='waferVLM-actions/0.1'
                         and self.headers.get('Authorization')=='Bearer test-only')
                self.send_response(200 if allowed else 403);self.end_headers()
                if allowed:self.wfile.write(payload)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url='http://127.0.0.1:'+str(server.server_port)
            meta=dict(size=len(payload),sha256=hashlib.sha256(payload).hexdigest())
            self.assertEqual(controller.download_archive(url,'test-only',meta),payload)
            with self.assertRaises(ValueError):controller.download_archive(url,'test-only',dict(meta,sha256='0'*64))
            with self.assertRaises(ValueError):controller.download_archive(url,'test-only',dict(meta,size=len(payload)-1))
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_worker_failure_retains_safe_frame_location(self):
        import os
        from runpod import dinov3_study_worker as worker
        env=dict(DINO_PHASE='validation',DINO_CONTROLLER_SHA='c'*40,WAFERVLM_SOURCE_SHA='d'*40,
                 DINO_IMAGE='test-image',DINO_OVERLAY_SHA256='e'*64)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,env), \
             patch.object(worker.w,'OUT',Path(tmp)/'results'),patch.object(worker.w,'ARCHIVE',Path(tmp)/'result.tar.gz'), \
             patch.object(worker.w,'STATE',{}),patch.object(worker,'plan',side_effect=ValueError('private-url-do-not-publish')):
            worker.work()
            result=json.loads((Path(tmp)/'results/worker.json').read_text())
            self.assertEqual(result['status'],'rejected')
            self.assertEqual(result['failed_phase'],'protocol')
            self.assertTrue(result['error_frames'])
            self.assertNotIn('private-url-do-not-publish',json.dumps(result))

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

    def test_diagnostics_keep_http_categories_without_remote_payloads(self):
        from runpod.provider import RemoteHTTPError
        with tempfile.TemporaryDirectory() as tmp, patch.object(controller,'OUT',Path(tmp)):
            controller.stage('create_pod')
            result=controller.error_record(RemoteHTTPError(400,['capacity']))
            self.assertEqual(result,dict(error_type='RemoteHTTPError',http_status=400,categories=['capacity'],stage='create_pod'))
            sensitive='https://example.test?token=do-not-publish'
            self.assertNotIn(sensitive,json.dumps(controller.error_record(RuntimeError(sensitive))))

    def test_catalog_reobserves_shortage_without_creating(self):
        api=Mock(); gpu='NVIDIA GeForce RTX 5090'
        api.call.side_effect=[dict(price=dict(secure=1.39),dataCenters=[]),
                              dict(price=dict(secure=1.39),dataCenters=[dict(id='EUR-IS-1',availability='LOW')])]
        with tempfile.TemporaryDirectory() as tmp, patch.object(controller,'OUT',Path(tmp)), patch.object(controller.time,'sleep') as sleep:
            selected=controller.select_capacity(api,[gpu],dict(usd=.71,seconds=2000))
            self.assertEqual(selected[:2],(gpu,['EUR-IS-1']))
            self.assertEqual(sleep.call_args_list,[((15,),)])
            self.assertEqual(len(json.loads((Path(tmp)/'catalog.json').read_text())),2)
            self.assertTrue(all(len(c.args)==1 and c.args[0].startswith('/catalog/gpus/') for c in api.call.call_args_list))

    def test_catalog_wait_is_bounded_and_keeps_price_cap(self):
        api=Mock();gpu='NVIDIA GeForce RTX 5090'
        api.call.return_value=dict(price=dict(secure=1.40),dataCenters=[dict(id='EUR-IS-1',availability='HIGH')])
        with tempfile.TemporaryDirectory() as tmp, patch.object(controller,'OUT',Path(tmp)), patch.object(controller.time,'sleep') as sleep:
            with self.assertRaises(ValueError):controller.select_capacity(api,[gpu],dict(usd=.71,seconds=2000))
            self.assertEqual(api.call.call_count,4)
            self.assertEqual(sleep.call_count,3)

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
