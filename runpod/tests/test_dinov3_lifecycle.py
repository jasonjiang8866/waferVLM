import copy
import hashlib
import json
import os
import tempfile
import threading
import types
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch
from http.server import ThreadingHTTPServer

from runpod import dinov3_controller as c, dinov3_worker as w
from runpod.dinov3_bundle import recipe, select, trials, weight_report


class DinoLifecycleTests(unittest.TestCase):
    def setUp(self):
        # Production workflow inputs must never turn unit fakes into live API
        # calls or replace their intentionally synthetic one-run history.
        env = patch.dict(os.environ, PARALLEL_VLM_RUN_ID='', REPLACE_QUEUED_RUN_ID='')
        env.start()
        self.addCleanup(env.stop)

    def test_budget_includes_disk_cleanup_and_reserve(self):
        q = c.cost(.89)
        self.assertLess(q['estimated_max_usd'], 1.5)
        self.assertGreater(q['reserve_usd'], 1)
        for rate in [True, float('nan'), float('inf'), 0, .90]:
            with self.assertRaises(ValueError):
                c.cost(rate)

    def test_lr_selection_requires_complete_sweep_and_uses_lower_lr_ties(self):
        rows = [{**t, 'macro_f1': .5} for t in trials()]
        chosen = select(rows)
        self.assertEqual(chosen['linear']['learning_rate'], .001)
        self.assertEqual(chosen['lora']['learning_rate'], .00005)
        for changed in [rows[:-1], rows[::-1], [{**rows[0], 'seed': 42}, *rows[1:]]]:
            with self.assertRaises(ValueError):
                select(changed)

    def test_no_paid_call_on_guard_failure(self):
        api = Mock()
        with patch.object(c, 'ledger_guard', side_effect=ValueError('guard')):
            with self.assertRaises(ValueError):
                c.launch(api)
        api.call.assert_not_called()

    def test_old_dispatch_blocks_even_if_its_artifact_expired(self):
        with patch.object(c, 'manual_guard', return_value=('a'*40, 'fake')), patch.object(c, 'identity', return_value='7'), \
                patch.object(c, 'request', return_value={'total_count': 2, 'workflow_runs': [{'id': 7}, {'id': 6}]}):
            with self.assertRaisesRegex(ValueError, 'Prior DINO'):
                c.ledger_guard()

    def test_exact_upload_receipt_works_without_artifact_listing(self):
        artifact = dict(id=99, name=c.MARKER, expired=False, digest='sha256:'+'b'*64,
                        workflow_run={'id': 7, 'head_sha': 'a'*40})
        def read(url, **kwargs):
            if '/workflows/' in url:
                return {'total_count': 1, 'workflow_runs': [{'id': 7}]}
            self.assertTrue(url.endswith('/artifacts/99'))
            return artifact
        with patch.object(c, 'manual_guard', return_value=('a'*40, 'fake')), \
                patch.object(c, 'identity', return_value='7'), patch.object(c, 'request', side_effect=read), \
                patch.dict(os.environ, AUTHORIZATION_ARTIFACT_ID='99', EXPECTED_SHA='a'*40):
            c.ledger_guard()
            for key, value in [('id',8), ('head_sha','c'*40)]:
                previous = artifact['workflow_run'][key]
                artifact['workflow_run'][key] = value
                with self.assertRaises(ValueError):
                    c.ledger_guard()
                artifact['workflow_run'][key] = previous

    def test_upload_receipt_404_retries_reads_only(self):
        artifact = dict(id=99, name=c.MARKER, expired=False, digest='sha256:'+'b'*64,
                        workflow_run={'id': 7, 'head_sha': 'a'*40})
        with patch.object(c, 'manual_guard', return_value=('a'*40, 'fake')), \
                patch.object(c, 'identity', return_value='7'), patch.object(c.time, 'sleep'), \
                patch.object(c, 'request', side_effect=[
                    {'total_count':1,'workflow_runs':[{'id':7}]},
                    c.RemoteHTTPError(404, []), artifact]) as read, \
                patch.dict(os.environ, AUTHORIZATION_ARTIFACT_ID='99', EXPECTED_SHA='a'*40):
            c.ledger_guard()
            self.assertEqual(read.call_count, 3)
            self.assertTrue(all('method' not in x.kwargs for x in read.call_args_list))

    def test_live_preflight_uses_minimum_and_4090(self):
        api = Mock()
        api.listing.return_value = []
        api.call.return_value = dict(price={'secure': .69}, dataCenters=[{'id': 'EU-RO-1', 'availability': 'HIGH'}])
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'OUT', Path(tmp)), \
                patch.object(c, 'manual_guard', return_value=('a'*40, 'fake')), \
                patch.object(c, 'weight_report', return_value={}), patch.object(c, 'cache_files'), \
                patch.object(c, 'image_provenance'), patch.object(c, 'budget', return_value={}), \
                patch.dict(os.environ, DINO_IMAGE='test'):
            result = c.preflight(api)
        self.assertEqual(result['gpu'], c.GPU)
        self.assertIn('minCudaVersion=12.8', api.call.call_args.args[0])
        self.assertNotIn('allowedCudaVersions', api.call.call_args.args[0])
        self.assertEqual(api.call.call_count, 1)

    def test_cleanup_refuses_foreign_pod_or_persistent_mount(self):
        api = Mock()
        for pod in [dict(id='abc', name=c.AUTH+'-7', env={}),
                    dict(id='abc', name=c.AUTH+'-7', env={'WAFER_MANAGED_BY': c.REPOSITORY, 'DINO_RUN_ID': '7', 'DINO_AUTH': c.AUTH}, mounts=['persistent'])]:
            api.listing.return_value = [pod]
            with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'OUT', Path(tmp)), patch.object(c, 'identity', return_value='7'):
                with self.assertRaises(ValueError):
                    c.cleanup(api)
        api.call.assert_not_called()

    def test_unknown_create_is_never_retried_and_always_cleans_up(self):
        api = Mock()
        api.call.side_effect = RuntimeError('ambiguous create')
        evidence = dict(source_sha='a'*40, image='image@sha256:'+'b'*64, weights={},
                        cost=c.cost(.69), min_cuda_version='12.8', regions=['EU-RO-1'])
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'OUT', Path(tmp)), \
                patch.object(c, 'identity', return_value='7'), patch.object(c, 'ledger_guard'), \
                patch.object(c, 'preflight', return_value=evidence), patch.object(c, 'cache_files', return_value={}), \
                patch.object(c, 'cleanup') as cleanup, patch('builtins.print'):
            with self.assertRaises(RuntimeError):
                c.launch(api)
            self.assertEqual(api.call.call_count, 1)
            cleanup.assert_called_once_with(api)
            payload = api.call.call_args.args[2]
            self.assertNotIn('HF_TOKEN', payload['env'])
            self.assertNotIn('RUNPOD_API_KEY', payload['env'])
            self.assertEqual(payload['gpu']['minCudaVersion'], '12.8')

    def test_absence_is_repeated_and_unknown_create_waits_longer(self):
        api = Mock()
        api.listing.return_value = []
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'OUT', Path(tmp)), \
                patch.object(c, 'identity', return_value='7'), patch.object(c.time, 'sleep'):
            c.cleanup(api)
            result = json.loads((Path(tmp)/'cleanup.json').read_text())
        self.assertEqual(api.listing.call_count, 12)
        self.assertEqual(result['status'], 'verified_absent')

    def test_worker_runs_six_trials_and_two_reloads_without_cli_log_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / 'data'
            dataset.mkdir()
            for name in ('dataset.json', 'validation.jsonl', 'acquisition.json'):
                (dataset/name).write_text('{}')
            cfg = {**recipe(), 'dataset_sha256': hashlib.sha256(b'{}').hexdigest()}
            calls = []
            def process(command, **kwargs):
                calls.append(command)
                if 'train' in command or 'evaluate' in command:
                    destination = Path(command[command.index('--output')+1])
                    # The real CLI refuses an existing sibling .log before launch.
                    self.assertFalse(destination.with_suffix('.log').exists())
                    destination.mkdir()
                    (destination/'metrics.json').write_text('{"macro_f1":0.5}')
                self.assertNotIn('HF_TOKEN', kwargs['env'])
                self.assertNotIn('RUNPOD_API_KEY', kwargs['env'])
                self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
                return Mock(wait=Mock(return_value=0), poll=Mock(return_value=0))
            ready = threading.Event()
            ready.set()
            torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True,
                is_bf16_supported=lambda:True,get_device_name=lambda:'synthetic GPU'),
                __version__='test', version=types.SimpleNamespace(cuda='12.8'))
            with patch.object(w, 'ROOT', root), patch.object(w, 'OUT', root/'result'), \
                    patch.object(w, 'ARCHIVE', root/'result.tar.gz'), patch.object(w, 'DATASET', dataset), \
                    patch.object(w, 'recipe', return_value=cfg), patch.object(w, 'READY', ready), \
                    patch.object(w, 'STATE', {}), patch.dict('sys.modules', torch=torch), \
                    patch.object(w.subprocess, 'Popen', side_effect=process), \
                    patch.dict(os.environ, WAFERVLM_SOURCE_SHA='a'*40, DINO_IMAGE='image',
                        DINO_DEADLINE=str(c.time.time()+5400), HF_TOKEN='test-only', RUNPOD_API_KEY='test-only'):
                w.work()
                self.assertEqual(w.STATE['status'], 'accepted')
                self.assertEqual(sum('train' in command for command in calls), 6)
                self.assertEqual(sum('evaluate' in command for command in calls), 2)
                self.assertTrue((root/'result.tar.gz').is_file())

    def test_authenticated_weight_delivery_hashes_and_no_overwrite(self):
        data = {'config.json': b'{"model_type":"dinov3_vit"}', 'model.safetensors': b'synthetic fixture only'}
        with tempfile.TemporaryDirectory() as tmp, patch.object(w, 'TOKEN', 'test-only'), \
                patch.object(w, 'WEIGHTS', {'files_sha256': {n: hashlib.sha256(v).hexdigest() for n,v in data.items()}}), \
                patch.object(w, 'STATE', {}), patch.object(w, 'READY', threading.Event()), \
                patch.object(w, 'cache_root', return_value=Path(tmp)):
            server = ThreadingHTTPServer(('127.0.0.1', 0), w.Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = 'http://127.0.0.1:' + str(server.server_port)
            def put(name, content, token='test-only'):
                req = urllib.request.Request(url+'/weights/'+name, data=content, method='PUT',
                    headers={'Authorization': 'Bearer '+token,
                             'Content-Range': f'bytes 0-{len(content)-1}/{len(content)}',
                             'X-Chunk-SHA256': hashlib.sha256(content).hexdigest()})
                return urllib.request.urlopen(req)
            try:
                for name, content, token in [('config.json', data['config.json'], 'wrong'),
                                             ('../escape', b'x', 'test-only'),
                                             ('config.json', b'wrong hash', 'test-only')]:
                    with self.assertRaises(urllib.error.HTTPError):
                        put(name, content, token)
                for name, content in data.items():
                    with put(name, content) as response:
                        self.assertEqual(response.status, 204)
                self.assertTrue(w.READY.is_set())
                with put('config.json', data['config.json']) as replay:
                    self.assertEqual(replay.status, 204)
                with self.assertRaises(urllib.error.HTTPError):
                    put('config.json', b'cannot replace verified content')
                self.assertEqual((Path(tmp)/'model.safetensors').read_bytes(), data['model.safetensors'])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
