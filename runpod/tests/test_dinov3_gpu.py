import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from runpod import dinov3_controller as c

GPU5090 = 'NVIDIA GeForce RTX 5090'


class GPUSelectionTests(unittest.TestCase):
    def test_explicit_gpu_price_and_aggregate_budget(self):
        with patch.dict(os.environ, DINO_GPU=GPU5090):
            q = c.cost(1.19)
            self.assertEqual(c.selected_gpu(), GPU5090)
            self.assertEqual(q['total_cap_usd'], 3.5)
            self.assertEqual(q['prior_spend_usd'], .1)
            self.assertEqual(q['prior_runtime_seconds'], 360)
            self.assertLess(q['estimated_max_usd'], 2)
            self.assertGreater(q['reserve_usd'], 1)
            with self.assertRaises(ValueError): c.cost(1.20)
        with patch.dict(os.environ, DINO_GPU=''):
            self.assertEqual(c.selected_gpu(), c.GPU)
            with self.assertRaises(ValueError): c.cost(.90)
        with patch.dict(os.environ, DINO_GPU='unapproved'):
            with self.assertRaises(ValueError): c.selected_gpu()

    def test_5090_is_quoted_and_created_without_exact_cuda_filter(self):
        api = Mock()
        api.listing.return_value = []
        api.call.return_value = dict(price={'secure':1.19},dataCenters=[{'id':'US-NC-1','availability':'HIGH'}])
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,'OUT',Path(tmp)), \
                patch.dict(os.environ,DINO_GPU=GPU5090,PARALLEL_VLM_RUN_ID=''), \
                patch.object(c,'manual_guard',return_value=('a'*40,'fake')), \
                patch.object(c,'weight_report',return_value={}), patch.object(c,'cache_files',return_value={}), \
                patch.object(c,'image_provenance'), patch.object(c,'budget',return_value={}) as account:
            evidence = c.preflight(api)
            url = api.call.call_args.args[0]
            self.assertIn('5090', url)
            self.assertIn('minCudaVersion=12.8', url)
            account.assert_called_once_with(api,reserve=3.5)
            api.call.reset_mock()
            api.call.side_effect = RuntimeError('ambiguous create')
            evidence['image'] = 'image@sha256:'+'b'*64
            with patch.object(c,'preflight',return_value=evidence), patch.object(c,'ledger_guard'), \
                    patch.object(c,'identity',return_value='7'), patch.object(c,'cleanup') as cleanup, patch('builtins.print'):
                with self.assertRaises(RuntimeError): c.launch(api)
                cleanup.assert_called_once_with(api)
            self.assertEqual(api.call.call_count,1)
            payload = api.call.call_args.args[2]
            self.assertEqual(payload['gpu']['id'],GPU5090)
            self.assertEqual(payload['env']['DINO_GPU'],GPU5090)
            self.assertEqual(payload['gpu']['minCudaVersion'],'12.8')
            self.assertNotIn('allowedCudaVersions',payload['gpu'])

    def test_independent_cleanup_recovers_5090_without_dispatch_gpu_input(self):
        api = Mock()
        pod = dict(id='owned',name=c.AUTH+'-7',mounts=[],env={
            'WAFER_MANAGED_BY':c.REPOSITORY,'DINO_RUN_ID':'7','DINO_AUTH':c.AUTH,
            'DINO_STARTED':'900','DINO_RATE':'1.19','DINO_GPU':GPU5090})
        api.listing.side_effect = [[pod],[],[],[]]
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,'OUT',Path(tmp)), \
                patch.dict(os.environ,DINO_GPU=''), patch.object(c,'identity',return_value='7'), \
                patch.object(c.time,'time',return_value=1000), patch.object(c.time,'sleep'):
            c.cleanup(api)
            result=json.loads((Path(tmp)/'cleanup.json').read_text())
            state=json.loads((Path(tmp)/'provision.json').read_text())
        self.assertEqual(state['gpu'],GPU5090)
        self.assertTrue(result['within_cap'])
        self.assertEqual(result['status'],'verified_absent')
        api.call.assert_called_once_with('/pods/owned','DELETE')
