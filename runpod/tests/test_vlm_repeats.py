import io
import json
import os
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from runpod import vlm_controller as c, vlm_ledger as ledger, vlm_worker as worker
from runpod import vlm_repeat_inputs as inputs


class RepeatLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        env = patch.dict(os.environ, WAFER_VLM_PHASE='repeat', GITHUB_RUN_ID='1234',
                         GITHUB_REPOSITORY=c.REPOSITORY, GH_TOKEN='test-only')
        env.start(); self.addCleanup(env.stop)
        out = patch.object(c, 'OUT', self.out); out.start(); self.addCleanup(out.stop)
        failed = patch.object(ledger, 'REPEAT_ATTEMPTS', [])
        failed.start(); self.addCleanup(failed.stop)
        self.ledger = dict(usd=8.132281676712218, seconds=38442.25880065918,
                          attempts=[dict(seed=s, accepted=True) for s in (42, 123)])
        self.evidence = dict(controller_sha='a'*40, config_sha256='b'*64,
            inputs={'manifest_sha256':'c'*64}, gpu='NVIDIA GeForce RTX 4090', regions=['TEST-1'],
            cost=c.quote(.89,self.ledger), ledger=self.ledger)

    def pod(self):
        return dict(id='pod1',name=c.AUTHORIZATION+'-vlmrepeats-1234',
            env={c.OWNER:c.REPOSITORY,'WAFER_RUN_ID':'1234','WAFER_AUTHORIZATION':c.AUTHORIZATION,
                 'WAFER_EXPERIMENT':'vlm-repeats','WAFER_STARTED':str(time.time()-60),'WAFER_RATE':'.89',
                 'WAFER_PRIOR_USD':str(self.ledger['usd']),'WAFER_PRIOR_SECONDS':str(self.ledger['seconds'])})

    def test_evaluation_reserves_full_60_plus_10_minutes(self):
        q=c.quote(.89,self.ledger)
        self.assertEqual(q['work_seconds'],3600)
        self.assertEqual(q['cleanup_seconds'],600)
        self.assertAlmostEqual(q['planned_total_usd'],9.181031676712218)
        with self.assertRaises(ValueError):c.quote(.89,dict(self.ledger,seconds=62000))
        with self.assertRaises(ValueError):c.quote(.89,dict(self.ledger,usd=29))

    def test_retry_deducts_failed_create_time_and_bootstrap_roundtrips(self):
        import ast, base64, gzip
        retry=dict(self.ledger,usd=8.16852580926767,seconds=38587.4081276989,repeat_seconds=145.14932703971863)
        q=c.quote(.89,retry)
        self.assertEqual(q['work_seconds'],3454)
        self.assertLessEqual(q['planned_total_seconds'],self.evidence['cost']['planned_total_seconds'])
        code=c.worker_code();self.assertLess(len(code),32000)
        assignment=next(n for n in ast.parse(code).body if isinstance(n,ast.Assign))
        encoded=assignment.value.args[0].args[0].args[0].value
        payload=json.loads(gzip.decompress(base64.b64decode(encoded)))
        for name,content in payload.items():self.assertEqual(content,(c.ROOT/name).read_text())

    def test_repeat_cleanup_recovers_ledger_and_cannot_delete_training(self):
        api=Mock();api.listing.side_effect=[[self.pod()],[],[],[]]
        with patch.object(c.time,'sleep'):c.cleanup(api)
        self.assertEqual(json.loads((self.out/'cleanup.json').read_text())['status'],'verified_absent')
        self.assertGreater(json.loads((self.out/'cleanup.json').read_text())['estimated_total_usd'],self.ledger['usd'])
        (self.out/'provision.json').unlink()
        wrong=self.pod();wrong['env']['WAFER_EXPERIMENT']='vlm-seed123'
        api=Mock();api.listing.return_value=[wrong]
        with self.assertRaises(RuntimeError):c.cleanup(api)
        api.call.assert_not_called()

    def test_repeat_launch_has_one_create_and_no_account_credentials(self):
        api=Mock();api.call.side_effect=RuntimeError('unknown create outcome')
        with patch.object(c,'authorization_guard'),patch.object(c,'preflight',return_value=self.evidence), \
             patch.object(c,'cleanup') as cleanup,patch('builtins.print'),self.assertRaises(RuntimeError):
            c.launch(api)
        self.assertEqual(api.call.call_count,1)
        payload=api.call.call_args.args[2]
        self.assertEqual(payload['env']['WAFER_EXPERIMENT'],'vlm-repeats')
        self.assertNotIn('WAFER_SEED',payload['env'])
        self.assertNotIn('WAFER_INPUT_URLS',payload['env'])
        self.assertFalse({'GH_TOKEN','GITHUB_TOKEN','RUNPOD_API_KEY'} & payload['env'].keys())
        self.assertNotIn('scoped-read-url',(self.out/'provision.json').read_text())
        self.assertNotIn('scoped-read-url',(self.out/'launch-spec.json').read_text())
        cleanup.assert_called_once_with(api)

    def test_missing_input_or_seed_acceptance_prevents_any_provider_call(self):
        for attempts in ([dict(seed=42,accepted=True)],self.ledger['attempts']):
            api=Mock()
            with patch.object(c,'provenance',return_value={}), \
                 patch.object(ledger,'reconcile',return_value=dict(self.ledger,attempts=attempts)), \
                 patch.object(inputs,'preflight_inputs',side_effect=ValueError('bad frozen weights')), \
                 self.assertRaises((ValueError,RuntimeError)):
                c.preflight(api)
            api.call.assert_not_called();api.listing.assert_not_called()

    def test_repeat_marker_cannot_authorize_training(self):
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:
            z.writestr('run-id.txt','1234\n')
            z.writestr('intent.json',json.dumps(dict(phase='validation-repeats',controller_sha='a'*40,
                                                    run_id='1234',workload='vlm-repeats')))
        ledger.verify_marker(out.getvalue(),1234,True,'a'*40,'repeat')
        with self.assertRaises(RuntimeError):ledger.verify_marker(out.getvalue(),1234,True,'a'*40,123)

    def test_signed_download_destination_and_corrupted_cache_are_rejected(self):
        for url in ('http://example.com/a','https://github.com/a','https://user:pass@x.blob.core.windows.net/a',
                    'https://x.blob.core.windows.net.evil.example/a','https://x.blob.core.windows.net:8443/a'):
            with self.subTest(url=url),self.assertRaises(ValueError):inputs.validated_url(url)
        (self.out/'a.zip').write_bytes(b'corrupt')
        with patch.object(inputs,'inputs',return_value={'a':dict(zip_sha256='a'*64)}),self.assertRaises(ValueError):
            inputs.archives(self.out,urls={'a':'https://x.blob.core.windows.net/a'})

    def test_progress_reports_completed_passes_without_exposing_logs(self):
        (self.out/'repeats').mkdir();(self.out/'repeats/passes.json').write_text(json.dumps([{}]*7))
        with patch.object(worker,'OUT',self.out),patch.object(worker,'RESULT',dict(status='running',phase='validation')):
            self.assertEqual(worker.progress_snapshot()['progress'],dict(completed_passes=7,total_passes=18))
            (self.out/'repeats/passes.json').write_text('[')
            self.assertNotIn('progress',worker.progress_snapshot())

    def test_deleted_marker_cannot_hide_a_prior_paid_repeat(self):
        def read(url,**kw):
            if 'vlm-repeats.yml/runs?' in url:
                return {'workflow_runs':[dict(id=999,status='completed',run_attempt=1)]}
            if url.endswith('/999/jobs?per_page=100'):
                return {'jobs':[{'steps':[dict(name='Validate 18 passes, retrieve evidence, delete Pod',conclusion='failure')]}]}
            return {'workflow_runs':[]}
        with patch.object(ledger,'reconciled_precreate',return_value=None), \
             patch.object(ledger,'completed_receipts',return_value=([],self.ledger['usd'],self.ledger['seconds'])), \
             patch.object(ledger,'request',side_effect=read),self.assertRaisesRegex(RuntimeError,'prior paid repeat'):
            ledger.reconcile('1234',False,self.out,seed='repeat')


if __name__ == '__main__':
    unittest.main()
