import hashlib
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
from runpod.tests.test_confirmation_contract import specification


def zipped(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as z:
        for name, value in files.items():
            z.writestr(name, value)
    return out.getvalue()


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.env = patch.dict(os.environ, GH_TOKEN='test-token', GITHUB_SHA='a'*40)
        self.env.start(); self.addCleanup(self.env.stop)
        self.precreate = patch.object(ledger, 'reconciled_precreate', return_value=None)
        self.precreate.start(); self.addCleanup(self.precreate.stop)
        self.artifacts = [dict(name=ledger.MARKER, id=x['marker_artifact_id'],
            digest='sha256:'+x['marker_sha256'], expired=False,
            workflow_run={'id': x['run_id']}) for x in ledger.HISTORY]
        self.current = dict(name=ledger.MARKER, id=999, digest='sha256:'+'a'*64,
                            expired=False, workflow_run={'id': 1234, 'head_sha':'a'*40})
        cleanup = json.loads((c.ROOT/'docs/evidence/continuation-42682266526.json').read_text())['cleanup']
        self.raw = zipped({'cleanup.json': json.dumps(cleanup)})
        self.digest_patch = patch.object(ledger, 'RESULT_SHA', hashlib.sha256(self.raw).hexdigest())
        self.digest_patch.start(); self.addCleanup(self.digest_patch.stop)

    def download(self, artifact_id, *args):
        if artifact_id == ledger.RESULT_ID:
            return self.raw
        if artifact_id == 999:
            return zipped({'run-id.txt': '1234\n', 'intent.json': json.dumps(dict(
                seed=42, controller_sha='a'*40, run_id='1234', workload='vlm-seed42'))})
        row = next(x for x in ledger.HISTORY if x['marker_artifact_id'] == artifact_id)
        return zipped({'run-id.txt': str(row['run_id'])+'\n'})

    def reconcile(self, artifacts, require_current=False):
        with patch.object(ledger, 'request', side_effect=lambda url, **kw:
                          {'artifacts': artifacts} if '/artifacts?' in url else {'workflow_runs': []}), \
             patch.object(ledger, 'artifact_bytes', side_effect=self.download):
            return ledger.reconcile('1234', require_current, self.out)

    def test_deleted_marker_cannot_hide_a_previous_paid_dispatch(self):
        previous = {'id':888, 'status':'completed', 'run_attempt':1}
        with patch.object(ledger, 'request', side_effect=[{'workflow_runs':[previous]},
                {'jobs':[{'steps':[{'name':ledger.PAID_STEP,'conclusion':'failure'}]}]}]), \
             self.assertRaisesRegex(RuntimeError, 'prior paid'):
            ledger.reconcile('1234',False,self.out)

    def test_all_seven_attempts_and_current_intent_verified(self):
        result = self.reconcile(self.artifacts+[self.current], True)
        self.assertEqual(len(result['attempts']), 7)
        self.assertEqual(result['usd'], 5.903039)
        self.assertEqual(result['seconds'], 491.911*60)

    def test_unknown_attempt_missing_history_and_wrong_digest_block(self):
        for rows in [self.artifacts[1:], self.artifacts+[dict(self.current, workflow_run={'id':888})],
                     self.artifacts+[dict(self.current, name='runpod-authorization-dinov3')],
                     [dict(self.artifacts[0], digest='sha256:'+'0'*64)]+self.artifacts[1:],
                     self.artifacts+[self.artifacts[0]]]:
            with self.subTest(rows=len(rows)), self.assertRaises(RuntimeError):
                self.reconcile(rows)

    def test_paid_mode_requires_current_marker(self):
        with patch.dict(os.environ, AUTHORIZATION_ARTIFACT_ID=''), self.assertRaises(RuntimeError):
            self.reconcile(self.artifacts, True)

    def test_upload_receipt_survives_list_visibility_delay(self):
        def read(url, **kwargs):
            if '/artifacts?' in url: return {'artifacts':self.artifacts}
            if url.endswith('/artifacts/999'): return self.current
            return {'workflow_runs':[]}
        with patch.dict(os.environ, AUTHORIZATION_ARTIFACT_ID='999'), \
             patch.object(ledger,'request',side_effect=read), \
             patch.object(ledger,'artifact_bytes',side_effect=self.download):
            self.assertEqual(len(ledger.reconcile('1234',True,self.out)['attempts']),7)

    def test_receipt_cannot_substitute_another_run_or_commit(self):
        for field, value in [('id',888),('head_sha','c'*40)]:
            artifact = dict(self.current, workflow_run=dict(self.current['workflow_run']))
            artifact['workflow_run'][field] = value
            with self.assertRaises(RuntimeError): ledger.verify_current_artifact(artifact,'1234')

    def test_corrupted_historical_cache_blocks(self):
        (self.out/'historical-result.zip').write_bytes(b'changed')
        with self.assertRaises(RuntimeError):
            self.reconcile(self.artifacts)

    def test_wrong_current_seed_and_duplicate_zip_members_block(self):
        data = zipped({'run-id.txt':'1234', 'intent.json':json.dumps(dict(
            seed=123, controller_sha='a'*40, run_id='1234', workload='vlm-seed42'))})
        with self.assertRaises(RuntimeError):
            ledger.verify_marker(data, 1234, True)

    def test_dataset_metadata_reconstructed_without_images(self):
        target = ledger.dataset_metadata(self.out)
        self.assertEqual([x.name for x in target.iterdir()], ['dataset.json'])
        self.assertEqual(hashlib.sha256((target/'dataset.json').read_bytes()).hexdigest(),
                         'c588b3ef59d9dba57cfc745ec20db903e6ff2dcc75504112348f8b9f1ad3ab68')


class PrecreateReconciliation(unittest.TestCase):
    def setUp(self):
        self.p = ledger.PRECREATE
        self.run = dict(head_sha=self.p['controller_sha'], run_attempt=1,
                        status='completed', conclusion='failure')
        self.result = zipped({'cleanup.json':json.dumps(self.p['cleanup']),
                              'ledger.json':'{}','preflight.json':'{}'})
        self.cleanup = zipped({'cleanup.json':json.dumps(self.p['independent_cleanup'])})

    def test_exact_failed_guard_run_is_reconciled(self):
        with patch.dict(os.environ, GH_TOKEN='test-token'), \
             patch.object(ledger,'request',return_value=self.run), \
             patch.object(ledger,'artifact_bytes',side_effect=[self.result,self.cleanup]):
            self.assertEqual(ledger.reconciled_precreate()['run_id'],43553982790)

    def test_rerun_cannot_inherit_zero_cost_reconciliation(self):
        with patch.dict(os.environ, GH_TOKEN='test-token'), \
             patch.object(ledger,'request',return_value=dict(self.run,run_attempt=2)), \
             patch.object(ledger,'artifact_bytes') as download, self.assertRaises(RuntimeError):
            ledger.reconciled_precreate()
        download.assert_not_called()

    def test_resource_intent_or_changed_cleanup_blocks_reconciliation(self):
        bad=zipped({'cleanup.json':json.dumps(self.p['cleanup']), 'ledger.json':'{}',
                    'preflight.json':'{}', 'provision.json':'{}'})
        with patch.dict(os.environ, GH_TOKEN='test-token'), \
             patch.object(ledger,'request',return_value=self.run), \
             patch.object(ledger,'artifact_bytes',return_value=bad), self.assertRaises(RuntimeError):
            ledger.reconciled_precreate()


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        patches = [patch.object(c, 'OUT', self.out), patch.dict(os.environ,
            GITHUB_REPOSITORY=c.REPOSITORY, GITHUB_RUN_ID='1234', GITHUB_RUN_ATTEMPT='1',
            GITHUB_EVENT_NAME='workflow_dispatch', GITHUB_REF='refs/heads/main',
            EXPECTED_SHA='a'*40, CONFIRM_PAID='true', GH_TOKEN='test-token')]
        for p in patches:
            p.start(); self.addCleanup(p.stop)
        self.evidence = dict(controller_sha='a'*40, config_sha256='b'*64,
            gpu='NVIDIA GeForce RTX 4090', regions=['TEST-1'], cost=c.quote(.89),
            ledger=specification()['ledger'])

    def pod(self):
        return dict(id='pod1', name=c.AUTHORIZATION+'-vlm42-1234',
            env={c.OWNER:c.REPOSITORY, 'WAFER_RUN_ID':'1234',
                 'WAFER_AUTHORIZATION':c.AUTHORIZATION, 'WAFER_EXPERIMENT':'vlm-seed42',
                 'WAFER_STARTED':str(time.time()-60), 'WAFER_RATE':'.89'},
            gpu={'id':self.evidence['gpu']}, dataCenterId='TEST-1', cost=.89)

    def test_source_diff_compares_image_commit_to_head(self):
        with patch.object(c, 'git', side_effect=['a'*40, 'src/wafervlm/modeling.py']) as git:
            with self.assertRaises(RuntimeError): c.provenance()
        self.assertEqual(git.call_args.args[:4], ('diff', '--name-only', c.SOURCE, 'a'*40))

    def test_expiry_rerun_wrong_ref_and_wrong_sha_rejected(self):
        with patch.object(c, 'git', return_value='a'*40):
            for key, value in [('GITHUB_RUN_ATTEMPT','2'), ('GITHUB_REF','refs/heads/feature'),
                               ('EXPECTED_SHA','b'*40), ('CONFIRM_PAID','false')]:
                with patch.dict(os.environ, {key:value}), self.assertRaises(RuntimeError):
                    c.authorization_guard()
            with patch.object(c.time, 'time', return_value=c.AUTHORIZATION_EXPIRES), self.assertRaises(RuntimeError):
                c.authorization_guard()

    def test_unknown_create_is_never_retried_and_always_cleans(self):
        api = Mock(); api.call.side_effect = RuntimeError('unknown outcome')
        with patch.object(c,'authorization_guard'), patch.object(c,'preflight',return_value=self.evidence), \
             patch.object(c,'cleanup') as cleanup, patch('builtins.print'), self.assertRaises(RuntimeError):
            c.launch(api)
        self.assertEqual(api.call.call_count, 1)
        args = api.call.call_args.args
        self.assertEqual(args[:2], ('/pods','POST'))
        self.assertEqual(args[2]['image'], c.IMAGE)
        self.assertEqual(args[2]['gpu']['minCudaVersion'], '12.8')
        self.assertEqual(args[2]['env']['WAFER_SEED'], '42')
        self.assertFalse({'RUNPOD_API_KEY','GH_TOKEN','GITHUB_TOKEN'} & args[2]['env'].keys())
        self.assertIsNone(json.loads((self.out/'provision.json').read_text())['pod_id'])
        cleanup.assert_called_once_with(api)

    def test_cleanup_recovers_exact_owned_pod_and_retains_cumulative_cost(self):
        api = Mock(); api.listing.side_effect = [[self.pod()],[],[],[]]
        with patch.object(c.time,'sleep'): c.cleanup(api)
        api.call.assert_called_once_with('/pods/pod1','DELETE')
        result = json.loads((self.out/'cleanup.json').read_text())
        self.assertEqual(result['status'],'verified_absent')
        self.assertGreater(result['estimated_total_usd'], 5.903039)
        self.assertGreater(result['estimated_total_seconds'], 491.911*60)

    def test_cleanup_refuses_other_experiment_even_with_same_owner(self):
        pod = self.pod(); pod['env']['WAFER_EXPERIMENT']='dinov3'
        api = Mock(); api.listing.return_value = [pod]
        with self.assertRaises(RuntimeError): c.cleanup(api)
        api.call.assert_not_called()

    def test_no_capacity_saves_read_only_observation(self):
        api=Mock(); api.listing.return_value=[]
        api.call.return_value={'price':{'secure':.5},'dataCenters':[]}
        with patch.object(c,'provenance',return_value={}), \
             patch.object(ledger,'reconcile',return_value=specification()['ledger']), self.assertRaises(RuntimeError):
            c.preflight(api)
        self.assertTrue(all(len(x.args)==1 for x in api.call.call_args_list))
        self.assertEqual(json.loads((self.out/'preflight.json').read_text())['status'], 'blocked_no_approved_gpu')

    def test_bound_includes_prior_history_disk_and_cleanup(self):
        q=c.quote(.89)
        self.assertEqual(q['work_seconds'],11100)
        self.assertEqual(q['cleanup_seconds'],600)
        self.assertAlmostEqual(q['planned_total_usd'], 8.824556857142857)
        self.assertEqual(q['planned_total_seconds'],(491.911+195)*60)

    def test_worker_telemetry_uses_latest_stage_only(self):
        (self.out/'training').mkdir()
        (self.out/'training/train-100.log').write_text('100/100 [done]')
        (self.out/'training/train-150.log').write_text('123/150 [running]')
        with patch.object(worker,'OUT',self.out), patch.object(worker,'RESULT',dict(status='running',phase='training')):
            self.assertEqual(worker.progress_snapshot()['progress'], dict(optimizer_step=123,optimizer_steps=150))


if __name__ == '__main__':
    unittest.main()
