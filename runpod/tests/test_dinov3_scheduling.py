import copy
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from runpod import dinov3_controller as c, dinov3_scheduling as s


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.environ = patch.dict(os.environ, PARALLEL_VLM_RUN_ID='', REPLACE_QUEUED_RUN_ID='')
        self.environ.start()
        self.addCleanup(self.environ.stop)
        self.vlm_run = dict(id=int(s.VLM_RUN), head_sha=s.VLM_SHA, head_branch='main',
                            event='workflow_dispatch', path='.github/workflows/vlm-seed42.yml',
                            run_attempt=1, status='in_progress', conclusion=None)
        self.pod = dict(id='vlmpod', name=s.VLM_AUTH+'-vlm42-'+s.VLM_RUN,
                        status='RUNNING', cost=.89, mounts=[], env={
                            'WAFER_MANAGED_BY': s.REPOSITORY, 'WAFER_RUN_ID': s.VLM_RUN,
                            'WAFER_AUTHORIZATION': s.VLM_AUTH, 'WAFER_EXPERIMENT': 'vlm-seed42',
                            'WAFER_CONTROLLER_SHA': s.VLM_SHA, 'WAFER_RESULT_TOKEN': 'do-not-emit'})
        self.previous = dict(id=int(s.QUEUED_DINO_RUN), head_sha=s.QUEUED_DINO_SHA,
                             head_branch='main', event='workflow_dispatch', run_attempt=1,
                             path='.github/workflows/dinov3-gpu.yml', status='completed', conclusion='cancelled')

    def test_default_still_blocks_existing_project_and_orphan_name(self):
        self.assertEqual(s.companion([], 'fake')['account_reserve_usd'], 0)
        for pod in [self.pod, dict(name='wafervlm-dinov3-orphan', env={})]:
            with self.assertRaises(ValueError), patch.object(s, 'request') as read:
                s.companion([pod], 'fake')
            read.assert_not_called()

    def test_opt_in_checks_live_run_and_reserves_full_companion_budget(self):
        os.environ['PARALLEL_VLM_RUN_ID'] = s.VLM_RUN
        with patch.object(s, 'request', return_value=self.vlm_run) as read:
            result = s.companion([self.pod], 'fake')
        self.assertEqual(result, dict(run_id=s.VLM_RUN, active_pod=True, account_reserve_usd=30))
        self.assertNotIn('do-not-emit', str(result))
        read.assert_called_once_with(s.BASE+'/actions/runs/'+s.VLM_RUN, token='fake')

    def test_unknown_run_moved_source_wrong_workflow_or_rerun_rejected(self):
        os.environ['PARALLEL_VLM_RUN_ID'] = '999'
        with self.assertRaises(ValueError), patch.object(s, 'request') as read:
            s.companion([], 'fake')
        read.assert_not_called()
        os.environ['PARALLEL_VLM_RUN_ID'] = s.VLM_RUN
        for key, value in [('head_sha','a'*40), ('path','wrong.yml'), ('run_attempt',2),
                           ('status','queued'), ('head_branch','other'), ('event','pull_request')]:
            with self.subTest(key=key), patch.object(s, 'request', return_value={**self.vlm_run,key:value}):
                with self.assertRaises(ValueError):
                    s.companion([self.pod], 'fake')

    def test_second_pod_bad_identity_mount_rate_or_finished_residue_rejected(self):
        os.environ['PARALLEL_VLM_RUN_ID'] = s.VLM_RUN
        variants = []
        for key, value in [('name','wafervlm-dinov3-duplicate'), ('status','EXITED'),
                           ('mounts',['volume']), ('cost',True), ('cost',1.21), ('id','bad/id')]:
            variants.append({**self.pod,key:value})
        for key in self.pod['env']:
            if key == 'WAFER_RESULT_TOKEN':
                continue
            altered = copy.deepcopy(self.pod)
            altered['env'][key] = 'wrong'
            variants.append(altered)
        with patch.object(s, 'request', return_value=self.vlm_run):
            for pods in [[self.pod,self.pod], *[[p] for p in variants]]:
                with self.subTest(pods=len(pods)), self.assertRaises(ValueError):
                    s.companion(pods, 'fake')
        with patch.object(s, 'request', return_value={**self.vlm_run,'status':'completed'}):
            with self.assertRaises(ValueError):
                s.companion([self.pod], 'fake')
            self.assertEqual(s.companion([], 'fake')['account_reserve_usd'], 30)

    def test_account_guard_reserves_dino_plus_companion(self):
        api = Mock()
        api.listing.return_value = [self.pod]
        api.call.return_value = dict(price={'secure':.89},dataCenters=[{'id':'EU-RO-1','availability':'HIGH'}])
        os.environ['PARALLEL_VLM_RUN_ID'] = s.VLM_RUN
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,'OUT',Path(tmp)), \
                patch.object(c,'manual_guard',return_value=('a'*40,'fake')), \
                patch.object(c,'weight_report',return_value={}), patch.object(c,'cache_files'), \
                patch.object(c,'image_provenance'), patch.object(c,'budget',return_value={}) as budget, \
                patch.object(s,'request',return_value=self.vlm_run):
            result = c.preflight(api)
        budget.assert_called_once_with(api,reserve=33.0)
        self.assertEqual(result['cost']['total_cap_usd'],3)
        self.assertEqual(result['cost']['approved_total_usd'],10)
        self.assertEqual(result['companion']['run_id'],s.VLM_RUN)

    def test_dino_cleanup_does_not_delete_companion(self):
        api = Mock()
        api.listing.return_value = [self.pod]
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,'OUT',Path(tmp)), \
                patch.object(c,'identity',return_value='123'), patch.object(c.time,'sleep'):
            c.cleanup(api)
        api.call.assert_not_called()
        self.assertEqual(api.listing.call_count,12)

    def test_exact_cancelled_zero_job_run_can_be_replaced(self):
        os.environ['REPLACE_QUEUED_RUN_ID'] = s.QUEUED_DINO_RUN
        history = dict(total_count=3,workflow_runs=[{'id':999},self.previous,{'id':int(s.CPU_FAILED_RUN)}])
        with patch.object(s,'request',side_effect=[self.previous,{'total_count':0,'jobs':[]},
                                                  {'total_count':0,'artifacts':[]}]) as read, \
                patch.object(s,'reconciled_cpu_failure') as cpu:
            s.dispatch_history(history,'999','fake')
        cpu.assert_called_once_with('fake')
        self.assertEqual(read.call_count,3)
        self.assertIn('filter=all',read.call_args_list[1].args[0])
        self.assertTrue(all('method' not in x.kwargs for x in read.call_args_list))

    def test_pending_running_rerun_or_wrong_source_is_never_a_free_replacement(self):
        for key,value in [('status','pending'),('conclusion','success'),('run_attempt',2),
                          ('head_sha','b'*40),('id',5),('path','other.yml')]:
            with self.subTest(key=key), patch.object(s,'request') as read:
                with self.assertRaises(ValueError):
                    s.cancelled_before_jobs({**self.previous,key:value},'fake')
            read.assert_not_called()

    def test_any_job_or_artifact_blocks_zero_spend_claim(self):
        cases = [({'total_count':1,'jobs':[{'id':1}]},None),
                 ({'total_count':0,'jobs':[]},{'total_count':1,'artifacts':[{'id':1}]}),
                 ({'jobs':[]},None)]
        for jobs,artifacts in cases:
            with patch.object(s,'request',side_effect=[jobs,artifacts]), self.assertRaises(ValueError):
                s.cancelled_before_jobs(self.previous,'fake')

    def test_duplicate_history_or_unapproved_replacement_still_blocks(self):
        os.environ['REPLACE_QUEUED_RUN_ID'] = s.QUEUED_DINO_RUN
        for rows in [[{'id':999}], [{'id':999},{'id':888}],
                     [{'id':999},self.previous,{'id':888}]]:
            with self.assertRaises(ValueError), patch.object(s,'request') as read:
                s.dispatch_history(dict(total_count=len(rows),workflow_runs=rows),'999','fake')
            read.assert_not_called()


class CpuFailureReceiptTests(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads(Path(s.__file__).with_name('dinov3-precreate.json').read_text())
        p = self.receipt
        self.run = dict(id=int(s.CPU_FAILED_RUN),head_sha=p['source_sha'],head_branch='main',
                        event='workflow_dispatch',path='.github/workflows/dinov3-gpu.yml',
                        run_attempt=1,status='completed',conclusion='failure')
        self.job = dict(id=p['job_id'],status='completed',conclusion='failure',
                        steps=[dict(name=n,status='completed',conclusion=c) for n,c in p['required_steps'].items()])
        self.metadata = {}
        self.payload = {}
        for kind in ('result','independent'):
            r = p[kind]
            self.metadata[r['artifact_id']] = dict(id=r['artifact_id'],name=r['name'],expired=False,
                digest='sha256:'+r['sha256'],workflow_run=dict(id=int(r['run_id']),head_sha=r['source_sha']))
            self.payload[r['artifact_id']] = {f:(r['cleanup'] if f=='cleanup.json' else {}) for f in r['files']}

    def read(self,url,**kwargs):
        if '/artifacts/' in url:
            return self.metadata[int(url.rsplit('/',1)[1])]
        if '/jobs?' in url:
            return dict(total_count=1,jobs=[self.job])
        return self.run

    def archive(self,artifact_id,sha,limit):
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:
            for name,data in self.payload[artifact_id].items():
                z.writestr(name,json.dumps(data))
        return out.getvalue()

    def check(self):
        with patch.object(s,'request',side_effect=self.read),patch.object(s,'artifact_bytes',side_effect=self.archive):
            s.reconciled_cpu_failure('fake')

    def test_exact_cpu_failure_with_two_cleanup_receipts_is_reconciled(self):
        self.check()

    def test_rerun_or_paid_step_cannot_inherit_zero_spend(self):
        self.run['run_attempt']=2
        with self.assertRaises(ValueError): self.check()
        self.run['run_attempt']=1
        for step in self.job['steps']:
            if step['name'].startswith('Run six trials'):
                step['conclusion']='failure'
        with self.assertRaises(ValueError): self.check()

    def test_changed_receipt_or_cleanup_is_rejected(self):
        artifact=self.receipt['independent']['artifact_id']
        self.metadata[artifact]['digest']='sha256:'+'0'*64
        with self.assertRaises(ValueError): self.check()
        self.metadata[artifact]['digest']='sha256:'+self.receipt['independent']['sha256']
        self.payload[artifact]['cleanup.json']={**self.receipt['independent']['cleanup'],'checks':1}
        with self.assertRaises(ValueError): self.check()

    def test_resource_intent_or_extra_job_is_rejected(self):
        artifact=self.receipt['result']['artifact_id']
        self.payload[artifact]['provision.json']={'pod_id':'unexpected'}
        with self.assertRaises(ValueError): self.check()
        del self.payload[artifact]['provision.json']
        self.job['id']+=1
        with self.assertRaises(ValueError): self.check()


if __name__ == '__main__':
    unittest.main()
