import copy
import os
import tempfile
import unittest
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
        history = dict(total_count=2,workflow_runs=[{'id':999},self.previous])
        with patch.object(s,'request',side_effect=[self.previous,{'total_count':0,'jobs':[]},
                                                  {'total_count':0,'artifacts':[]}]) as read:
            s.dispatch_history(history,'999','fake')
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


if __name__ == '__main__':
    unittest.main()
