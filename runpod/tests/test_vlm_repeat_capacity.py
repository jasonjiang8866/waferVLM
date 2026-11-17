import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from runpod import vlm_repeat_capacity as capacity, vlm_ledger as ledger
from runpod.vlm_repeat_resume import PIN as source, RECEIPT


def zipped(rows):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        for name,value in rows.items():z.writestr(name,json.dumps(value))
    return out.getvalue()


class CapacityRecovery(unittest.TestCase):
    def setUp(self):
        p=capacity.PIN;name=f"wafer-20261107-usd20-vlmrepeats-{p['run_id']}"
        self.prior_usd=8.979243733782864;self.prior_seconds=41834.144988689426
        self.run=dict(head_sha=p['controller_sha'],run_attempt=1,status='completed',conclusion='failure',
                      event='workflow_dispatch',head_branch='main',path='.github/workflows/vlm-repeats.yml')
        self.clean=dict(name=name,status='verified_absent',creation_confirmed=False,consecutive_absent_checks=8,
            within_budget=True,within_runtime=True,persistent_volumes_created=False,checked_epoch=1144,
            elapsed_billing_bound_s=144,estimated_total_seconds=self.prior_seconds+144,
            estimated_total_usd=self.prior_usd+144/3600*.899)
        self.rows={n:{} for n in ['dataset-metadata/dataset.json','input-verification.json','ledger.json','preflight.json']}
        self.rows.update({'cleanup.json':self.clean,
            'create-error.json':dict(status='create_not_confirmed',http_status=400,categories=['capacity']),
            'provision.json':dict(pod_id=None,name=name,run_id=str(p['run_id']),started_epoch=1000,
                cost=dict(prior_spend_reserve_usd=self.prior_usd,prior_runtime_reserve_seconds=self.prior_seconds,
                          gpu_hourly_usd=.89,disk_hourly_usd=.009)),
            'launch-spec.json':dict(controller_sha=p['controller_sha'],resume_source_run_id=source['run_id'],
                                   ledger=dict(usd=self.prior_usd,seconds=self.prior_seconds)),
            'resume-verification.json':json.loads(RECEIPT.read_text())})
        self.independent=dict(name=name,status='verified_absent',consecutive_absent_checks=3,checked_epoch=1164)

    def verify(self):
        p=capacity.PIN
        def artifact(i,*args):
            if i==p['result_artifact_id']:return zipped(self.rows)
            if i==p['independent_cleanup_artifact_id']:return zipped({'cleanup.json':self.independent})
            if i==p['marker_artifact_id']:return b'marker-checked-by-existing-contract'
            raise AssertionError('Unexpected artifact')
        with tempfile.TemporaryDirectory() as tmp,patch.dict(capacity.os.environ,GH_TOKEN='test-only'), \
             patch.object(ledger,'request',return_value=self.run),patch.object(ledger,'artifact_bytes',side_effect=artifact), \
             patch.object(ledger,'verify_marker'):
            return capacity.reconcile(Path(tmp),self.prior_usd,self.prior_seconds)

    def test_exact_capacity_rejection_keeps_cost_and_elapsed(self):
        _,usd,seconds,elapsed=self.verify()
        self.assertEqual(usd,self.clean['estimated_total_usd']);self.assertEqual(elapsed,144)
        self.assertEqual(seconds,self.prior_seconds+144)

    def test_other_errors_allocations_or_incomplete_cleanup_block_retry(self):
        self.rows['create-error.json']['categories']=['auth']
        with self.assertRaises(RuntimeError):self.verify()
        self.rows['create-error.json']['categories']=['capacity']
        self.rows['allocation.json']={}
        with self.assertRaises(RuntimeError):self.verify()
        del self.rows['allocation.json']
        self.independent['consecutive_absent_checks']=2
        with self.assertRaises(RuntimeError):self.verify()

    def test_understated_cost_and_replayed_run_block_retry(self):
        self.clean['estimated_total_usd']-=.01
        with self.assertRaisesRegex(RuntimeError,'cumulative'):self.verify()
        self.run['run_attempt']=2
        with self.assertRaisesRegex(RuntimeError,'identity'):self.verify()
