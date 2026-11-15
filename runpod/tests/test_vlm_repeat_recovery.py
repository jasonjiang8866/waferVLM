import copy
import io
import json
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from runpod import vlm_ledger as ledger


def zipped(rows):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        for name,value in rows.items():z.writestr(name,value)
    return stream.getvalue()


class RepeatRecovery(unittest.TestCase):
    def setUp(self):
        self.pin=copy.deepcopy(ledger.REPEAT_ATTEMPTS[0])
        p=self.pin
        self.run=dict(head_sha=p['controller_sha'],run_attempt=1,status='completed',conclusion='failure',
                      event='workflow_dispatch',head_branch='main',path='.github/workflows/vlm-repeats.yml')
        self.rows={name+'.json':json.dumps(p[name.replace('-','_')])
                   for name in ('cleanup','create-error','provision','launch-spec')}
        for name in ('dataset-metadata/dataset.json','input-verification.json','ledger.json','preflight.json'):
            self.rows[name]='{}'
        self.independent=p['independent_cleanup']
        self.before=p['launch_spec']['ledger']

    def verify(self):
        p=self.pin
        def artifact(artifact_id,*args):
            if artifact_id==p['result_artifact_id']:return zipped(self.rows)
            if artifact_id==p['independent_cleanup_artifact_id']:
                return zipped({'cleanup.json':json.dumps(self.independent)})
            if artifact_id==p['marker_artifact_id']:
                return zipped({'run-id.txt':str(p['run_id'])+'\n','intent.json':json.dumps(dict(
                    phase='validation-repeats',workload='vlm-repeats',run_id=str(p['run_id']),controller_sha=p['controller_sha']))})
            raise AssertionError('Unexpected artifact')
        with tempfile.TemporaryDirectory() as tmp,patch.object(ledger,'REPEAT_ATTEMPTS',[p]), \
             patch.object(ledger,'request',return_value=self.run),patch.object(ledger,'artifact_bytes',side_effect=artifact), \
             patch.dict(ledger.os.environ,GH_TOKEN='test-only'):
            return ledger.repeat_failures(tmp,self.before['usd'],self.before['seconds'])

    def test_exact_failed_create_retains_conservative_cost(self):
        rows,usd,seconds=self.verify()
        self.assertEqual(rows,[self.pin])
        self.assertEqual(usd,8.16852580926767)
        self.assertEqual(seconds,38587.4081276989)

    def test_rerun_changed_receipt_or_incomplete_cleanup_blocks(self):
        self.run['run_attempt']=2
        with self.assertRaises(RuntimeError):self.verify()
        self.run['run_attempt']=1
        self.rows['allocation.json']='{}'
        with self.assertRaises(RuntimeError):self.verify()
        del self.rows['allocation.json']
        self.independent=dict(self.independent,consecutive_absent_checks=2)
        with self.assertRaises(RuntimeError):self.verify()

    def test_changed_budget_chain_and_understated_cost_block(self):
        self.before=dict(self.before,usd=self.before['usd']-.01)
        with self.assertRaises(RuntimeError):self.verify()
        self.before=self.pin['launch_spec']['ledger']
        self.pin['cleanup']['estimated_total_usd']-=.01
        self.rows['cleanup.json']=json.dumps(self.pin['cleanup'])
        with self.assertRaisesRegex(RuntimeError,'cumulative consumption'):self.verify()


if __name__=='__main__':unittest.main()
