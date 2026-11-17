import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from runpod import vlm_repeat_finish_capacity as recovery, vlm_repeat_resume as resume, vlm_ledger as ledger


def zipped(rows):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        for name, value in rows.items(): z.writestr(name, json.dumps(value))
    return stream.getvalue()


class FinalCapacityRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        p = recovery.PIN
        self.prior_usd, self.prior_seconds = 9.472521519981784, 44333.58444801331
        name = f"wafer-20261107-usd20-vlmrepeats-{p['run_id']}"
        self.run = dict(head_sha=p['controller_sha'], run_attempt=1, status='completed',
                        conclusion='failure', event='workflow_dispatch', head_branch='main',
                        path='.github/workflows/vlm-repeats.yml')
        self.rows = {n: {} for n in ['dataset-metadata/dataset.json', 'input-verification.json', 'ledger.json']}
        cost = dict(prior_spend_reserve_usd=self.prior_usd, prior_runtime_reserve_seconds=self.prior_seconds,
                    gpu_hourly_usd=1.39, disk_hourly_usd=.009)
        self.clean = dict(name=name, status='verified_absent', creation_confirmed=False, consecutive_absent_checks=8,
            within_budget=True, within_runtime=True, persistent_volumes_created=False, checked_epoch=1144,
            elapsed_billing_bound_s=144, estimated_total_seconds=self.prior_seconds+144,
            estimated_total_usd=self.prior_usd+144/3600*1.399)
        self.rows.update({
            'cleanup.json': self.clean,
            'create-error.json': dict(status='create_not_confirmed', http_status=400, categories=['capacity']),
            'provision.json': dict(pod_id=None, name=name, run_id=str(p['run_id']), started_epoch=1000, cost=cost),
            'launch-spec.json': dict(controller_sha=p['controller_sha'], resume_source_run_id=999,
                                    gpu='NVIDIA GeForce RTX 5090', ledger=dict(usd=self.prior_usd, seconds=self.prior_seconds)),
            'preflight.json': dict(status='preflight_passed', gpu='NVIDIA GeForce RTX 5090', regions=['TEST-1'], cost=cost),
            'resume-verification.json': {'source': 'original14'},
            'finish-verification.json': {'source': 'verified16'},
            'capacity-recovery.json': {'source': 'prior-capacity-rejection'},
        })
        self.independent = dict(name=name, status='verified_absent', consecutive_absent_checks=3, checked_epoch=1164)
        for name in ['resume-verification.json', 'finish-verification.json', 'capacity-recovery.json']:
            (self.root/name).write_text(json.dumps(self.rows[name]))

    def verify(self):
        p = recovery.PIN
        def artifact(i, digest, limit):
            if i == p['result_artifact_id']:
                self.assertEqual(digest, p['result_sha256']); return zipped(self.rows)
            if i == p['independent_cleanup_artifact_id']: return zipped({'cleanup.json': self.independent})
            if i == p['marker_artifact_id']: return b'existing-marker-contract'
            raise AssertionError('Unexpected artifact')
        with patch.dict(ledger.os.environ, GH_TOKEN='test-only'), \
             patch.object(resume, 'RECEIPT', self.root/'resume-verification.json'), \
             patch.object(resume, 'FINISH_RECEIPT', self.root/'finish-verification.json'), \
             patch.object(resume, 'finish_pin', return_value={'run_id': 999}), \
             patch.object(ledger, 'request', return_value=self.run), \
             patch.object(ledger, 'artifact_bytes', side_effect=artifact), patch.object(ledger, 'verify_marker'):
            return recovery.reconcile(self.root, self.prior_usd, self.prior_seconds)

    def test_complete_chain_preserves_cost_and_derives_rejected_placement(self):
        p, usd, seconds, elapsed, rejected = self.verify()
        self.assertEqual(p, recovery.PIN)
        self.assertEqual((usd, seconds, elapsed), (self.clean['estimated_total_usd'], self.prior_seconds+144, 144))
        self.assertEqual(rejected, [dict(gpu='NVIDIA GeForce RTX 5090', region='TEST-1', retry_after_epoch=2944)])

    def test_unknown_create_error_allocation_and_incomplete_cleanup_block(self):
        self.rows['create-error.json']['categories'] = ['transport_or_response_error']
        with self.assertRaises(RuntimeError): self.verify()
        self.rows['create-error.json']['categories'] = ['capacity']
        self.rows['allocation.json'] = {}
        with self.assertRaises(RuntimeError): self.verify()
        del self.rows['allocation.json']
        self.independent['consecutive_absent_checks'] = 2
        with self.assertRaises(RuntimeError): self.verify()

    def test_changed_source_or_cumulative_cost_blocks(self):
        self.rows['finish-verification.json'] = {'source': 'not16'}
        with self.assertRaises(RuntimeError): self.verify()
        self.rows['finish-verification.json'] = {'source': 'verified16'}
        self.clean['estimated_total_usd'] -= .01
        with self.assertRaisesRegex(RuntimeError, 'cumulative'): self.verify()

    def test_rerun_or_untrusted_placement_cannot_unlock_retry(self):
        self.run['run_attempt'] = 2
        with self.assertRaisesRegex(RuntimeError, 'identity'): self.verify()
        self.run['run_attempt'] = 1
        for regions in ([], ['TEST-1', 'TEST-1'], [['TEST-1']], ['../invalid']):
            self.rows['preflight.json']['regions'] = regions
            with self.subTest(regions=regions), self.assertRaisesRegex(RuntimeError, 'placement'):
                self.verify()


if __name__ == '__main__': unittest.main()
