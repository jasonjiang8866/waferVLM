import io
import json
import os
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

from runpod import vlm_repeat_startup_recovery as r, vlm_ledger as ledger, vlm_controller as c
from runpod.vlm_repeat_resume import finish_pin


def zipped(rows):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as z:
        for name, value in rows.items(): z.writestr(name, json.dumps(value))
    return out.getvalue()


class StartupRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.usd, self.seconds, elapsed = 9.528627960290688, 44477.96864977837, 1258.0931179523468
        self.name = f"wafer-20261107-usd20-vlmrepeats-{r.PIN['run_id']}"
        self.run = dict(head_sha=r.PIN['controller_sha'], run_attempt=1, status='completed',
            conclusion='failure', event='workflow_dispatch', head_branch='main', path='.github/workflows/vlm-repeats.yml')
        self.independent_run = dict(self.run, conclusion='success', event='workflow_run',
                                    path='.github/workflows/vlm-repeats-cleanup.yml')
        self.rows = {n: {} for n in ('dataset-metadata/dataset.json', 'input-verification.json', 'ledger.json')}
        cost = dict(gpu_hourly_usd=2.69, disk_hourly_usd=c.DISK_HOURLY,
                    prior_spend_reserve_usd=self.usd, prior_runtime_reserve_seconds=self.seconds,
                    work_seconds=2255, cleanup_seconds=600)
        self.clean = dict(status='verified_absent', name=self.name, creation_confirmed=True,
            consecutive_absent_checks=3, within_budget=True, within_runtime=True, persistent_volumes_created=False,
            checked_epoch=1000+elapsed, elapsed_billing_bound_s=elapsed,
            estimated_total_usd=self.usd+elapsed/3600*(2.69+c.DISK_HOURLY),
            estimated_total_seconds=self.seconds+elapsed)
        self.rows.update({
            'cleanup.json': self.clean,
            'allocation.json': dict(id=r.POD, gpu=r.GPU, data_center=r.REGION, observed_hourly_usd=2.69),
            'provision.json': dict(pod_id=r.POD, name=self.name, run_id=str(r.PIN['run_id']), started_epoch=1000, cost=cost),
            'launch-spec.json': dict(phase='validation-repeats', gpu=r.GPU, controller_sha=r.PIN['controller_sha'],
                resume_source_run_id=finish_pin()['run_id'], ledger=dict(usd=self.usd, seconds=self.seconds)),
            'preflight.json': dict(status='preflight_passed', gpu=r.GPU, regions=[r.REGION]),
        })
        for n in ('resume-verification.json', 'finish-verification.json', 'capacity-recovery.json', 'finish-capacity-recovery.json'):
            self.rows[n] = {'verified': n}
            (self.root/n).write_text(json.dumps(self.rows[n]))
        self.independent = dict(name=self.name, status='verified_absent', consecutive_absent_checks=3,
                                checked_epoch=self.clean['checked_epoch']+40)

    def verify(self):
        p = r.PIN
        values = {p['result_artifact_id']: (p['result_sha256'], zipped(self.rows)),
            p['independent_cleanup_artifact_id']: (p['independent_cleanup_sha256'], zipped({'cleanup.json': self.independent})),
            p['marker_artifact_id']: (p['marker_sha256'], b'marker')}
        def artifact(i, sha, limit):
            expected, raw = values[i]; self.assertEqual(expected, sha); self.assertLess(len(raw), limit); return raw
        with patch.dict(os.environ, GH_TOKEN='test-only'), \
             patch.object(ledger, 'request', side_effect=[self.run, self.independent_run]), \
             patch.object(ledger, 'artifact_bytes', side_effect=artifact), patch.object(ledger, 'verify_marker') as marker:
            result = r.reconcile(self.root, self.usd, self.seconds)
            marker.assert_called_once_with(b'marker', p['run_id'], True, p['controller_sha'], 'repeat')
            return result

    def test_paid_allocation_preserves_cost_and_failed_region(self):
        p, usd, seconds, elapsed, rejected = self.verify()
        self.assertEqual(p, r.PIN)
        self.assertAlmostEqual(usd, 10.471823366279366)
        self.assertGreater(usd, self.usd)
        self.assertAlmostEqual(seconds-self.seconds, elapsed)
        self.assertEqual(rejected['region'], r.REGION)
        self.assertEqual(rejected['gpu'], r.GPU)
        self.assertEqual(rejected['retry_after_epoch'], self.independent['checked_epoch']+1800)

    def test_rerun_running_or_changed_source_cannot_be_reused(self):
        for field, value in [('run_attempt', 2), ('status', 'in_progress'), ('head_sha', 'bad')]:
            old = self.run[field]; self.run[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'identity'): self.verify()
            self.run[field] = old

    def test_wrong_allocation_and_claimed_free_attempt_rejected(self):
        self.rows['allocation.json']['id'] = 'other'
        with self.assertRaises(RuntimeError): self.verify()
        self.rows['allocation.json']['id'] = r.POD
        self.clean['creation_confirmed'] = False
        with self.assertRaises(RuntimeError): self.verify()

    def test_changed_cost_chain_and_completed_output_rejected(self):
        self.clean['estimated_total_usd'] -= .01
        with self.assertRaisesRegex(RuntimeError, 'cumulative'): self.verify()
        self.clean['estimated_total_usd'] += .01
        self.rows['result.json'] = {'status': 'accepted'}
        with self.assertRaisesRegex(RuntimeError, 'Unexpected'): self.verify()

    def test_independent_cleanup_must_be_complete_and_later(self):
        self.independent['consecutive_absent_checks'] = 2
        with self.assertRaisesRegex(RuntimeError, 'Independent'): self.verify()
        self.independent['consecutive_absent_checks'] = 3
        self.independent['checked_epoch'] = 1
        with self.assertRaisesRegex(RuntimeError, 'Independent'): self.verify()

    def test_earlier_receipt_cannot_be_substituted(self):
        self.rows['finish-verification.json'] = {'modified': True}
        with self.assertRaises(RuntimeError): self.verify()

    def test_window_reallocated_only_inside_existing_total_caps(self):
        env = dict(WAFER_VLM_PHASE='repeat', WAFER_VLM_RESUME='1', WAFER_VLM_RESUME_RETRY='1',
                   WAFER_VLM_FINISH='1', WAFER_VLM_STARTUP_RETRY='1')
        data = dict(usd=10.471823366279366, seconds=45736.06176773072,
            resume_source_run_id=finish_pin()['run_id'], resume_capacity_run_id=c.CAPACITY_PIN['run_id'],
            finish_capacity_run_id=c.FINISH_CAPACITY_PIN['run_id'], finish_capacity_seconds=144.38420176506042,
            startup_failure_run_id=r.PIN['run_id'], startup_failure_seconds=1258.0931179523468)
        with patch.dict(os.environ, env):
            quote = c.quote(2.69, data)
            self.assertEqual((quote['work_seconds'], quote['cleanup_seconds']), (3600, 600))
            self.assertLess(quote['planned_total_usd'], 13.63)
            self.assertLess(quote['planned_total_seconds'], 66000)
            self.assertEqual(quote['approved_usd'], 30)
            for change in ({'usd': 29}, {'seconds': 64000}, {'startup_failure_run_id': 0}, {'startup_failure_seconds': 0}):
                with self.subTest(change=change), self.assertRaises(ValueError): c.quote(2.69, dict(data, **change))
            with patch.dict(os.environ, WAFER_VLM_PHASE='training'), self.assertRaises(ValueError): c.quote(.89, data)


class ArtifactRetries(unittest.TestCase):
    def test_transient_read_is_retried_with_same_integrity_arguments(self):
        error = urllib.error.HTTPError('https://example.invalid', 503, 'private-error', {}, None)
        with patch.object(ledger, '_artifact_bytes_once', side_effect=[error, b'verified']) as read, \
             patch.object(ledger.time, 'sleep') as sleep:
            self.assertEqual(ledger.artifact_bytes(1, 'a'*64, 100), b'verified')
        self.assertEqual(read.call_count, 2)
        self.assertTrue(all(call.args == (1, 'a'*64, 100) for call in read.call_args_list))
        sleep.assert_called_once_with(2)

    def test_attempts_are_bounded_and_never_expose_error_body(self):
        error = urllib.error.HTTPError('https://secret-url.invalid', 503, 'private-error', {}, None)
        with patch.object(ledger, '_artifact_bytes_once', side_effect=error) as read, \
             patch.object(ledger.time, 'sleep') as sleep, \
             self.assertRaisesRegex(RuntimeError, '^Artifact GET unavailable after three bounded attempts$'):
            ledger.artifact_bytes(1, 'a'*64, 100)
        self.assertEqual(read.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_auth_error_and_corrupt_artifact_are_not_retried(self):
        for error in (urllib.error.HTTPError('https://example.invalid', 403, 'private', {}, None),
                      RuntimeError('Artifact size or SHA256 mismatch')):
            with patch.object(ledger, '_artifact_bytes_once', side_effect=error) as read, \
                 patch.object(ledger.time, 'sleep') as sleep, self.assertRaises(RuntimeError):
                ledger.artifact_bytes(1, 'a'*64, 100)
            self.assertEqual(read.call_count, 1)
            sleep.assert_not_called()


if __name__ == '__main__': unittest.main()
