import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from runpod import controller as c, worker as w


class RetryTests(unittest.TestCase):
    def test_aggregate_runtime_and_spend_include_prior_attempt(self):
        for rate in c.GPU_CAPS.values():
            quote = c.quote(rate)
            self.assertEqual(quote['planned_total_seconds'], 490 * 60)
            self.assertLessEqual(quote['planned_total_seconds'], 1100 * 60)
            self.assertGreater(quote['planned_total_usd'], 5.10)
            self.assertLessEqual(quote['planned_total_usd'], 30)
        with patch.object(c, 'PRIOR_RUNTIME_RESERVE_SECONDS', 1080 * 60):
            self.assertEqual(c.quote(.69)['work_seconds'], 600)
        with patch.object(c, 'PRIOR_RUNTIME_RESERVE_SECONDS', 1081 * 60):
            with self.assertRaises(ValueError):
                c.quote(.69)
        with patch.object(c, 'PRIOR_SPEND_RESERVE', 29.9):
            with self.assertRaises(ValueError):
                c.quote(.69)

    def test_reconciled_attempt_requires_exact_artifact(self):
        env = {'GITHUB_REPOSITORY': c.REPOSITORY, 'GITHUB_RUN_ID': '1234',
               'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_EVENT_NAME': 'workflow_dispatch',
               'GITHUB_REF': 'refs/heads/main', 'CONFIRM_PAID': 'true',
               'EXPECTED_SHA': 'a'*40, 'GH_TOKEN': 'unit-test'}
        for artifact_id, passes in [(13564060530, True), (13564060531, False)]:
            artifacts = [{'name': c.MARKER, 'workflow_run': {'id': 1234}},
                         {'name': c.MARKER, 'workflow_run': {'id': 42346201550}, 'id': artifact_id}]
            with patch.dict(os.environ, env), patch.object(c, 'git', return_value='a'*40), \
                 patch.object(c.time, 'time', return_value=1794036179), \
                 patch.object(c, 'request', side_effect=[{'object': {'sha': 'a'*40}}, {'artifacts': artifacts}]):
                if passes:
                    c.authorization_guard()
                else:
                    with self.assertRaises(RuntimeError):
                        c.authorization_guard()

    def test_training_progress_is_numeric_and_not_eval_progress(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(w, 'OUT', Path(directory)), \
             patch.object(w, 'RESULT', {'status': 'running', 'phase': 'baseline'}):
            log = Path(directory) / 'baseline.log'
            log.write_text('private diagnostic\r 50/100 [00:25]\r 74/100 [00:40]\r 160/160 [00:05]')
            snapshot = w.progress_snapshot()
            self.assertEqual(snapshot['progress'], {'optimizer_step': 74, 'optimizer_steps': 100})
            self.assertNotIn('private', str(snapshot))
            log.write_text('100/1000 [x] 101/100 [x] 160/160 [x]')
            self.assertNotIn('progress', w.progress_snapshot())
            log.unlink()
            self.assertNotIn('progress', w.progress_snapshot())

    def test_stage_budget_fits_existing_image_cli(self):
        self.assertEqual(w.PROFILE_WORK_SECONDS, 1200)
        self.assertGreater(c.MAX_WORK_SECONDS, w.PROFILE_WORK_SECONDS + 1200 + 120 + 120)
