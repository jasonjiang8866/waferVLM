import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runpod import dinov3_controller as c


class DiagnosticsTests(unittest.TestCase):
    def test_known_guard_reason_is_reported_but_remote_message_is_redacted(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'OUT', Path(tmp)):
            for message in ('No 4090 capacity meeting CUDA minimum', 'secret-token-in-remote-response'):
                with self.assertRaises(ValueError), c.preflight_stage('capacity'):
                    raise ValueError(message)
                record = json.loads((Path(tmp) / 'preflight-progress.json').read_text())
                self.assertEqual(record['stage'], 'capacity')
                self.assertEqual(record.get('reason'), message if message in c.SAFE_PREFLIGHT_REASONS else None)
                self.assertNotIn('secret-token', json.dumps(record))

    def test_observation_cannot_mutate_provider_even_on_failure(self):
        with patch.dict(os.environ, RUNPOD_API_KEY='fake'), patch.object(c.Runpod, 'call') as remote:
            api = c.ReadOnlyRunpod()
            for method in ('POST', 'DELETE', 'PATCH', 'PUT'):
                with self.assertRaisesRegex(ValueError, 'mutation forbidden'):
                    api.call('/pods', method)
            with self.assertRaises(ValueError):
                api.call('/pods', body={'anything': True})
            remote.assert_not_called()
            api.call('/pods')
            remote.assert_called_once_with('/pods', 'GET', None)

    def test_observation_has_no_paid_authorization_or_launch(self):
        env = dict(RUNPOD_API_KEY='fake', GH_TOKEN='fake', GITHUB_REPOSITORY=c.REPOSITORY,
                   GITHUB_RUN_ID='7', GITHUB_EVENT_NAME='workflow_dispatch', GITHUB_REF='refs/heads/main',
                   DIAGNOSTIC_IMAGE_SOURCE_SHA='a'*40, CONFIRM_PAID='false')
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'OUT', Path(tmp)), patch.dict(os.environ, env), \
                patch.object(c, 'git', return_value='b'*40), patch.object(c, 'manual_guard') as guard, \
                patch.object(c, 'launch') as launch, patch.object(c, 'cleanup') as cleanup, \
                patch.object(c, 'preflight_checks', return_value={}) as checks:
            api = c.ReadOnlyRunpod()
            result = c.observe(api)
            checks.assert_called_once_with(api, 'a'*40, 'fake')
            self.assertFalse(result['paid_compute_started'])
            self.assertEqual(result['controller_sha'], 'b'*40)
            guard.assert_not_called()
            launch.assert_not_called()
            cleanup.assert_not_called()
