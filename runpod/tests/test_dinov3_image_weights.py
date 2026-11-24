"""Synthetic boundary tests; CI also loads the real model with networking off."""
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from contextlib import ExitStack
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from runpod import dinov3_image_weights as m, dinov3_worker as w, dinov3_controller as c


class EmbeddedWeightsTests(unittest.TestCase):
    def fixture(self, root, stack):
        for name, path in [('HF_ROOT', root/'hf'), ('LICENSE', root/'LICENSE.md'), ('METADATA', root/'metadata')]:
            stack.enter_context(patch.object(m, name, path))
        stack.enter_context(patch.dict(os.environ, HF_HOME=str(m.HF_ROOT)))
        weights = m.snapshot()
        weights.mkdir(parents=True)
        data = {'config.json': b'{}', 'model.safetensors': b'synthetic test model'}
        for name, content in data.items():
            (weights/name).write_bytes(content)
        stack.enter_context(patch.dict(m.BACKBONES, small=replace(m.BACKBONES['small'],
            config_sha256=m.digest(weights/'config.json'), weights_sha256=m.digest(weights/'model.safetensors'))))
        m.LICENSE.write_text('Synthetic license for tests only\n'*50)
        m.METADATA.mkdir()
        (m.METADATA/'manifest.json').write_text(json.dumps(
            dict(weights=m.report(), license_sha256=m.digest(m.LICENSE))))
        return weights

    def test_relocated_cache_matches_independent_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            self.fixture(Path(tmp), stack)
            self.assertTrue(m.verify_files(m.report())['verified'])
            with self.assertRaisesRegex(ValueError, 'controller CPU fingerprint'):
                m.verify_files({**m.report(), 'parameters': 1})

    def test_corrupted_missing_extra_weights_and_license_are_rejected(self):
        for change in ('corrupt', 'missing', 'extra', 'license', 'manifest', 'symlink'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                weights = self.fixture(Path(tmp), stack)
                if change == 'corrupt':
                    (weights/'model.safetensors').write_bytes(b'changed')
                elif change == 'missing':
                    (weights/'model.safetensors').unlink()
                elif change == 'extra':
                    (weights/'token').write_text('test-only')
                elif change == 'license':
                    m.LICENSE.unlink()
                elif change == 'manifest':
                    (m.METADATA/'manifest.json').write_text('{}')
                else:
                    (weights/'config.json').unlink()
                    (weights/'config.json').symlink_to(m.LICENSE)
                with self.assertRaises(ValueError):
                    m.verify_files(m.report())

    def test_missing_embedded_weights_reject_before_cuda_or_training(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(w, 'OUT', Path(tmp)/'results'), \
                patch.object(w, 'ARCHIVE', Path(tmp)/'result.tar.gz'), patch.object(w, 'STATE', {}), \
                patch.dict(os.environ, WAFERVLM_SOURCE_SHA='a'*40, DINO_IMAGE='test', DINO_WEIGHTS_MODE='embedded'), \
                patch.object(w, 'verify_files', side_effect=ValueError('missing weights')), \
                patch.object(w, 'hardware_smoke') as hardware, patch.object(w.subprocess, 'Popen') as child:
            w.work()
            self.assertEqual(w.STATE['status'], 'rejected')
            self.assertEqual(w.STATE['phase'], 'verifying_embedded_weights')
            hardware.assert_not_called()
            child.assert_not_called()

    def test_embedded_worker_refuses_http_upload(self):
        with patch.dict(os.environ, DINO_WEIGHTS_MODE='embedded'), patch.object(w, 'TOKEN', 'test-only'):
            server = ThreadingHTTPServer(('127.0.0.1', 0), w.Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/weights/config.json',
                    method='PUT', data=b'{}', headers={'Authorization': 'Bearer test-only'})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request)
                self.assertEqual(caught.exception.code, 405)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_controller_rejects_unweighted_image_before_allocation(self):
        required = ['WAFERVLM_SOURCE_SHA='+'a'*40, 'DINO_WEIGHTS_MODE=embedded',
                    'HF_HOME=/opt/dinov3-hf', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1']
        for absent in [None, *required[1:]]:
            config = json.dumps({'config': {'Env': [value for value in required if value != absent]}}).encode()
            manifest = json.dumps({'config': {'digest': 'sha256:'+c.hashlib.sha256(config).hexdigest()}}).encode()
            image = 'ghcr.io/jasonjiang8866/wafervlm@sha256:'+c.hashlib.sha256(manifest).hexdigest()
            responses = [io.BytesIO(b'{"token":"test-only"}'), io.BytesIO(manifest), io.BytesIO(config)]
            with self.subTest(absent=absent), patch.object(c.urllib.request, 'urlopen', side_effect=responses):
                if absent is None:
                    c.image_provenance(image, 'a'*40)
                else:
                    with self.assertRaisesRegex(ValueError, 'Offline embedded-weight image required'):
                        c.image_provenance(image, 'a'*40)


if __name__ == '__main__':
    unittest.main()
