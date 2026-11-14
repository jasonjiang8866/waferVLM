"""Exercise the full transport-to-independent-verifier-to-cleanup path offline."""
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_confirmation_archive as fixture_module
from runpod import vlm_controller as controller
from runpod.tests.test_confirmation_contract import specification


class ArchiveLifecycle(unittest.TestCase):
    def test_retrieved_training_verified_and_corruption_still_cleans(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as tmp:
                spec = specification()
                spec.update(source_sha=controller.SOURCE, controller_sha='a'*40, image=controller.IMAGE)
                fixture = fixture_module.TrainingArchive()
                with patch.object(fixture_module, 'specification', return_value=spec):
                    fixture.setUp()
                try:
                    if corrupt:
                        (fixture.training/'step200/checkpoints/checkpoint-200/optimizer.pt').write_text('corrupt')
                    archive = fixture.pack()
                    out = Path(tmp)
                    evidence = dict(controller_sha='a'*40, config_sha256=fixture.plan['config_sha256'],
                        gpu=spec['gpu'], regions=['TEST-1'], cost=controller.quote(spec['gpu_hourly_usd']),
                        ledger=spec['ledger'])
                    pod = dict(id='pod1', name=controller.AUTHORIZATION+'-vlm42-1234',
                        env={controller.OWNER:controller.REPOSITORY, 'WAFER_RUN_ID':'1234',
                             'WAFER_AUTHORIZATION':controller.AUTHORIZATION, 'WAFER_EXPERIMENT':'vlm-seed42'},
                        gpu={'id':spec['gpu']}, dataCenterId='TEST-1', cost=.89)
                    api = Mock()
                    api.call.side_effect = lambda path, method='GET', body=None: {'id':'pod1'} if method=='POST' else pod
                    api.listing.side_effect = [[pod], [], [], []]
                    result = dict(status='accepted', profiles=['seed-confirmation'], seed=42,
                        source_sha=spec['source_sha'], controller_sha=spec['controller_sha'],
                        config_sha256=evidence['config_sha256'], archive={'size':archive.stat().st_size,'sha256':'b'*64})
                    with patch.dict(os.environ, GITHUB_REPOSITORY=controller.REPOSITORY, GITHUB_RUN_ID='1234'), \
                         patch.object(controller, 'OUT', out), patch.object(controller, 'authorization_guard'), \
                         patch.object(controller, 'preflight', return_value=evidence), \
                         patch.object(controller, 'request', return_value=result), \
                         patch.object(controller, 'download', side_effect=lambda *a:shutil.copyfile(archive,out/'experiment.tar.gz')), \
                         patch('runpod.vlm_ledger.dataset_metadata', return_value=fixture.data), \
                         patch.object(controller.time, 'sleep'), patch('builtins.print'):
                        if corrupt:
                            with self.assertRaises(ValueError): controller.launch(api)
                        else:
                            controller.launch(api)
                    self.assertEqual(json.loads((out/'cleanup.json').read_text())['status'],'verified_absent')
                    self.assertEqual((out/'acceptance.json').exists(), not corrupt)
                    if not corrupt:
                        accepted=json.loads((out/'acceptance.json').read_text())
                        self.assertEqual([x['step'] for x in accepted['endpoints']],[200,250])
                        self.assertFalse(accepted['quality_accepted'])
                    self.assertEqual(sum(x.args[1:2]==('POST',) for x in api.call.call_args_list),1)
                    self.assertIn(unittest.mock.call('/pods/pod1','DELETE'),api.call.call_args_list)
                finally:
                    fixture.doCleanups()


if __name__ == '__main__':
    unittest.main()
