import copy
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_confirmation as fixture_module
from wafervlm import confirmation as c
from wafervlm.core import sha256_file, write_json, write_jsonl
from wafervlm.experiment import seal_output
from runpod import vlm_repeat_verify as verifier
from runpod import vlm_controller as controller, vlm_repeat_inputs as inputs


class RepeatArchive(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture_module.ConfirmationTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root/'results';self.root.mkdir()
        self.repeats=self.root/'repeats';self.repeats.mkdir()
        records,reference=self.fixture.repeat_fixture()
        write_jsonl(self.repeats/'validation-reference.jsonl',reference)
        self.cfg=c.recipe();self.cfg['validation_sha256']=sha256_file(self.repeats/'validation-reference.jsonl')
        for r in records:
            src=Path(r['output']);dest=self.repeats/src.name;shutil.move(src,dest);r['output']=dest.name
            run=json.loads((dest/'run.json').read_text());run['validation_sha256']=self.cfg['validation_sha256'];run['gpu']='NVIDIA GeForce RTX 4090'
            write_json(dest/'run.json',run);seal_output(dest)
        self.records=records
        for obj in (c,verifier):
            p=patch.object(obj,'recipe',return_value=self.cfg);p.start();self.addCleanup(p.stop)
        self.frozen=self.fixture.root/'freeze.json'
        rows=[dict(seed=s,step=t,files={'adapter_model.safetensors':'weights'},run_sha256='run')
              for s in (17,42,123) for t in (200,250)]
        write_json(self.frozen,rows)
        p=patch.object(verifier,'FROZEN',self.frozen);p.start();self.addCleanup(p.stop)
        self.spec=dict(source_sha='a'*40,controller_sha='c'*40,image='image@sha256:'+'b'*64,
                       config_sha256='d'*64,input_manifest_sha256=sha256_file(self.frozen),gpu='NVIDIA GeForce RTX 4090')
        write_json(self.root/'worker.json',dict(status='accepted',profiles=['validation-repeats'],quality_accepted=False,
            **{k:self.spec[k] for k in ('source_sha','controller_sha','config_sha256','input_manifest_sha256')}))
        write_json(self.root/'hardware.json',dict(gpu=self.spec['gpu'],image_digest=self.spec['image'].split('@')[1]))
        acquisition=json.loads((c.ROOT/'docs/evidence/wm811k-lot17.json').read_text())
        write_json(self.root/'acquisition.json',acquisition)
        self.acquisition=self.fixture.root/'acquisition.json';write_json(self.acquisition,acquisition)
        p=patch.object(verifier,'ACQUISITION',self.acquisition);p.start();self.addCleanup(p.stop)
        write_json(self.repeats/'endpoints.json',rows)
        write_json(self.repeats/'workload.json',self.cfg)
        write_json(self.repeats/'status.json',dict(status='completed',passes=18,test_accessed=False,holdout_accessed=False))
        write_json(self.repeats/'passes.json',records)
        write_json(self.repeats/'summary.json',c.summarize_repeats(records,reference,base=self.repeats))

    def pack(self):
        seal_output(self.repeats)
        archive=self.fixture.root/'results.tar.gz'
        with tarfile.open(archive,'w:gz') as t:t.add(self.root,arcname='results')
        return archive

    def test_complete_package_recomputes_first_pass_not_best(self):
        got=verifier.verify_repeat_archive(self.pack(),self.spec)
        self.assertEqual(got['passes'],18)
        self.assertEqual(got['summary']['pairs'][0]['headline']['accuracy'],159/160)
        self.assertEqual(got['summary']['pairs'][0]['passes'][1]['score']['accuracy'],1)
        self.assertFalse(got['holdout_accessed'])

    def test_resealed_missing_pass_wrong_weights_and_changed_summary_fail(self):
        for change in ('missing','weights','summary'):
            with self.subTest(change=change):
                records=copy.deepcopy(self.records)
                if change=='missing':records.pop()
                if change=='weights':records[0]['adapter_sha256']='wrong'
                write_json(self.repeats/'passes.json',records)
                summary=json.loads((self.repeats/'summary.json').read_text())
                if change=='summary':
                    summary['pairs'][0]['headline']['accuracy']=1
                    write_json(self.repeats/'summary.json',summary)
                with self.assertRaises(ValueError):verifier.verify_repeat_archive(self.pack(),self.spec)

    def test_cross_run_source_or_holdout_output_rejected(self):
        bad=dict(self.spec,source_sha='e'*40)
        with self.assertRaises(ValueError):verifier.verify_repeat_archive(self.pack(),bad)
        (self.root/'holdout').mkdir()
        with self.assertRaises(ValueError):verifier.verify_repeat_archive(self.pack(),self.spec)

    def test_retrieval_acceptance_and_corruption_both_reach_owned_cleanup(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt),tempfile.TemporaryDirectory() as tmp:
                if corrupt:
                    path=self.repeats/self.records[0]['output']/'predictions.jsonl'
                    path.write_text('corrupt')
                archive=self.pack();out=Path(tmp)
                prior=dict(usd=8.132281676712218,seconds=38442.25880065918,attempts=[])
                with patch.dict(os.environ,WAFER_VLM_PHASE='repeat',GITHUB_REPOSITORY=controller.REPOSITORY,GITHUB_RUN_ID='1234'):
                    evidence=dict(controller_sha=self.spec['controller_sha'],config_sha256=self.spec['config_sha256'],
                        inputs=dict(manifest_sha256=self.spec['input_manifest_sha256']),gpu=self.spec['gpu'],
                        regions=['TEST-1'],cost=controller.quote(.89,prior),ledger=prior)
                    pod=dict(id='pod1',name=controller.AUTHORIZATION+'-vlmrepeats-1234',
                        env={controller.OWNER:controller.REPOSITORY,'WAFER_RUN_ID':'1234',
                             'WAFER_AUTHORIZATION':controller.AUTHORIZATION,'WAFER_EXPERIMENT':'vlm-repeats'},
                        gpu={'id':self.spec['gpu']},dataCenterId='TEST-1',cost=.89)
                    api=Mock();api.call.side_effect=lambda path,method='GET',body=None:{'id':'pod1'} if method=='POST' else pod
                    api.listing.side_effect=[[pod],[],[],[]]
                    result=json.loads((self.root/'worker.json').read_text())
                    result['archive']=dict(size=archive.stat().st_size,sha256=sha256_file(archive))
                    # This fake provider never executes the bootstrap. Worker images
                    # intentionally omit the controller's production input receipts.
                    with patch.object(controller,'OUT',out),patch.object(controller,'SOURCE',self.spec['source_sha']), \
                         patch.object(controller,'IMAGE',self.spec['image']),patch.object(controller,'authorization_guard'), \
                         patch.object(controller,'preflight',return_value=evidence), \
                         patch.object(controller,'worker_code',return_value='pass'), \
                         patch.object(controller,'request',return_value=result), \
                         patch.object(controller,'download',side_effect=lambda *a:shutil.copyfile(archive,out/'experiment.tar.gz')), \
                         patch.object(controller.time,'sleep'),patch('builtins.print'):
                        if corrupt:
                            with self.assertRaises(ValueError):controller.launch(api)
                        else:
                            controller.launch(api)
                    self.assertEqual(json.loads((out/'cleanup.json').read_text())['status'],'verified_absent')
                    self.assertEqual((out/'acceptance.json').exists(),not corrupt)
                    self.assertEqual(sum(x.args[1:2]==('POST',) for x in api.call.call_args_list),1)


if __name__ == '__main__':
    unittest.main()
