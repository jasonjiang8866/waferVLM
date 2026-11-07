import json
import tempfile
import unittest
from pathlib import Path
from wafervlm.core import LABELS, PROMPT, sha256_file
from wafervlm.modeling import MODEL_ID, MODEL_REVISION
from wafervlm.resume import REQUIRED, check_resume


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);self.data=root/'data';self.data.mkdir()
        (self.data/'dataset.json').write_text('{}')
        self.train=root/'train';self.cp=self.train/'checkpoints/checkpoint-100';self.cp.mkdir(parents=True)
        self.cfg=dict(steps=150,seed=17,accumulation=8,rank=8,learning_rate=.0002,max_length=1024,device='cuda')
        for name in REQUIRED:(self.cp/name).write_text('{}')
        (self.cp/'trainer_state.json').write_text('{"global_step":100}')
        (self.train/'run.json').write_text(json.dumps({'model_id':MODEL_ID,'model_revision':MODEL_REVISION,
            'dataset_sha256':sha256_file(self.data/'dataset.json'),'labels':list(LABELS),'prompt':PROMPT,
            'config':dict(self.cfg,steps=100)}))
        self.seal()

    def seal(self):
        files=[self.train/'run.json',*self.cp.iterdir()]
        (self.train/'checksums.json').write_text(json.dumps({str(p.relative_to(self.train)):sha256_file(p) for p in files}))

    def test_complete_checkpoint_and_hyperparameters(self):
        self.assertEqual(check_resume(self.cp,self.data,MODEL_REVISION,self.cfg)['global_step'],100)
        for key,value in [('steps',100),('seed',18),('rank',4),('accumulation',1),('learning_rate',.001)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                check_resume(self.cp,self.data,MODEL_REVISION,dict(self.cfg,**{key:value}))

    def test_optimizer_corruption_rejected_before_deserialization(self):
        (self.cp/'optimizer.pt').write_text('corrupted')
        with self.assertRaisesRegex(ValueError,'checksum'):check_resume(self.cp,self.data,MODEL_REVISION,self.cfg)

    def test_missing_rng_rejected(self):
        (self.cp/'rng_state.pth').unlink()
        with self.assertRaisesRegex(ValueError,'Incomplete'):check_resume(self.cp,self.data,MODEL_REVISION,self.cfg)

    def test_data_change_rejected(self):
        (self.data/'dataset.json').write_text('{"different":true}')
        with self.assertRaisesRegex(ValueError,'data/model'):check_resume(self.cp,self.data,MODEL_REVISION,self.cfg)
