import json
import tempfile
import unittest
from pathlib import Path
from wafervlm.core import LABELS, PROMPT, sha256_file
from wafervlm.modeling import MODEL_ID, MODEL_REVISION
from wafervlm.resume import REQUIRED, check_resume, lower_constant_learning_rate


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

    def test_learning_rate_change_is_explicit_and_lower_only(self):
        cfg=dict(self.cfg,learning_rate=.00005)
        with self.assertRaisesRegex(ValueError,'learning_rate'):
            check_resume(self.cp,self.data,MODEL_REVISION,cfg)
        result=check_resume(self.cp,self.data,MODEL_REVISION,cfg,lower_learning_rate=True)
        self.assertEqual(result['parent_learning_rate'],.0002)
        self.assertTrue(result['lower_learning_rate'])
        for lr in [.0002,.001,0.,float('nan'),float('inf')]:
            with self.subTest(lr=lr),self.assertRaises(ValueError):
                check_resume(self.cp,self.data,MODEL_REVISION,dict(self.cfg,learning_rate=lr),lower_learning_rate=True)

    def test_lower_lr_preserves_adam_moments_and_scheduler_position(self):
        import copy
        import torch
        p=torch.nn.Parameter(torch.tensor([1.]))
        opt=torch.optim.AdamW([p],lr=.0002)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1.)
        for _ in range(3):
            p.grad=torch.ones_like(p);opt.step();sched.step()
        before=copy.deepcopy(opt.state[p]);position=(sched.last_epoch,sched._step_count)
        lower_constant_learning_rate(opt,sched,.0002,.00005)
        for key,value in before.items():torch.testing.assert_close(value,opt.state[p][key],rtol=0,atol=0)
        self.assertEqual(position,(sched.last_epoch,sched._step_count))
        self.assertEqual(opt.param_groups[0]['lr'],.00005)
        opt.step();sched.step()
        self.assertEqual(sched.get_last_lr(),[.00005])
        self.assertEqual(int(opt.state[p]['step']),4)

    def test_intervention_rejects_wrong_restored_lr_and_nonconstant_scheduler(self):
        import torch
        p=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([p],lr=.001)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1.)
        with self.assertRaises(ValueError):lower_constant_learning_rate(opt,sched,.0002,.00005)
        opt=torch.optim.AdamW([p],lr=.0002)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:.9**step)
        with self.assertRaises(ValueError):lower_constant_learning_rate(opt,sched,.0002,.00005)
