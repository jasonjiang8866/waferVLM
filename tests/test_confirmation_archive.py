import copy
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wafervlm import confirmation as c
from wafervlm.core import LABELS, PROMPT, write_json, sha256_file
from wafervlm.experiment import seal_output
from wafervlm.modeling import MODEL_ID, MODEL_REVISION
from wafervlm.resume import REQUIRED, check_resume
from runpod.confirmation_verify import verify_training_archive
from runpod.tests.test_confirmation_contract import specification


class TrainingArchive(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.data=self.root/'data';self.data.mkdir()
        write_json(self.data/'dataset.json',{})
        self.cfg=c.recipe();self.cfg['dataset_sha256']=sha256_file(self.data/'dataset.json')
        self.spec=specification();self.results=self.root/'results';self.results.mkdir()
        self.training=self.results/'training';self.training.mkdir()
        self.mock=patch.object(c,'recipe',return_value=self.cfg);self.mock.start();self.addCleanup(self.mock.stop)
        from runpod.confirmation_contract import launch_plan
        self.plan=launch_plan(self.spec)
        write_json(self.results/'worker.json',dict(status='accepted',profiles=['seed-confirmation'],
            source_sha=self.spec['source_sha'],controller_sha=self.spec['controller_sha'],
            config_sha256=self.plan['config_sha256'],seed=self.spec['seed']))
        write_json(self.results/'hardware.json',dict(gpu='fixture',image_digest=self.spec['image'].split('@')[1]))
        write_json(self.results/'acquisition.json',json.loads((c.ROOT/'docs/evidence/wm811k-lot17.json').read_text()))
        write_json(self.training/'workload.json',self.cfg)
        write_json(self.training/'status.json',dict(status='completed',seed=self.spec['seed'],end_step=250,
                   test_accessed=False,holdout_accessed=False))
        start=0;parent=None;endpoints=[]
        for end in self.cfg['stages']:
            stage=self.training/f'step{end}';cp=stage/f'checkpoints/checkpoint-{end}';cp.mkdir(parents=True)
            for name in REQUIRED:(cp/name).write_text('{}')
            write_json(cp/'trainer_state.json',{'global_step':end})
            config=c.training_config(self.spec['seed'],end)
            resume=check_resume(parent,self.data,MODEL_REVISION,config,lower_learning_rate=start==100) if parent else None
            write_json(stage/'run.json',dict(model_id=MODEL_ID,model_revision=MODEL_REVISION,
                dataset_sha256=self.cfg['dataset_sha256'],labels=list(LABELS),prompt=PROMPT,config=config,
                resume=resume,source_sha=self.spec['source_sha'],image_digest=self.spec['image'].split('@')[1],gpu='fixture'))
            write_json(stage/'training.json',dict(start_step=start,end_step=end,executed_steps=end-start,
                gradient_checks=end-start,nonzero_gradient_checks=end-start,train_loss=.4,peak_allocated_bytes=100,
                frozen_parameters_verified=True,optimizer_steps_at_resume=[start],scheduler_step_at_resume=start,
                learning_rate_trace=[dict(step=s,rates=[config['learning_rate']]) for s in range(start+1,end+1)],
                learning_rate_intervention={'from':.0002,'to':.00005,'scheduler_last_epoch':100} if start==100 else None))
            seal_output(stage)
            if end in (200,250):endpoints.append(c.inspect_checkpoint(cp,self.data,self.spec['seed'],end))
            start=end;parent=cp
        write_json(self.training/'endpoints.json',endpoints);seal_output(self.training)

    def pack(self):
        archive=self.root/'output.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:tar.add(self.results,arcname='results')
        return archive

    def test_full_fresh_training_package_accepted_after_relocation(self):
        r=verify_training_archive(self.pack(),self.data,self.spec)
        self.assertEqual(r['status'],'accepted');self.assertFalse(r['quality_accepted'])
        self.assertEqual([e['step'] for e in r['endpoints']],[200,250])

    def test_tampered_state_rejected(self):
        (self.training/'step200/checkpoints/checkpoint-200/rng_state.pth').write_text('corrupt')
        with self.assertRaises(ValueError):verify_training_archive(self.pack(),self.data,self.spec)

    def test_resealed_wrong_seed_and_provenance_rejected(self):
        for field,value in [('source_sha','f'*40),('config',c.training_config(123 if self.spec['seed'] == 42 else 42,150))]:
            p=self.training/'step150/run.json';original=json.loads(p.read_text());run=copy.deepcopy(original)
            run[field]=value;write_json(p,run);seal_output(p.parent);seal_output(self.training)
            with self.assertRaises(ValueError):verify_training_archive(self.pack(),self.data,self.spec)
            write_json(p,original);seal_output(p.parent);seal_output(self.training)

    def test_failed_completion_or_holdout_rejected(self):
        p=self.training/'status.json';v=json.loads(p.read_text());v['holdout_accessed']=True
        write_json(p,v);seal_output(self.training)
        with self.assertRaises(ValueError):verify_training_archive(self.pack(),self.data,self.spec)

    def test_resealed_lr_drift_rejected(self):
        p=self.training/'step150/training.json';v=json.loads(p.read_text())
        v['learning_rate_trace'][0]['rates']=[.0002]
        write_json(p,v);seal_output(p.parent);seal_output(self.training)
        with self.assertRaisesRegex(ValueError,'LR'):
            verify_training_archive(self.pack(),self.data,self.spec)

    def test_missing_optimizer_state_rejected_after_resealing(self):
        p=self.training/'step250/checkpoints/checkpoint-250/optimizer.pt';p.unlink()
        seal_output(self.training/'step250');seal_output(self.training)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            verify_training_archive(self.pack(),self.data,self.spec)
