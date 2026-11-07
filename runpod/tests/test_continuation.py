import copy
import hashlib
import json
import unittest
from pathlib import Path
from runpod.provider import IMAGE,SOURCE
from runpod.verify import _verify_continuation


class ContinuationTests(unittest.TestCase):
    def fixture(self):
        root=Path(__file__).resolve().parents[2]
        cfg=json.loads((root/'configs/continuation.json').read_text())
        evidence=json.loads((root/'docs/evidence/wm811k-lot17.json').read_text())
        base='continuation/'
        values={'worker.json':{'status':'accepted','profiles':['continuation'],'source_sha':SOURCE,
            'controller_sha':'a'*40,'config_sha256':hashlib.sha256((root/'configs/continuation.json').read_bytes()).hexdigest()},
            'acquisition.json':evidence,
            base+'resume-origin.json':{'run_id':cfg['parent_run_id'],'artifact_id':cfg['parent_artifact_id'],
                'archive_sha256':cfg['parent_archive_sha256'],'global_step':100},
            base+'status.json':{'status':'completed','start_step':100,'end_step':200,'additional_steps':100,'best_step':150}}
        files={};scores=[]
        for part in ('','step100/','step150/','step200/','validation-100/','validation-150/','validation-200/','selected-test/'):
            values[base+part+'checksums.json']={'fixture':'b'*64}
            files['results/'+base+part+'checksums.json']=(None,'b'*64)
        for name in ['step100/adapter/adapter_model.safetensors','step150/checkpoints/checkpoint-150/adapter_model.safetensors',
                     'step150/checkpoints/checkpoint-150/optimizer.pt','step150/checkpoints/checkpoint-150/scheduler.pt',
                     'step150/checkpoints/checkpoint-150/rng_state.pth']:
            files['results/'+base+name]=(None,'b'*64)
        run={'source_sha':SOURCE,'image_digest':IMAGE.split('@')[1],'dataset_sha256':evidence['dataset_sha256'],
             'config':{'device':'cuda','split':'validation'},'adapter_checksums_sha256':'b'*64}
        for step in (150,200):
            values[base+f'step{step}/training.json']={'start_step':step-50,'end_step':step,'gradient_checks':50,
                'executed_steps':50,'nonzero_gradient_checks':50,'train_loss':.3,'peak_allocated_bytes':100}
            values[base+f'step{step}/run.json']={**run,'resume':{'global_step':step-50,'adapter_sha256':'b'*64,
                'optimizer_sha256':'d1168af8d924c1e5654ad7854dd9d2344763aa68c119f8eae4716a8aa32e7fbe' if step==150 else 'b'*64,
                'scheduler_sha256':'b'*64,'rng_sha256':'b'*64}}
        for step,score in [(100,.5),(150,.7),(200,.6)]:
            m={'n':160,'answer_token_loss':.4,'answer_tokens':800,'macro_f1':score}
            values[base+f'validation-{step}/metrics.json']=m
            values[base+f'validation-{step}/run.json']=copy.deepcopy(run)
            scores.append({'step':step,**m})
        values[base+'selection.json']={'validation':scores,'best_step':150,'test_used_for_selection':False,'criterion':cfg['selection']}
        values[base+'selected-test/metrics.json']={'n':160,'macro_f1':.6}
        values[base+'selected-test/run.json']={**run,'config':{'device':'cuda','split':'test'}}
        return values,files

    def test_accept_and_reject_step_parent_selection_and_validation_faults(self):
        values,files=self.fixture();_verify_continuation(values.__getitem__,files,'a'*40)
        cases=[('continuation/step150/training.json','executed_steps',100),
               ('continuation/resume-origin.json','run_id',1),
               ('continuation/selection.json','best_step',200),
               ('continuation/selection.json','test_used_for_selection',True),
               ('continuation/validation-150/metrics.json','n',159),
               ('continuation/validation-150/metrics.json','answer_token_loss',float('nan')),
               ('continuation/selected-test/metrics.json','n',159)]
        for path,key,value in cases:
            v,f=self.fixture();v[path][key]=value
            with self.subTest(path=path,key=key),self.assertRaises(ValueError):_verify_continuation(v.__getitem__,f,'a'*40)
        v,f=self.fixture();v['continuation/step200/run.json']['resume']['rng_sha256']='bad'
        with self.assertRaises(ValueError):_verify_continuation(v.__getitem__,f,'a'*40)
