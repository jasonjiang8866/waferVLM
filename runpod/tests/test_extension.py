import copy
import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from runpod.provider import IMAGE,SOURCE
from runpod.verify import _verify_extension
from runpod import worker


class ExtensionTests(unittest.TestCase):
    def fixture(self):
        root=Path(__file__).resolve().parents[2]
        cfg=json.loads((root/'configs/extension.json').read_text())
        expected=json.loads((root/'docs/evidence/wm811k-lot17.json').read_text())
        core=runpy.run_path(str(root/'src/wafervlm/core.py'))
        rows=[dict(id=str(i),label=core['LABELS'][i//20],image_sha256=f'{i:064x}',prediction=core['LABELS'][i//20]) for i in range(160)]
        base='extension/';sha='b'*64
        v={'worker.json':dict(status='accepted',profiles=['extension'],source_sha=SOURCE,controller_sha='a'*40,
           config_sha256=hashlib.sha256((root/'configs/extension.json').read_bytes()).hexdigest()),'acquisition.json':expected,
           base+'workload.json':cfg,base+'validation-reference.jsonl':rows,
           base+'resume-origin.json':dict(run_id=cfg['parent_run_id'],release_tag=cfg['parent_release_tag'],
               zip_sha256=cfg['parent_zip_sha256'],archive_sha256=cfg['parent_archive_sha256'],
               adapter_sha256=cfg['parent_adapter_sha256'],global_step=200),
           base+'status.json':dict(status='completed',start_step=200,end_step=300,additional_steps=100,
               best_step=200,test_accessed=False,elapsed_seconds=100)}
        files={'results/'+base+'validation-reference.jsonl':(None,expected['dataset']['manifest_sha256']['validation']),
               'results/'+base+'step200/adapter/adapter_model.safetensors':(None,cfg['parent_adapter_sha256'])}
        for part in ('','step200/','step250/','step300/','validation-200/','validation-250/','validation-300/'):
            v[base+part+'checksums.json']={'fixture':sha};files['results/'+base+part+'checksums.json']=(None,sha)
        common=dict(source_sha=SOURCE,image_digest=IMAGE.split('@')[1],dataset_sha256=expected['dataset_sha256'],
                    model_revision='2fc06364715b967f1860aea9cf38778875588b17',gpu='test-cuda',adapter_checksums_sha256=sha)
        for step in (250,300):
            resume=dict(global_step=step-50,lower_learning_rate=False,parent_learning_rate=cfg['learning_rate'])
            for kind in ('adapter','optimizer','scheduler','rng'):resume[kind+'_sha256']=cfg['parent_'+kind+'_sha256'] if step==250 else sha
            v[base+f'step{step}/run.json']={**common,'config':dict(device='cuda',**{k:cfg[k] for k in ('seed','rank','accumulation','learning_rate','max_length')}),'resume':resume}
            v[base+f'step{step}/training.json']=dict(start_step=step-50,end_step=step,executed_steps=50,gradient_checks=50,
                nonzero_gradient_checks=50,optimizer_steps_at_resume=[step-50],train_loss=.2,peak_allocated_bytes=100,
                learning_rate_intervention=None,learning_rate_trace=[{'step':i,'rates':[cfg['learning_rate']]} for i in range(step-49,step+1)])
            prefix=base+f'step{step}/checkpoints/checkpoint-{step}/'
            for name in ('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','training_args.bin'):
                files['results/'+prefix+name]=(None,sha)
            v[prefix+'trainer_state.json']={'global_step':step};files['results/'+base+f'step{step}/adapter/adapter_model.safetensors']=(None,sha)
        scores=[]
        for step in (200,250,300):
            pred=copy.deepcopy(rows)
            m={**core['metrics'](pred),'answer_token_loss':.2,'answer_tokens':320}
            v[base+f'validation-{step}/predictions.jsonl']=pred;v[base+f'validation-{step}/metrics.json']=m
            v[base+f'validation-{step}/run.json']={**common,'config':{'device':'cuda','split':'validation'}}
            scores.append({'step':step,**m})
        v[base+'selection.json']=dict(criterion=cfg['selection'],best_step=200,validation=scores,test_used_for_selection=False,test_accessed=False)
        return v,files

    def test_accept_ties_keep_parent_and_reject_state_metric_and_test_leakage(self):
        v,f=self.fixture();_verify_extension(v.__getitem__,f,'a'*40)
        for path,key,value in [
            ('step250/training.json','executed_steps',100),('step300/training.json','optimizer_steps_at_resume',[0]),
            ('step250/training.json','learning_rate_intervention',{'from':.0002}),
            ('validation-250/metrics.json','macro_f1',.5),('validation-300/metrics.json','answer_token_loss',float('nan')),
            ('selection.json','best_step',300),('selection.json','test_accessed',True),('status.json','elapsed_seconds',10000)]:
            v,f=self.fixture();v['extension/'+path][key]=value
            with self.subTest(path=path,key=key),self.assertRaises(ValueError):_verify_extension(v.__getitem__,f,'a'*40)
        v,f=self.fixture();v['extension/step300/run.json']['resume']['rng_sha256']='bad'
        with self.assertRaises(ValueError):_verify_extension(v.__getitem__,f,'a'*40)
        v,f=self.fixture();v['extension/validation-250/predictions.jsonl'][0]['id']='wrong'
        with self.assertRaises(ValueError):_verify_extension(v.__getitem__,f,'a'*40)
        v,f=self.fixture();f['results/extension/selected-test/metrics.json']=(None,'b'*64)
        with self.assertRaisesRegex(ValueError,'validation-only'):_verify_extension(v.__getitem__,f,'a'*40)
        v,f=self.fixture();del f['results/extension/step300/checkpoints/checkpoint-300/optimizer.pt']
        with self.assertRaisesRegex(ValueError,'resumable'):_verify_extension(v.__getitem__,f,'a'*40)

    def test_progress_distinguishes_validation_from_training(self):
        with tempfile.TemporaryDirectory() as d, patch.object(worker,'OUT',Path(d)),patch.object(worker,'RESULT',{'status':'running','phase':'extension'}):
            log=Path(d)/'extension.log'
            log.write_text('EXTENSION_STAGE=train-250\n250/250 [done]\nEXTENSION_STAGE=validation-250\n160/160 [done]')
            self.assertNotIn('progress',worker.progress_snapshot())
            log.write_text(log.read_text()+'\nEXTENSION_STAGE=train-300\n251/300 [running]')
            state=worker.progress_snapshot()
            self.assertEqual(state['extension_stage'],'train-300')
            self.assertEqual(state['progress']['optimizer_step'],251)
