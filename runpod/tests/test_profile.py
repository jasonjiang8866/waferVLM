import copy
import hashlib
import json
import unittest
from pathlib import Path
from runpod.verify import _verify_profile
from runpod.provider import IMAGE, SOURCE


class ProfileTests(unittest.TestCase):
    def fixture(self):
        root=Path(__file__).resolve().parents[2]
        cfg=json.loads((root/'configs/profile.json').read_text())
        evidence=json.loads((root/'docs/evidence/wm811k-lot17.json').read_text())
        values={'worker.json':{'status':'accepted','profiles':['profile'],'source_sha':SOURCE,
            'controller_sha':'a'*40,'config_sha256':hashlib.sha256((root/'configs/profile.json').read_bytes()).hexdigest()},
            'acquisition.json':evidence, 'profile/workload.json':cfg,
            'profile/resume-origin.json':{'run_id':42502913001,'global_step':200,'adapter_sha256':cfg['parent_adapter_sha256']},
            'profile/status.json':{'status':'completed','diagnostic_only':True,'test_accessed':False,'elapsed_seconds':800},
            'profile/comparison.json':{'adapter_allclose':True,'all_pairs_allclose':True,'max_abs_adapter_difference':0,
                'arm_wall_seconds':{'unprofiled':200,'profiled':250}},
            'profile/profiled/timings.json':{'device':'cuda','synchronized':True,'events':[
                {'phase':p,'wall_seconds':.1} for p in ('image_verify_decode','image_preprocess','tokenize',
                'host_to_device','forward_and_loss','backward','optimizer_step','gradient_check','checkpoint_save','adapter_export')]},
            'profile/profiled/execution.json':{'parameter_devices':['cuda:0']},
            'profile/profiled/operators.json':{'cuda_event_count':10},
            'profile/evaluation-probe.json':{'split':'validation','n':8,'samples':[{}]*8},
            'validation-analysis/metrics.json':{'split':'validation','checkpoints':{str(s):{'n':160} for s in (100,150,200)}}}
        for arm in cfg['arms']:
            values[f'profile/{arm}/training.json']={'start_step':200,'end_step':204,'executed_steps':4,
                'gradient_checks':4,'nonzero_gradient_checks':4,'optimizer_steps_at_resume':[200],
                'train_loss':.2,'peak_allocated_bytes':100,
                'learning_rate_trace':[{'step':s,'rates':[cfg['learning_rate']]} for s in range(201,205)]}
            values[f'profile/{arm}/run.json']={'source_sha':SOURCE,'image_digest':IMAGE.split('@')[1],
                'dataset_sha256':evidence['dataset_sha256'],'model_revision':'2fc06364715b967f1860aea9cf38778875588b17',
                'config':{'device':'cuda',**{k:cfg[k] for k in ('seed','accumulation','rank','learning_rate','max_length')}},
                'resume':{'adapter_sha256':cfg['parent_adapter_sha256'],'optimizer_sha256':cfg['parent_optimizer_sha256']}}
        return values

    def test_valid_diagnostic_and_failure_gates(self):
        valid=self.fixture();_verify_profile(valid.__getitem__,{},'a'*40)
        for path,key,value in [
            ('profile/status.json','test_accessed',True),
            ('profile/status.json','elapsed_seconds',1500),
            ('profile/resume-origin.json','run_id',1),
            ('profile/unprofiled/training.json','optimizer_steps_at_resume',[0]),
            ('profile/profiled/training.json','executed_steps',3),
            ('profile/profiled/training.json','train_loss',float('nan')),
            ('profile/profiled/timings.json','events',[]),
            ('profile/profiled/execution.json','parameter_devices',['cpu']),
            ('profile/profiled/operators.json','cuda_event_count',0),
            ('profile/comparison.json','adapter_allclose',False),
            ('profile/comparison.json','all_pairs_allclose',False),
            ('profile/evaluation-probe.json','split','test')]:
            with self.subTest(path=path,key=key):
                values=copy.deepcopy(valid);values[path][key]=value
                with self.assertRaises(ValueError):_verify_profile(values.__getitem__,{},'a'*40)
        values=copy.deepcopy(valid)
        values['profile/profiled/training.json']['learning_rate_trace'][0]['rates']=[.0002]
        with self.assertRaises(ValueError):_verify_profile(values.__getitem__,{},'a'*40)
        with self.assertRaises(ValueError):
            _verify_profile(valid.__getitem__,{'results/profile/profiled/checkpoints/checkpoint-204/optimizer.pt':(None,'b'*64)},'a'*40)


if __name__=='__main__':unittest.main()
