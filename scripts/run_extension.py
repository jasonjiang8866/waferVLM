"""Validation-only 200→250→300 continuation from the immutable released full state."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from wafervlm.core import audit_dataset, sha256_file, write_json
from wafervlm.experiment import seal_output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--parent-train',default='/app/profile-input/train')
    p.add_argument('--timeout-seconds',type=int,default=9600)
    a=p.parse_args()
    if not 60<=a.timeout_seconds<=9600:p.error('Invalid extension deadline')
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/extension.json').read_text())
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    audit_dataset(a.dataset);write_json(root/'workload.json',cfg)
    shutil.copyfile(Path(a.dataset)/'validation.jsonl',root/'validation-reference.jsonl')
    start=time.monotonic()
    def run(*parts):
        remaining=a.timeout_seconds-(time.monotonic()-start)
        if remaining<=0:raise TimeoutError('Extension deadline')
        subprocess.run([sys.executable,'-m','wafervlm.cli',*map(str,parts)],check=True,timeout=remaining)
    try:
        from wafervlm.resume import check_resume
        from wafervlm.modeling import MODEL_REVISION
        parent=Path(a.parent_train);checkpoint=parent/'checkpoints/checkpoint-200'
        for key,name in [('parent_adapter_sha256','adapter_model.safetensors'),('parent_optimizer_sha256','optimizer.pt')]:
            if sha256_file(checkpoint/name)!=cfg[key]:raise ValueError('Unexpected parent state')
        check_resume(checkpoint,a.dataset,MODEL_REVISION,dict(steps=250,device='cuda',
            **{k:cfg[k] for k in ('seed','accumulation','rank','learning_rate','max_length')}))
        origin=root/'step200';(origin/'adapter').mkdir(parents=True)
        for name in ('adapter_config.json','adapter_model.safetensors'):
            shutil.copyfile(checkpoint/name,origin/'adapter'/name)
        shutil.copyfile(parent/'run.json',origin/'run.json');seal_output(origin)
        shutil.copyfile(parent/'resume-origin.json',root/'resume-origin.json')
        candidates={200:origin/'adapter'};scores=[];common=['--dataset',a.dataset,'--device','cuda']
        for step in cfg['validation_steps']:
            if step>200:
                print(f'EXTENSION_STAGE=train-{step}',flush=True)
                out=root/f'step{step}'
                run('train',*common,'--output',out,'--steps',step,'--resume-from',checkpoint,
                    '--seed',cfg['seed'],'--accumulation',cfg['accumulation'],'--rank',cfg['rank'],
                    '--learning-rate',cfg['learning_rate'],'--max-length',cfg['max_length'])
                checkpoint=out/f'checkpoints/checkpoint-{step}';candidates[step]=out/'adapter'
                training=json.loads((out/'training.json').read_text())
                if training['executed_steps']!=50 or training['end_step']!=step:
                    raise ValueError('Expected exactly 50 new optimizer steps')
            print(f'EXTENSION_STAGE=validation-{step}',flush=True)
            dest=root/f'validation-{step}'
            run('evaluate',*common,'--split','validation','--adapter',candidates[step],
                '--include-loss','--output',dest)
            scores.append({'step':step,**json.loads((dest/'metrics.json').read_text())})
        best=max(scores,key=lambda r:(r['macro_f1'],-r['step']))
        write_json(root/'selection.json',{'criterion':cfg['selection'],'best_step':best['step'],
            'validation':scores,'test_used_for_selection':False,'test_accessed':False})
        write_json(root/'status.json',{'status':'completed','start_step':200,'end_step':300,
            'additional_steps':100,'best_step':best['step'],'test_accessed':False,
            'elapsed_seconds':time.monotonic()-start})
    except BaseException as exc:
        write_json(root/'status.json',{'status':'failed','error_type':type(exc).__name__,
            'elapsed_seconds':time.monotonic()-start})
        raise
    finally:seal_output(root)


if __name__=='__main__':main()
