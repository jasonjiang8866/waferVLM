"""Resume 100 additional optimizer steps, select on validation, test once after selection."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from wafervlm.core import audit_dataset, write_json
from wafervlm.experiment import seal_output


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--parent-train',default='/app/resume-input/train');p.add_argument('--timeout-seconds',type=int,default=9600)
    a=p.parse_args()
    if not 60 <= a.timeout_seconds <= 10800: p.error('Invalid continuation deadline')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/continuation.json').read_text())
    audit_dataset(a.dataset);write_json(root/'workload.json',cfg)
    started=time.monotonic()
    def run(*parts):
        remaining=a.timeout_seconds-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('Continuation deadline')
        subprocess.run([sys.executable,'-m','wafervlm.cli',*map(str,parts)],check=True,timeout=remaining)
    try:
        parent=Path(a.parent_train)
        checkpoint=parent/'checkpoints/checkpoint-100'
        from wafervlm.resume import check_resume
        from wafervlm.modeling import MODEL_REVISION
        check_resume(checkpoint,a.dataset,MODEL_REVISION,dict(steps=150,device='cuda',
            **{k:cfg[k] for k in ('seed','accumulation','rank','learning_rate','max_length')}),
            lower_learning_rate=cfg['lower_learning_rate'])
        # Export a small independently sealed adapter for evaluation; no optimizer loading here.
        origin=root/'step100';(origin/'adapter').mkdir(parents=True)
        for name in ['adapter_config.json','adapter_model.safetensors']:
            shutil.copyfile(checkpoint/name,origin/'adapter'/name)
        shutil.copyfile(parent/'run.json',origin/'run.json');seal_output(origin)
        shutil.copyfile(parent/'resume-origin.json',root/'resume-origin.json')
        candidates={100:origin/'adapter'};scores=[]
        common=['--dataset',a.dataset,'--device','cuda']
        for step in cfg['validation_steps']:
            if step>100:
                out=root/f'step{step}'
                intervention = ['--lower-learning-rate'] if step == 150 and cfg['lower_learning_rate'] else []
                run('train',*common,'--output',out,'--steps',step,'--resume-from',checkpoint,
                    '--seed',cfg['seed'],'--accumulation',cfg['accumulation'],'--rank',cfg['rank'],
                    '--learning-rate',cfg['learning_rate'],'--max-length',cfg['max_length'],*intervention)
                checkpoint=out/f'checkpoints/checkpoint-{step}'
                candidates[step]=out/'adapter'
                training=json.loads((out/'training.json').read_text())
                if training['executed_steps']!=50 or training['end_step']!=step:
                    raise ValueError('Expected exactly 50 additional steps per segment')
            dest=root/f'validation-{step}'
            run('evaluate',*common,'--split','validation','--adapter',candidates[step],
                '--include-loss','--output',dest)
            scores.append({'step':step,**json.loads((dest/'metrics.json').read_text())})
        best=max(scores,key=lambda row:(row['macro_f1'],-row['step']))
        selection={'criterion':cfg['selection'],'best_step':best['step'],'validation':scores,
                   'test_used_for_selection':False}
        write_json(root/'selection.json',selection)  # fixed before accessing test
        run('evaluate',*common,'--split','test','--adapter',candidates[best['step']],
            '--output',root/'selected-test')
        write_json(root/'status.json',{'status':'completed','start_step':100,'end_step':200,
            'additional_steps':100,'best_step':best['step'],'elapsed_seconds':time.monotonic()-started})
    except BaseException as exc:
        write_json(root/'status.json',{'status':'failed','error_type':type(exc).__name__,
                   'elapsed_seconds':time.monotonic()-started})
        raise
    finally:seal_output(root)


if __name__=='__main__':main()
