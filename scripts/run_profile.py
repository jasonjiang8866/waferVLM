"""Paired short resume timings; never promote diagnostic training or read test scores."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from wafervlm.core import sha256_file, write_json
from wafervlm.experiment import seal_output


def evaluation_probe(dataset, checkpoint, root):
    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import GenerationConfig, set_seed
    from wafervlm.core import LABELS, audit_dataset, checked_image, parse_label
    from wafervlm.modeling import load_model, encode
    from wafervlm.profiling import PhaseRecorder
    rec=PhaseRecorder(root, 'cuda'); set_seed(17)
    rows=audit_dataset(dataset)['validation']
    samples=[sorted((r for r in rows if r['label']==label),key=lambda r:r['id'])[0] for label in LABELS]
    with rec.scope('evaluation_model_load'):
        model,processor=load_model()
        model=PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
        model.eval()
    eos=processor.tokenizer.eos_token_id
    gen=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=16,eos_token_id=eos,
                         pad_token_id=processor.tokenizer.pad_token_id or eos,use_cache=True)
    results=[]
    torch.cuda.reset_peak_memory_stats()
    for row in samples:
        with rec.scope('evaluation_image_decode'):
            with Image.open(checked_image(dataset,row)) as image: image=image.convert('RGB')
        with rec.scope('evaluation_encode'): batch=encode(processor,image)
        with rec.scope('evaluation_host_to_device'): batch=batch.to('cuda')
        with rec.scope('evaluation_generate'), torch.inference_mode(): generated=model.generate(**batch,generation_config=gen)
        tokens=generated[0,batch['input_ids'].shape[-1]:].tolist()
        text=processor.tokenizer.decode(tokens,skip_special_tokens=True)
        results.append({'id':row['id'],'label':row['label'],'image_sha256':row['image_sha256'],
            'prediction':parse_label(text,len(tokens)>=16 and eos not in tokens),'generated_tokens':len(tokens)})
    write_json(root/'evaluation-probe.json',{'split':'validation','n':len(results),'samples':results,
        'timings':rec.events,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'purpose':'fixed eight-image timing probe only; not a quality estimate or checkpoint selection'})


def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset', required=True); p.add_argument('--output', required=True)
    p.add_argument('--parent-train', default='/app/profile-input/train'); a=p.parse_args()
    root=Path(a.output); root.mkdir(parents=True, exist_ok=False)
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/profile.json').read_text())
    write_json(root/'workload.json', cfg)
    parent=Path(a.parent_train); cp=parent/'checkpoints/checkpoint-200'
    shutil.copyfile(parent/'resume-origin.json', root/'resume-origin.json')
    if sha256_file(cp/'adapter_model.safetensors') != cfg['parent_adapter_sha256']:
        raise ValueError('Unexpected diagnostic parent')
    start=time.monotonic(); durations={}
    try:
        for arm in cfg['arms']:
            remaining=cfg['workload_seconds']-(time.monotonic()-start)
            if remaining < 30: raise TimeoutError('Diagnostic deadline')
            args=[sys.executable, '-m', 'wafervlm.cli', 'train', '--dataset', a.dataset,
                  '--device', 'cuda', '--output', str(root/arm), '--resume-from', str(cp),
                  '--steps', str(cfg['parent_step']+cfg['steps_per_arm'])]
            for name in ('seed', 'accumulation', 'rank', 'learning_rate', 'max_length'):
                args += ['--'+name.replace('_','-'), str(cfg[name])]
            if cfg['full_determinism']: args.append('--full-determinism')
            if arm == 'profiled': args.append('--profile')
            begin=time.monotonic(); subprocess.run(args, check=True, timeout=remaining)
            durations[arm]=time.monotonic()-begin
        import torch
        from safetensors.torch import load_file
        paths=[root/arm/'adapter/adapter_model.safetensors' for arm in cfg['arms']]
        states=[load_file(p) for p in paths]
        if states[0].keys() != states[1].keys(): raise ValueError('Adapter keys changed')
        max_abs=max(float((states[0][k]-states[1][k]).abs().max()) for k in states[0])
        close=all(torch.allclose(states[0][k], states[1][k], atol=1e-6, rtol=1e-5) for k in states[0])
        write_json(root/'comparison.json', {'arm_wall_seconds': durations,
            'instrumented_over_unprofiled_wall_ratio': durations['profiled']/durations['unprofiled'],
            'max_abs_adapter_difference': max_abs, 'adapter_allclose': close,
            'adapter_sha256': {arm:sha256_file(path) for arm,path in zip(cfg['arms'],paths)},
            'atol':1e-6,'rtol':1e-5,'full_determinism':cfg['full_determinism'],
            'notes':'Same complete step-200 parent, seed and data position. Instrumentation overhead and cold-loading order are not speedup.'})
        evaluation_probe(a.dataset, cp, root)
        if not close: raise ValueError('Paired training trajectory tolerance exceeded')
        # Disposable diagnostic weights are not new usable/released checkpoints.
        for arm in cfg['arms']:
            shutil.rmtree(root/arm/'adapter'); shutil.rmtree(root/arm/'checkpoints')
            seal_output(root/arm)
        write_json(root/'status.json', {'status':'completed','parent_step':200,
            'diagnostic_only':True,'test_accessed':False,'elapsed_seconds':time.monotonic()-start})
    except BaseException as exc:
        write_json(root/'status.json', {'status':'failed','error_type':type(exc).__name__,
            'elapsed_seconds':time.monotonic()-start})
        raise
    finally: seal_output(root)


if __name__ == '__main__': main()
