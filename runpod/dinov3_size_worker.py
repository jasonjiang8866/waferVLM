"""One temporary 4090 for four seed-17 development fits."""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from runpod import dinov3_worker as w
from runpod.dinov3_size17 import ARMS, train_args
from wafervlm.dinov3_backbones import BACKBONES


class BaseUpload(w.Handler):
    weight_limits = {'config.json': 1000000, 'model.safetensors': 342662192}

    def allows_upload(self):
        return w.STATE.get('phase') == 'await-base' or w.READY.is_set()

    def upload_root(self):
        base = BACKBONES['base']
        return Path(os.environ['HF_HOME'])/'hub'/('models--'+base.model_id.replace('/','--'))/'snapshots'/base.revision


def work():
    record = dict(phase='size17', controller_sha=os.environ['DINO_CONTROLLER_SHA'],
                  image=os.environ['DINO_IMAGE'], image_source_sha=os.environ['WAFERVLM_SOURCE_SHA'],
                  plan_sha256=os.environ['DINO_PLAN_SHA256'], overlay_sha256=os.environ['DINO_OVERLAY_SHA256'])
    try:
        w.OUT.mkdir(parents=True,exist_ok=False)
        w.write('hardware.json',w.hardware_smoke())
        w.write('small-weights.json',w.embedded_weights())
        base = BACKBONES['base']
        w.WEIGHTS = dict(files_sha256={'config.json':base.config_sha256,'model.safetensors':base.weights_sha256})
        w.STATE.update(phase='await-base',received=[])
        if not w.READY.wait(750):
            raise TimeoutError('Base transfer exceeded startup window')
        environment = dict(os.environ, WAFERVLM_SOURCE_SHA=record['controller_sha'])
        def run(label,args):
            w.STATE['phase'] = label
            remaining = float(os.environ['DINO_DEADLINE']) - time.time() - 120
            if remaining < 60:
                raise TimeoutError('Insufficient time for next arm')
            with (w.OUT / (label+'.log')).open('w') as stream:
                subprocess.run([sys.executable,*args],cwd=w.ROOT,env=environment,stdout=stream,
                               stderr=subprocess.STDOUT,timeout=remaining,check=True)
        run('data',['scripts/prepare_public_data.py','--trust-pickle','--cache','/workspace/dino-source',
                    '--output',str(w.DATASET)])
        shutil.copyfile(w.DATASET/'validation.jsonl',w.OUT/'validation.jsonl')
        for size, mode in ARMS:
            name = size+'-'+mode
            run(name,train_args(size,mode,w.DATASET,w.OUT/'models'/name))
        record['status']='completed'
    except Exception as exc:
        record.update(status='rejected',error_type=type(exc).__name__,failed_phase=w.STATE.get('phase'))
    finally:
        if w.OUT.is_dir():
            w.write('worker.json',record)
            with tarfile.open(w.ARCHIVE,'w:gz') as archive:
                archive.add(w.OUT,arcname='results')
            record['archive']=dict(size=w.ARCHIVE.stat().st_size,sha256=w.digest(w.ARCHIVE))
        w.STATE.update(record)


def main():
    w.TOKEN = os.environ.pop('DINO_RESULT_TOKEN')
    w.WEIGHTS = json.loads(os.environ.pop('DINO_WEIGHTS'))
    server = ThreadingHTTPServer(('0.0.0.0',8888),BaseUpload)
    threading.Thread(target=work,daemon=True).start()
    server.serve_forever()


if __name__=='__main__':
    main()
