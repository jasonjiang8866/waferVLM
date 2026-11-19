"""Bounded DINO-only controller; reuse the verified image and freeze before holdout."""
import base64
import gzip
import hashlib
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from runpod.provider import REPOSITORY, Runpod, NoRedirect, RemoteHTTPError, request, finite, budget
from runpod.dinov3_controller import image_provenance
from runpod.dinov3_study import PLAN, FROZEN, plan, input_pin, verify_results
from runpod.dinov3_study_ledger import reconcile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dinov3-study-results'
AUTH = 'wafervlm-dinov3-study-v1'
BASE = f'https://api.github.com/repos/{REPOSITORY}'
GPU_CAPS = {'NVIDIA GeForce RTX 4090': .89, 'NVIDIA GeForce RTX 5090': 1.39}
DISK_GB = 60
DISK_RATE = DISK_GB * .10 / (28 * 24)
WORK_SECONDS = 2700
CLEANUP_SECONDS = 600
RUNTIME_FILES = ['src/wafervlm/' + n + '.py' for n in
                 ('dinov3','core','experiment','data','acquisition','vision_comparison','confirmation')]
RUNTIME_FILES += ['scripts/run_dinov3.py','scripts/prepare_public_data.py','configs/dinov3.json','uv.lock',
                  'runpod/dinov3_worker.py','runpod/dinov3_bundle.py','runpod/dinov3_image_weights.py']


def save(name, data):
    OUT.mkdir(exist_ok=True)
    p = OUT / name; tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n'); tmp.replace(p)


def stage(name):
    save('progress.json', dict(stage=name, checked_epoch=time.time()))
    print('DINO controller stage: '+name, flush=True)


def error_record(exc):
    result = dict(error_type=type(exc).__name__)
    if isinstance(exc, RemoteHTTPError):
        result.update(http_status=exc.status, categories=exc.categories)
    # Never emit remote response bodies, URLs or exception strings.
    path = OUT / 'progress.json'
    if path.exists():
        result['stage'] = json.loads(path.read_text())['stage']
    return result


def identity():
    run = os.environ.get('GITHUB_RUN_ID', '')
    if not run.isdigit() or os.environ.get('GITHUB_REPOSITORY') != REPOSITORY:
        raise ValueError('Unexpected repository/run')
    return run


def manual_guard():
    run = identity()
    sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    if (os.environ.get('GITHUB_EVENT_NAME') != 'workflow_dispatch' or os.environ.get('GITHUB_REF') != 'refs/heads/main'
            or os.environ.get('GITHUB_RUN_ATTEMPT') != '1' or os.environ.get('CONFIRM_PAID') != 'true'
            or os.environ.get('EXPECTED_SHA') != sha or os.environ.get('DINO_PHASE') not in ('validation','holdout')):
        raise ValueError('First manual dispatch and exact reviewed main required')
    if request(BASE + '/git/ref/heads/main', token=os.environ['GH_TOKEN'])['object']['sha'] != sha:
        raise ValueError('Main moved before dispatch')
    return run, sha


def quote(rate, gpu, ledger):
    if gpu not in GPU_CAPS or not finite(rate, .001) or rate > GPU_CAPS[gpu]:
        raise ValueError('GPU quote exceeds approved cap')
    maximum = (WORK_SECONDS + CLEANUP_SECONDS) / 3600 * (rate + DISK_RATE)
    if maximum + 1 > 10 - ledger['usd'] or ledger['seconds'] + WORK_SECONDS + CLEANUP_SECONDS > 30000:
        raise ValueError('Insufficient cumulative DINO budget/time and cleanup reserve')
    return dict(gpu_hourly_usd=rate, disk_hourly_usd=DISK_RATE, work_seconds=WORK_SECONDS,
                cleanup_seconds=CLEANUP_SECONDS, maximum_estimated_usd=maximum,
                approved_total_usd=10, approved_total_seconds=30000)


def overlay(phase):
    files = ['runpod/dinov3_study.py','runpod/dinov3_study_worker.py','configs/dinov3-confirmation.json',
             'docs/showcase/dinov3-seed17/validation.jsonl']
    if phase == 'holdout':
        files += ['docs/evidence/dinov3-six-endpoints.json','runpod/vlm_holdout.py',
                  'docs/evidence/final-holdout-v1/manifest.jsonl']
    elif phase != 'validation':
        raise ValueError('Unknown phase')
    payload = json.dumps({n:(ROOT/n).read_text() for n in files}, sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    packed = base64.b64encode(gzip.compress(payload, mtime=0)).decode()
    code = ("import base64,gzip,json,pathlib,runpy,os,hashlib;os.chdir('/app');"
            "raw=gzip.decompress(base64.b64decode(" + repr(packed) + "));"
            "assert hashlib.sha256(raw).hexdigest()==os.environ['DINO_OVERLAY_SHA256'];"
            "payload=json.loads(raw);"
            "[(pathlib.Path(n).parent.mkdir(parents=True,exist_ok=True),pathlib.Path(n).write_text(v)) for n,v in payload.items()];"
            "runpy.run_module('runpod.dinov3_study_worker',run_name='__main__')")
    return code, digest


def select_capacity(api, candidates, ledger):
    """Reobserve transient catalog shortages; never retry a create request."""
    observations = []
    for attempt in range(4):
        for gpu in candidates:
            query = urllib.parse.urlencode(dict(include='AVAILABILITY',product='POD',cloud='SECURE',count=1,minCudaVersion='12.8'))
            catalog = api.call('/catalog/gpus/'+urllib.parse.quote(gpu,safe='')+'?'+query)
            rate = catalog.get('price',{}).get('secure')
            regions = sorted(d['id'] for d in catalog.get('dataCenters',[]) if d.get('availability') in ('LOW','MEDIUM','HIGH') and re.fullmatch('[A-Za-z0-9-]+',d.get('id','')))
            observations.append(dict(attempt=attempt+1,gpu=gpu,hourly_usd=rate,regions=regions,min_cuda='12.8',checked_epoch=time.time()))
            save('catalog.json', observations)
            if regions and finite(rate,.001) and rate <= GPU_CAPS[gpu]:
                return gpu,regions,quote(rate,gpu,ledger)
        if attempt < 3:
            time.sleep(15)
    raise ValueError('No approved GPU capacity at the CUDA minimum and price cap')


def preflight(api):
    stage('identity')
    run, sha = manual_guard()
    phase, p = os.environ['DINO_PHASE'], plan()
    stage('ledger')
    ledger = reconcile(run)
    save('ledger.json', ledger)
    for name in RUNTIME_FILES:
        old = subprocess.check_output(['git','show',p['image_source_sha']+':'+name], cwd=ROOT)
        if old != (ROOT/name).read_bytes():
            raise ValueError('Runtime differs from pinned DINO image: '+name)
    stage('image')
    image_provenance(p['image'], p['image_source_sha'])
    stage('input_artifact')
    pin = input_pin(phase)
    meta = request(BASE + '/actions/artifacts/' + str(pin['artifact_id']), token=os.environ['GH_TOKEN'])
    if meta.get('expired') is not False or meta.get('digest') != 'sha256:' + pin['zip_sha256']:
        raise ValueError('Frozen model input artifact changed/expired')
    stage('inventory')
    pods = api.listing('/pods','pods')
    if any(x.get('env', {}).get('WAFER_MANAGED_BY') == REPOSITORY or x.get('name','').startswith(('wafervlm-','wafer-20261107')) for x in pods):
        raise ValueError('Existing waferVLM Pod must be reconciled')
    choice = os.environ.get('DINO_GPU_CHOICE','auto')
    if choice != 'auto' and choice not in GPU_CAPS:
        raise ValueError('Unapproved GPU choice')
    candidates = list(GPU_CAPS) if choice == 'auto' else [choice]
    stage('catalog')
    gpu, regions, cost = select_capacity(api,candidates,ledger)
    stage('account_budget')
    account = budget(api,reserve=cost['maximum_estimated_usd']+1)
    _, overlay_sha = overlay(phase)
    spec = dict(run_id=run,controller_sha=sha,image_source_sha=p['image_source_sha'],image=p['image'],
                phase=phase,gpu=gpu,regions=regions,cost=cost,ledger=ledger,
                plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),overlay_sha256=overlay_sha,
                input_artifact=pin,account_checks=account,
                frozen_sha256=hashlib.sha256(FROZEN.read_bytes()).hexdigest() if phase=='holdout' else None,
                selection_reason='4090 first; 5090 only if 4090 capacity/quote fails' if choice=='auto' else 'explicit approved GPU')
    save('launch-spec.json',spec)
    return spec


def owned(pod, run):
    env=pod.get('env',{})
    return pod.get('name')==AUTH+'-'+run and env.get('WAFER_MANAGED_BY')==REPOSITORY and env.get('DINO_RUN_ID')==run and env.get('DINO_AUTH')==AUTH


def cleanup(api):
    run=identity(); path=OUT/'provision.json'
    state=json.loads(path.read_text()) if path.exists() else {}
    save('cleanup.json',dict(status='unknown',run_id=run))
    absent=0
    for _ in range(15):
        matches=[p for p in api.listing('/pods','pods') if p.get('name')==AUTH+'-'+run or (state.get('pod_id') and p.get('id')==state['pod_id'])]
        if not matches:
            absent+=1
            if absent >= (3 if state.get('pod_id') else 12):
                result=dict(status='verified_absent',run_id=run,checks=absent,persistent_volumes_created=False,checked_epoch=time.time())
                if state.get('started_epoch'):
                    elapsed=time.time()-state['started_epoch']; cost=elapsed/3600*(state['rate']+DISK_RATE)
                    result.update(elapsed_billing_bound_s=elapsed,estimated_usd=cost,
                                  cumulative_estimated_usd=state['prior_usd']+cost,
                                  cumulative_runtime_bound_s=state['prior_seconds']+elapsed,
                                  within_cap=state['prior_usd']+cost<=10 and state['prior_seconds']+elapsed<=30000 and elapsed<=WORK_SECONDS+CLEANUP_SECONDS)
                save('cleanup.json',result)
                if result.get('within_cap') is False:
                    raise RuntimeError('Cumulative DINO estimate/time exceeded cap')
                return
        else:
            absent=0
            if len(matches)!=1 or not owned(matches[0],run) or matches[0].get('mounts'):
                raise ValueError('Cleanup ownership/persistent mount mismatch')
            pod=matches[0]; env=pod['env']
            if not re.fullmatch('[A-Za-z0-9_-]{1,128}',pod.get('id','')):
                raise ValueError('Invalid Pod identity')
            if not state:
                state=dict(started_epoch=float(env['DINO_STARTED']),rate=float(env['DINO_RATE']),
                           prior_usd=float(env['DINO_PRIOR_USD']),prior_seconds=float(env['DINO_PRIOR_SECONDS']))
                if (not 0<state['started_epoch']<=time.time() or not 0<state['rate']<=1.39
                        or not 0<=state['prior_usd']<10 or not 0<=state['prior_seconds']<30000):
                    raise ValueError('Invalid recovered cost record')
            state['pod_id']=pod['id'];save('provision.json',state)
            try: api.call('/pods/'+pod['id'],'DELETE')
            except RuntimeError: pass
        time.sleep(10)
    raise RuntimeError('Temporary Pod deletion not verified')


def deliver(url, token, desired, phase):
    from runpod.dinov3_study_worker import validated_url
    if (not isinstance(desired,dict) or set(desired)!={'key','nonce'} or desired['key']!=phase
            or not re.fullmatch('[0-9a-f]{32}',desired['nonce'])):
        raise ValueError('Unexpected input URL request')
    pin=input_pin(phase)
    req=urllib.request.Request(BASE+'/actions/artifacts/'+str(pin['artifact_id'])+'/zip',headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'User-Agent':'waferVLM/0.1'})
    try:
        with urllib.request.build_opener(NoRedirect).open(req,timeout=30):
            raise ValueError('Expected artifact redirect')
    except urllib.error.HTTPError as exc:
        if exc.code!=302: raise RuntimeError('Scoped artifact URL unavailable') from None
        signed=validated_url(exc.headers.get('Location',''))
    print('::add-mask::'+signed,flush=True)
    if request(url+'/input-url',token=token,method='POST',body={**desired,'url':signed})!={'status':'accepted'}:
        raise RuntimeError('Input URL delivery not acknowledged')


class Deadline(BaseException):
    pass


def interrupt(*_):
    raise Deadline('Deadline/cancellation; begin owned cleanup')


def launch(api):
    spec=preflight(api)
    run=spec['run_id'];phase=spec['phase'];p=plan()
    marker=os.environ.get('AUTHORIZATION_ARTIFACT_ID','')
    if not marker.isdigit(): raise ValueError('Durable create intent receipt required')
    stage('intent_receipt')
    meta=request(BASE+'/actions/artifacts/'+marker,token=os.environ['GH_TOKEN'])
    if (meta.get('name')!=AUTH+'-'+run or str(meta.get('workflow_run',{}).get('id'))!=run
            or meta.get('workflow_run',{}).get('head_sha')!=spec['controller_sha'] or meta.get('expired') is not False):
        raise ValueError('Create intent identity mismatch')
    save('intent-receipt.json',dict(artifact_id=int(marker),sha256=meta.get('digest'),run_id=run,controller_sha=spec['controller_sha']))
    code,digest=overlay(phase)
    token=secrets.token_urlsafe(32);print('::add-mask::'+token,flush=True)
    started=time.time();deadline=started+WORK_SECONDS
    state=dict(pod_id=None,started_epoch=started,rate=spec['cost']['gpu_hourly_usd'],
               prior_usd=spec['ledger']['usd'],prior_seconds=spec['ledger']['seconds'])
    save('provision.json',state)
    weights=json.loads((ROOT/'docs/evidence/dinov3-seed17-43716864164.json').read_text())['weights']
    env=dict(WAFER_MANAGED_BY=REPOSITORY,DINO_RUN_ID=run,DINO_AUTH=AUTH,DINO_RESULT_TOKEN=token,
             DINO_DEADLINE=str(deadline-60),DINO_GPU=spec['gpu'],DINO_WEIGHTS=json.dumps(weights),
             DINO_IMAGE=p['image'],DINO_PHASE=phase,DINO_CONTROLLER_SHA=spec['controller_sha'],
             DINO_STARTED=str(started),DINO_RATE=str(state['rate']),DINO_PRIOR_USD=str(state['prior_usd']),
             DINO_PRIOR_SECONDS=str(state['prior_seconds']),DINO_OVERLAY_SHA256=digest,DINO_PLAN_SHA256=spec['plan_sha256'],
             DINO_FROZEN_SHA256=spec['frozen_sha256'] or '',WAFERVLM_IMAGE_DIGEST=p['image'].split('@')[1])
    payload=dict(name=AUTH+'-'+run,image=p['image'],cloud='SECURE',disk=DISK_GB,
        gpu=dict(id=spec['gpu'],count=1,minRamPerGpu=32,minVcpuCountPerGpu=4,minCudaVersion='12.8'),
        dataCenterIds=spec['regions'],startSsh=False,ports=['8888/http'],entrypoint=['/bin/bash','-lc'],
        cmd=['exec /app/.venv/bin/python -u -c '+shlex.quote(code)],env=env)
    handlers={s:signal.signal(s,interrupt) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    signal.alarm(WORK_SECONDS)
    try:
        stage('create_pod')
        pod=api.call('/pods','POST',payload)  # One create only. Ambiguous failures go directly to reconciliation.
        pid=pod.get('id','')
        if not re.fullmatch('[A-Za-z0-9_-]{1,128}',pid): raise ValueError('Create outcome unknown')
        state['pod_id']=pid;save('provision.json',state)
        stage('verify_allocation')
        detail=api.call('/pods/'+pid)
        if (not owned(detail,run) or detail.get('gpu',{}).get('id')!=spec['gpu'] or detail.get('dataCenterId') not in spec['regions']
                or not finite(detail.get('cost'),.001) or detail['cost']>state['rate']+DISK_RATE):
            raise ValueError('Allocated Pod/GPU/region/rate mismatch')
        save('allocation.json',dict(pod_id=pid,gpu=spec['gpu'],data_center=detail['dataCenterId'],hourly_usd=detail['cost']))
        stage('worker_startup')
        url='https://'+pid+'-8888.proxy.runpod.net';seen=False;last=None;next_inventory=0;delivered=set()
        while time.time()<deadline-60:
            if time.time()>=next_inventory:
                detail=api.call('/pods/'+pid)
                if not owned(detail,run) or detail.get('status') in ('ERROR','EXITED','TERMINATED'):
                    raise ValueError('Worker stopped or ownership changed')
                next_inventory=time.time()+60
            try: result=request(url+'/result',token=token)
            except RuntimeError:
                if not seen and time.time()-started>900: raise TimeoutError('Worker startup exceeded 15 minutes')
                time.sleep(10);continue
            seen=True
            current=result.get('phase')
            if current!=last and isinstance(current,str) and re.fullmatch('[a-z0-9_-]{1,50}',current):
                print('DINO study phase: '+current,flush=True);last=current
                save('worker-status.json',{k:result[k] for k in ('status','phase','error_type','failed_phase') if k in result})
            desired=result.get('input_request')
            if desired and desired.get('nonce') not in delivered:
                stage('deliver_scoped_input_url')
                deliver(url,token,desired,phase);delivered.add(desired['nonce'])
            if result.get('status') in ('completed','rejected'):
                save('result.json',result)
                archive=result.get('archive')
                if not archive: raise ValueError('Worker returned no result archive')
                size=archive.get('size');sha=archive.get('sha256')
                if type(size) is not int or not 0<size<=24*1024**2 or not re.fullmatch('[a-f0-9]{64}',sha or ''):
                    raise ValueError('Invalid result archive metadata')
                req=urllib.request.Request(url+'/archive',headers={'Authorization':'Bearer '+token})
                with urllib.request.build_opener(NoRedirect).open(req,timeout=60) as response:
                    data=response.read(size+1)
                if len(data)!=size or hashlib.sha256(data).hexdigest()!=sha: raise ValueError('Result archive checksum mismatch')
                (OUT/'experiment.tar.gz').write_bytes(data)
                if result['status']!='completed': raise ValueError('DINO study incomplete; partial archive retained')
                from runpod.confirmation_verify import unpack
                with tempfile.TemporaryDirectory() as tmp:
                    root=unpack(OUT/'experiment.tar.gz',Path(tmp)/'results')
                    worker=json.loads((root/'worker.json').read_text())
                    if any(worker.get(k)!=spec[k] for k in ('controller_sha','image_source_sha','image','plan_sha256','overlay_sha256')):
                        raise ValueError('Worker provenance differs from launch')
                    accepted=verify_results(root,phase,spec)
                accepted['archive_sha256']=sha;save('acceptance.json',accepted)
                return
            time.sleep(5)
        raise TimeoutError('No complete study before deadline')
    except Exception as exc:
        save('error.json',error_record(exc))
        raise
    finally:
        signal.alarm(0)
        for s,h in handlers.items(): signal.signal(s,h)
        cleanup(api)


if __name__=='__main__':
    import sys
    try:
        {'preflight':preflight,'run':launch,'cleanup':cleanup}[sys.argv[1]](Runpod())
    except Exception as exc:
        detail=error_record(exc)
        save('error.json',detail)
        print('DINO study failed: '+json.dumps(detail),flush=True)
        raise SystemExit(1) from None
