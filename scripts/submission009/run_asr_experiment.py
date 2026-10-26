"""Bounded ASR full-split evaluation followed by two performance rounds and cleanup."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile

ROOT=Path(__file__).resolve().parent
IMAGE=Path('/opt/guoxin-asr009')
TASK=IMAGE/'project/tasks/qwen3-asr-bw-vllm'
DATA=Path('/root/private_data/guoxin-session009/librispeech-final')
OUTPUT=Path('/root/private_data/guoxin-session009/asr001')
PYTHON=IMAGE/'venv/bin/python'


def step(args,name,timeout,env=None):
    with (OUTPUT/(name+'.log')).open('xb') as f:
        p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env)
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            code=124
    print(json.dumps({'phase':name,'exit':code}),flush=True)
    return code


def main():
    if not Path('/dev/kfd').exists():raise RuntimeError('GPU device is required; no-card mode is preparation only')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    run_id=uuid.uuid4().hex
    record={'UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'run_id':run_id,'source_files':{},
            'image_root':str(IMAGE),'dataset_manifest_sha256':hashlib.sha256((DATA/'data-manifest.json').read_bytes()).hexdigest(),
            'submitted':False,'phases':{}}
    for p in ROOT.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            record['source_files'][p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUTPUT/'run-start.json').write_text(json.dumps(record,indent=2))
    started=time.monotonic()
    try:
        preflight="import torch,json,importlib.metadata as m; assert torch.cuda.is_available(); a=torch.ones((16,16),device='cuda',dtype=torch.bfloat16); b=a@a; torch.cuda.synchronize(); assert float(b[0,0])==16; print(json.dumps({'name':torch.cuda.get_device_name(0),'torch':torch.__version__,'hip':torch.version.hip,'packages':{n:m.version(n) for n in ['vllm','vllm-hcu','transformers','soundfile','regex','more-itertools']},'bf16_matmul':True}))"
        record['phases']['device']=step([str(PYTHON),'-B','-c',preflight],'environment',120)
        if record['phases']['device']:raise RuntimeError('Device preflight failed')
        if shutil.which('hy-smi'):step(['hy-smi'],'hy-smi-before',30)
        # The canonical supervisor independently bounds the server at 9900 seconds.
        record['phases']['launch']=step(['bash',str(ROOT/'run_asr.sh'),'--run-id',run_id],'launch',1000)
        if record['phases']['launch']:raise RuntimeError('Service launch failed')
        record['phases']['evaluation']=step([str(PYTHON),'-B',str(ROOT/'evaluate_asr.py'),
            '--data',str(DATA),'--output',str(OUTPUT/'evaluation'),'--concurrency','4','--max-seconds','7200'],
            'evaluation',7500)
        record['phases']['benchmark']=step([str(PYTHON),'-B',str(ROOT/'benchmark_asr.py'),
            '--data',str(DATA),'--output',str(OUTPUT/'bench')],'benchmark',1000)
    except Exception as e:
        record['error']=str(e)
    finally:
        record['phases']['stop']=step([str(PYTHON),str(IMAGE/'project/scripts/recon2reason_runtime.py'),
            'stop','--task-dir',str(TASK),'--run-id',run_id],'stop',90)
        try:
            with urllib.request.urlopen('http://127.0.0.1:30002/health',timeout=3):record['service_stopped']=False
        except (OSError,urllib.error.URLError):record['service_stopped']=True
        runtime=TASK/'runtime'
        if runtime.is_dir():
            # This is the new image task runtime, not a shared historical directory.
            shutil.copytree(runtime,OUTPUT/'runtime',ignore=shutil.ignore_patterns('audio-venv','__pycache__'))
        if shutil.which('hy-smi'):step(['hy-smi'],'hy-smi-after',30)
        record['wall_seconds']=time.monotonic()-started
        record['ended_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        result_path=OUTPUT/'evaluation/verify_result.json'
        record['accuracy_result']=json.loads(result_path.read_text()) if result_path.exists() else None
        record['official_acceptance']=False
        record['submission_ready']=False
        (OUTPUT/'run-final.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
        with zipfile.ZipFile('/root/session009-asr001-results.zip','x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(OUTPUT.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(OUTPUT).as_posix())
        print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
