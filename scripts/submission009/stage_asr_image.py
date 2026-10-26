"""Prepare a self-contained ASR image candidate on the no-card system disk."""
import datetime as dt
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import shutil
import subprocess

BASE=Path('/opt/guoxin-reviewed008')
TARGET=Path('/opt/guoxin-asr009')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    if Path('/dev/kfd').exists() or TARGET.exists():
        raise RuntimeError('Requires no-card mode and a new target')
    source=BASE/'projects/guoxin-qwen3-asr-bw-vllm'
    venv=BASE/'environments/asr'
    model=Path('/root/private_data/SothisAI/model/Aihub/Qwen3-ASR-1.7B/main/Qwen3-ASR-1.7B')
    assert (source/'tasks/qwen3-asr-bw-vllm/compat').is_dir() and (venv/'bin/python').exists()
    assert (model/'config.json').is_file()
    TARGET.mkdir()
    shutil.copytree(source,TARGET/'project')
    shutil.copytree(venv,TARGET/'venv',symlinks=True)
    # Only the established checkpoint directory is copied, never the parent cache.
    shutil.copytree(model,TARGET/'model',ignore=shutil.ignore_patterns('.cache','.git','__pycache__'))
    weights=[]
    for path in sorted((TARGET/'model').rglob('*')):
        if path.is_file():
            relative=path.relative_to(TARGET/'model')
            before=model/relative
            assert sha(before)==sha(path)
            weights.append({'path':relative.as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    python=TARGET/'venv/bin/python'
    code="import importlib.metadata as m,json; names=['torch','vllm','vllm-hcu','transformers','soundfile','regex','more-itertools']; out={};\nfor n in names:\n try:out[n]={'version':m.version(n),'path':str(m.distribution(n).locate_file(''))}\n except m.PackageNotFoundError:out[n]=None\nprint(json.dumps(out))"
    p=subprocess.run([str(python),'-B','-c',code],capture_output=True,text=True,timeout=60,check=True)
    metadata=json.loads(p.stdout)
    assert '+das' in metadata['torch']['version'] and '+das' in metadata['vllm']['version']
    record={'UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'target':str(TARGET),
            'source_project':str(source),'source_venv':str(venv),'source_model':str(model),
            'model_files':weights,'packages':metadata,'GPU_loaded':False,
            'image_saved':False,'clean_replay_verified':False,'submission_ready':False}
    (TARGET/'PREPARATION.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
