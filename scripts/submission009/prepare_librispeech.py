"""Download the complete publisher-referenced LibriSpeech test-clean split.

CPU only; original OpenSLR MD5 is checked before safe extraction and a complete
per-file SHA256 manifest is created. Never executes code from the archive.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
import urllib.request

MD5 = '32fa31d27d2e1cad72775fee3f4849a9'
URLS = ['https://www.openslr.org/resources/12/test-clean.tar.gz',
        'https://openslr.magicdatatech.com/resources/12/test-clean.tar.gz',
        'https://openslr.trmal.net/resources/12/test-clean.tar.gz']


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    final = root/'data-manifest.json'
    if final.exists():
        raise SystemExit('A completed dataset manifest already exists; preserve this run')
    archive = root/'test-clean.tar.gz'
    if not archive.exists():
        errors=[]
        for i,url in enumerate(URLS):
            partial=root/f'test-clean.attempt{i}.partial'
            if partial.exists():
                continue
            try:
                started=time.monotonic()
                with urllib.request.urlopen(url, timeout=45) as r, partial.open('xb') as out:
                    length=0
                    last=0
                    while True:
                        block=r.read(1024*1024)
                        if not block:break
                        length+=len(block)
                        if length>500*1024*1024 or time.monotonic()-started>1800:
                            raise RuntimeError('Download size/time bound exceeded')
                        out.write(block)
                        if length-last>=32*1024*1024:
                            print(json.dumps({'downloaded_bytes':length,'source':url}),flush=True)
                            last=length
                if digest(partial,'md5')!=MD5:
                    raise ValueError('OpenSLR MD5 mismatch')
                partial.rename(archive)
                break
            except Exception as e:
                errors.append({'url':url,'error':str(e)})
                print(json.dumps(errors[-1]),flush=True)
        (root/'download-attempts.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    assert archive.is_file() and digest(archive,'md5')==MD5
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        for member in members:
            q=PurePosixPath(member.name)
            if q.is_absolute() or '..' in q.parts or not (member.isdir() or member.isfile()):
                raise ValueError('Unsafe archive member: '+member.name)
            if ':' in member.name or '\\' in member.name:
                raise ValueError('Invalid archive member')
        for member in members:
            target=root/member.name
            if member.isdir():
                target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    raise FileExistsError(target)
                with tar.extractfile(member) as source,target.open('xb') as output:
                    for block in iter(lambda:source.read(1024*1024),b''):
                        output.write(block)
    samples=[]
    for txt in sorted((root/'LibriSpeech/test-clean').rglob('*.trans.txt')):
        for line in txt.read_text(encoding='utf-8').splitlines():
            uid,text=line.split(' ',1)
            audio=txt.parent/(uid+'.flac')
            samples.append({'id':uid,'audio':audio.relative_to(root).as_posix(),
                            'reference':text,'audio_sha256':digest(audio)})
    samples.sort(key=lambda r:r['id'])
    assert len(samples)==2620 and len({r['id'] for r in samples})==2620
    assert len(list((root/'LibriSpeech/test-clean').rglob('*.flac')))==2620
    result={'UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'dataset':'LibriSpeech',
            'split':'test-clean','selection':'complete split; no sample selection',
            'license':'CC BY 4.0','publisher_data_page':'https://www.openslr.org/12/',
            'md5_source':'https://www.openslr.org/resources/12/md5sum.txt',
            'archive_md5':MD5,'archive_sha256':digest(archive),'samples':samples,
            'model_predictions_generated':False}
    with final.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'ready','samples':len(samples),'manifest_sha256':digest(final)}),flush=True)


if __name__=='__main__':main()
