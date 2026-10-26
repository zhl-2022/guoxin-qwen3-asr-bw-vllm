"""Full-split ASR measurements against publisher-reported LibriSpeech numbers.

Only audio is sent to the model. References are used after inference for scoring.
Both raw and normalized predictions are preserved. The normalization implementation
is disclosed, not represented as an unpublished Qwen evaluator.
"""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edit_counts(reference, hypothesis):
    """Return deterministic S,D,I counts for a minimum edit alignment."""
    r, h = reference.split(), hypothesis.split()
    previous = [(j, 0, 0, j) for j in range(len(h)+1)]
    for i, rw in enumerate(r, 1):
        current = [(i, 0, i, 0)]
        for j, hw in enumerate(h, 1):
            cost, s, d, ins = previous[j-1]
            substitution = (cost+(rw!=hw), s+(rw!=hw), d, ins)
            cost, s, d, ins = previous[j]
            deletion = (cost+1, s, d+1, ins)
            cost, s, d, ins = current[j-1]
            insertion = (cost+1, s, d, ins+1)
            current.append(min((substitution, deletion, insertion), key=lambda x:x[0]))
        previous=current
    _, substitutions,deletions,insertions=previous[-1]
    return {'substitutions':substitutions,'deletions':deletions,'insertions':insertions,'reference_words':len(r)}


def parse_transcription(content):
    if not isinstance(content,str) or content.count('<asr_text>') != 1:
        raise ValueError('Expected exactly one publisher ASR text delimiter')
    header,text=content.split('<asr_text>',1)
    if not header.startswith('language '):
        raise ValueError('Missing publisher language header')
    return header.removeprefix('language ').strip(),text.strip()


def one(sample, data_root, endpoint, model, timeout):
    import soundfile as sf
    path=data_root/sample['audio']
    if digest(path)!=sample['audio_sha256']:
        raise ValueError('Audio hash mismatch: '+sample['id'])
    audio,sr=sf.read(path,dtype='float32')
    if sr!=16000 or audio.ndim!=1:
        raise ValueError('Unexpected LibriSpeech audio format')
    wav=io.BytesIO()
    sf.write(wav,audio,sr,format='WAV',subtype='PCM_16')
    payload={'model':model,'messages':[{'role':'user','content':[{
        'type':'audio_url','audio_url':{'url':'data:audio/wav;base64,'+base64.b64encode(wav.getvalue()).decode()}}]}],
        'temperature':0,'max_tokens':1024,'stream':False}
    body=json.dumps(payload).encode()
    request=urllib.request.Request(endpoint+'/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
    result={'id':sample['id'],'audio_sha256':sample['audio_sha256'],'duration_seconds':len(audio)/sr,
            'request_sha256':hashlib.sha256(body).hexdigest(),'request_settings':{'temperature':0,'max_tokens':1024,'language':None},
            'started_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    start=time.perf_counter()
    try:
        with urllib.request.urlopen(request,timeout=timeout) as response:
            raw=response.read(8*1024*1024)
        value=json.loads(raw)
        choice=value['choices'][0]
        if choice.get('finish_reason')!='stop':
            raise ValueError('Incomplete ASR generation: '+str(choice.get('finish_reason')))
        language,text=parse_transcription(choice['message']['content'])
        result.update({'status':'ok','language':language,'prediction':text,'raw_response':value})
    except Exception as exc:
        result.update({'status':'failed','error':str(exc)})
    result['wall_seconds']=time.perf_counter()-start
    return result


def score(samples,results):
    from whisper_normalizers import EnglishTextNormalizer
    norm=EnglishTextNormalizer()
    expected={s['id']:s for s in samples}
    actual={r['id']:r for r in results}
    if len(actual)!=len(results) or set(actual)!=set(expected):
        raise ValueError('Prediction IDs do not match the complete dataset')
    records=[]
    totals={'substitutions':0,'deletions':0,'insertions':0,'reference_words':0}
    failures=[]
    for uid,s in expected.items():
        r=actual[uid]
        if r['status']!='ok':
            failures.append(uid)
            continue
        if r['audio_sha256']!=s['audio_sha256']:
            raise ValueError('Prediction audio identity mismatch')
        ref=norm(s['reference']).strip()
        hyp=norm(r['prediction']).strip()
        counts=edit_counts(ref,hyp)
        for key in totals:totals[key]+=counts[key]
        records.append({'id':uid,'raw_reference':s['reference'],'raw_prediction':r['prediction'],
                        'normalized_reference':ref,'normalized_prediction':hyp,**counts})
    complete=not failures
    wer=100*sum(totals[k] for k in ('substitutions','deletions','insertions'))/totals['reference_words'] if totals['reference_words'] else None
    baseline=1.63
    limit=baseline*1.02
    return records,{'sample_count':len(samples),'completed':len(records),'failed_ids':failures,'complete':complete,
        'micro_word_error_rate_percent':wer,'edit_counts':totals,'publisher_baseline_percent':baseline,
        'relative_degradation_allowance':0.02,'limit_percent':limit,
        'numeric_comparison_pass':complete and wer is not None and wer<=limit,
        'normalization':'OpenAI Whisper EnglishTextNormalizer, exact vendored source hashes in execution manifest',
        'publisher_normalization_implementation_published':False,'official_acceptance':False,
        'dataset_scope':'complete LibriSpeech test-clean (2620 utterances)',
        'publisher_source':'https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/7278e1e70fe206f11671096ffdd38061171dd6e5/README.md'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:30002')
    p.add_argument('--model',default='qwen3-asr-1.7b')
    p.add_argument('--concurrency',type=int,default=4,choices=(1,2,4,8))
    p.add_argument('--request-timeout',type=int,default=180)
    p.add_argument('--max-seconds',type=int,default=7200)
    p.add_argument('--score-only',type=Path)
    args=p.parse_args()
    if not args.endpoint.startswith('http://127.0.0.1:'):
        raise ValueError('Only this instance loopback service is allowed')
    manifest=args.data/'data-manifest.json'
    meta=json.loads(manifest.read_text(encoding='utf-8'))
    samples=meta['samples']
    assert meta['split']=='test-clean' and len(samples)==2620 and len({s['id'] for s in samples})==2620
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    execution={'UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'dataset_manifest_sha256':digest(manifest),
               'argv':vars(args).copy(),'source_sha256':digest(Path(__file__)),
               'normalizer_sources':{f.name:digest(f) for f in (Path(__file__).parent/'whisper_normalizers').iterdir() if f.is_file()},
               'references_sent_to_model':False,'publisher_reference_is_not_our_score':True}
    execution['argv']={k:str(v) if isinstance(v,Path) else v for k,v in execution['argv'].items()}
    (args.output/'execution.json').write_text(json.dumps(execution,indent=2),encoding='utf-8')
    if args.score_only:
        results=[json.loads(s) for s in args.score_only.read_text(encoding='utf-8').splitlines()]
    else:
        results=[]
        with (args.output/'predictions.jsonl').open('x',encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            def bounded(s):
                if time.perf_counter()-started>args.max_seconds:
                    return {'id':s['id'],'status':'failed','error':'whole-run deadline reached'}
                return one(s,args.data,args.endpoint,args.model,args.request_timeout)
            futures={pool.submit(bounded,s):s for s in samples}
            for future in as_completed(futures):
                sample=futures[future]
                try:result=future.result()
                except Exception as exc:result={'id':sample['id'],'status':'failed','error':str(exc)}
                results.append(result)
                stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                stream.flush()
                if len(results)%50==0:print(json.dumps({'completed_responses':len(results),'failed':sum(r['status']!='ok' for r in results),'wall':time.perf_counter()-started}),flush=True)
    records,summary=score(samples,results)
    with (args.output/'scored-samples.jsonl').open('x',encoding='utf-8') as out:
        for row in records:out.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary['evaluation_wall_seconds']=time.perf_counter()-started
    summary['inference_audio_seconds']=sum(r.get('duration_seconds',0) for r in results)
    summary['submission_ready']=False
    (args.output/'verify_result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    raise SystemExit(0 if summary['numeric_comparison_pass'] else 2)


if __name__=='__main__':main()
