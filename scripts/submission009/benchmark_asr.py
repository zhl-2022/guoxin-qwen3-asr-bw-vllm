"""Two rounds of reproducible ASR throughput/latency, with all requests retained."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import time
from evaluate_asr import one,digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:30002')
    p.add_argument('--model',default='qwen3-asr-1.7b')
    args=p.parse_args()
    if not args.endpoint.startswith('http://127.0.0.1:'):raise ValueError('Loopback only')
    samples=json.loads((args.data/'data-manifest.json').read_text())['samples']
    # Select by fixed SHA-256 order, independent of predictions and request latency.
    selected=sorted(samples,key=lambda r:hashlib.sha256(('asr009-bench-v1:'+r['id']).encode()).digest())[:24]
    args.output.mkdir(parents=True,exist_ok=False)
    meta={'UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'selected_ids':[r['id'] for r in selected],
          'dataset_sha256':digest(args.data/'data-manifest.json'),'source_sha256':digest(Path(__file__)),
          'method':'fixed 24 complete utterances, c1/c4, two rounds, same greedy 1024 settings; warmup excluded',
          'official_acceptance':False}
    (args.output/'protocol.json').write_text(json.dumps(meta,indent=2))
    warm=one(selected[0],args.data,args.endpoint,args.model,180)
    (args.output/'warmup.json').write_text(json.dumps(warm,ensure_ascii=False,indent=2))
    if warm['status']!='ok':raise RuntimeError('Warmup failed')
    results=[]
    for concurrency in (1,4):
        for round_id in (1,2):
            start=time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                rows=list(pool.map(lambda s:one(s,args.data,args.endpoint,args.model,180),selected))
            wall=time.perf_counter()-start
            with (args.output/f'c{concurrency}-round{round_id}.jsonl').open('x',encoding='utf-8') as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            success=[r for r in rows if r['status']=='ok']
            latencies=sorted(r['wall_seconds'] for r in success)
            audio=sum(r['duration_seconds'] for r in success)
            summary={'concurrency':concurrency,'round':round_id,'requests':len(rows),'success':len(success),
                    'failed':len(rows)-len(success),'wall_seconds':wall,'audio_seconds_successful':audio,
                    'real_time_factor_inverse':audio/wall,'request_per_second':len(success)/wall,
                    'latency_p50_seconds':statistics.median(latencies) if latencies else None,
                    'latency_p95_nearest_rank_seconds':latencies[min(len(latencies)-1,int((len(latencies)*95+99)//100)-1)] if latencies else None}
            results.append(summary)
            print(json.dumps(summary),flush=True)
    (args.output/'summary.json').write_text(json.dumps({'protocol':meta,'rounds':results,'all_successful':all(r['failed']==0 for r in results)},indent=2))
    raise SystemExit(0 if all(r['failed']==0 for r in results) else 2)


if __name__=='__main__':main()
