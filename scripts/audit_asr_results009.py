"""Independently recompute the downloaded BW ASR results using NLTK edit distance."""
import datetime as dt
import hashlib
import importlib.metadata as md
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'evidence/session009/asr001'
DATA=ROOT/'datasets/session009/librispeech'
sys.path.insert(0,str(ROOT/'scripts/submission009'))
from whisper_normalizers import EnglishTextNormalizer


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def main():
    # This installed NLTK module imports only stdlib operator/warnings. Loading it
    # directly avoids unrelated optional scipy/numpy imports in nltk.__init__.
    module_path=Path(md.distribution('nltk').locate_file('nltk/metrics/distance.py'))
    spec=importlib.util.spec_from_file_location('audit_nltk_distance',module_path)
    distance=importlib.util.module_from_spec(spec);spec.loader.exec_module(distance)
    samples=read(DATA/'data-manifest.json')['samples']
    predictions=rows(RESULT/'evaluation/predictions.jsonl')
    scored=rows(RESULT/'evaluation/scored-samples.jsonl')
    expected={x['id']:x for x in samples};actual={x['id']:x for x in predictions};scored_map={x['id']:x for x in scored}
    assert len(samples)==len(predictions)==len(scored)==len(actual)==len(expected)==2620
    assert set(actual)==set(expected)==set(scored_map)
    normalizer=EnglishTextNormalizer();error_count=word_count=0
    for uid,sample in expected.items():
        pred=actual[uid];declared=scored_map[uid]
        assert sha(DATA/sample['audio'])==sample['audio_sha256']==pred['audio_sha256']
        assert pred['status']=='ok'
        choice=pred['raw_response']['choices'][0]
        assert choice['finish_reason']=='stop'
        content=choice['message']['content']
        assert content.count('<asr_text>')==1 and content.startswith('language ')
        hypothesis=content.split('<asr_text>')[1].strip()
        assert hypothesis==pred['prediction']==declared['raw_prediction']
        assert sample['reference']==declared['raw_reference']
        reference=normalizer(sample['reference']).strip();normalized=normalizer(hypothesis).strip()
        assert reference==declared['normalized_reference'] and normalized==declared['normalized_prediction']
        error=distance.edit_distance(reference.split(),normalized.split(),transpositions=False)
        assert error==sum(declared[k] for k in ('substitutions','deletions','insertions'))
        assert len(reference.split())==declared['reference_words']
        error_count+=error;word_count+=len(reference.split())
    wer=100*error_count/word_count
    recorded=read(RESULT/'evaluation/verify_result.json')
    assert math.isclose(wer,recorded['micro_word_error_rate_percent'],abs_tol=1e-12)
    assert recorded['numeric_comparison_pass']==(wer<=1.63*1.02)
    bench=read(RESULT/'bench/summary.json');selected=bench['protocol']['selected_ids']
    independent_selection=[s['id'] for s in sorted(samples,key=lambda r:hashlib.sha256(('asr009-bench-v1:'+r['id']).encode()).digest())[:24]]
    assert selected==independent_selection
    requests=0
    for row in bench['rounds']:
        calls=rows(RESULT/f"bench/c{row['concurrency']}-round{row['round']}.jsonl")
        assert len(calls)==24 and [r['id'] for r in calls]==selected
        assert all(r['status']=='ok' and r['raw_response']['choices'][0]['finish_reason']=='stop' for r in calls)
        audio=sum(r['duration_seconds'] for r in calls)
        assert math.isclose(audio,row['audio_seconds_successful'],abs_tol=1e-9)
        assert math.isclose(audio/row['wall_seconds'],row['real_time_factor_inverse'],abs_tol=1e-9)
        assert math.isclose(statistics.median(r['wall_seconds'] for r in calls),row['latency_p50_seconds'],abs_tol=1e-9)
        requests+=len(calls)
    start=read(RESULT/'run-start.json');final=read(RESULT/'run-final.json')
    assert all(code==0 for code in final['phases'].values()) and final['service_stopped']
    for name,digest in start['source_files'].items():assert sha(ROOT/'scripts/submission009'/name)==digest,name
    result={'UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'passed':True,
        'archive_sha256':sha(ROOT/'dist/session009-asr001-results.zip'),
        'prediction_count':2620,'original_audio_hashes_checked':2620,'failed_requests':0,
        'word_errors_independently_recomputed':error_count,'reference_words':word_count,'wer_percent':wer,
        'publisher_baseline_percent':1.63,'relative_difference_percent':(wer/1.63-1)*100,
        'threshold_percent':1.6626,'numeric_comparison_pass':wer<=1.6626,
        'independent_distance_implementation':'NLTK standalone distance.py','nltk_version':md.version('nltk'),
        'distance_source_sha256':sha(module_path),'performance_requests_verified':requests,
        'runtime_sources_verified':len(start['source_files']),'service_stopped':True,
        'normalization_matches_all_saved_rows':True,'publisher_normalizer_is_unpublished':True,
        'official_acceptance':False,'submitted':False}
    with (ROOT/'evidence/session009/asr-independent-audit001.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
