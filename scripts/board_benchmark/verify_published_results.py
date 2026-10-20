"""Audit published summaries against per-record measurements and model hashes."""
import csv,hashlib,json,math
from pathlib import Path

root=Path(__file__).resolve().parents[2]
out=root/'results/board-20261020'
def read(name):return json.loads((out/name).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pct(values,p):
    a=sorted(values);x=(len(a)-1)*p;lo=math.floor(x);hi=math.ceil(x)
    return a[lo]+(a[hi]-a[lo])*(x-lo)
r=read('board-wav-results.json');m=read('input-manifest.json');rows=r['records']
deployment=json.loads((root/'model/cnn-v2/deployment.json').read_text(encoding='utf-8'))
assert sha(root/'model/cnn-v2/final_all_esc50.npz')==r['model_sha256']==deployment['model_sha256']
for name,digest in deployment['header_sha256'].items():assert sha(root/'app/audiodetect'/name)==digest
assert len(rows)==len(m['records'])==65
assert len({v['id'] for v in rows})==65
matrix=[[0]*5 for _ in range(5)]
for row,source in zip(rows,m['records']):
    assert row['id']==source['id'] and row['reference_label']==source['reference_label']
    assert row['input_sha256']==source['sha256'] and row['samples']==source['samples']
    assert row['model_sha256']==r['model_sha256']
    assert row['correct']==(row['prediction']==row['reference_label'])
    assert row['prediction']==source['host_reference_prediction']
    assert math.isclose(row['audio_seconds'],row['samples']/16000)
    assert math.isclose(row['real_time_factor'],row['board_command_seconds']/row['audio_seconds'])
    matrix[r['labels'].index(row['reference_label'])][r['labels'].index(row['prediction'])]+=1
assert matrix==r['confusion_matrix_truth_rows_prediction_columns']
assert sum(v['correct'] for v in rows)==r['correct']==52
assert r['accuracy']==52/65 and r['host_prediction_matches']==65
values=[v['board_command_seconds'] for v in rows]
assert math.isclose(sum(values)/65,r['board_command_seconds']['mean'])
assert math.isclose(pct(values,.95),r['board_command_seconds']['p95'])
assert math.isclose(sum(values),r['total_board_command_seconds'])
csvrows=list(csv.DictReader((out/'predictions.csv').open(encoding='utf-8')))
assert len(csvrows)==65 and [v['id'] for v in csvrows]==[v['id'] for v in rows]
for name,key in [('latency-memory.json','fixed_3s_inference_seconds'),('fixed-latency-recheck.json',None)]:
    value=read(name);value=value[key] if key else value
    raw=value['raw_seconds'];assert len(raw)==value['count']==20
    assert math.isclose(sum(raw)/len(raw),value['mean']) and math.isclose(pct(raw,.95),value['p95'])
a=read('cnn-host.json');b=read('panns-host.json');c=read('resource-comparison.json')
assert a['input_recording_sha256']==b['input_recording_sha256']==rows[0]['input_sha256']
assert a['torch_threads']==b['torch_threads']==1
assert a['profile']['learned_parameters']==14885
assert a['profile']['int8_learned_parameter_bytes']+a['profile']['quantization_scale_bytes']==14933
assert b['profile']['classifier_cached_external_predictions_match']
for x,y in zip(a['measurements'],b['measurements']):
    assert x['audio_seconds']==y['audio_seconds'] and x['pcm16_sha256']==y['pcm16_sha256']
    for measurement in (x,y):
        for key in ('prepared_input_inference','pcm_preprocess_and_inference'):
            t=measurement[key];raw=t['raw_seconds']
            assert len(raw)==t['count']==10 and math.isclose(sum(raw)/10,t['mean'])
assert math.isclose(c['learned_parameter_ratio_panns_to_cnn'],b['profile']['learned_parameters']/14885)
baselines=json.loads((root/'results/cnn-v2/baselines.json').read_text(encoding='utf-8'))
panns=next(v for v in baselines['baselines'] if v['baseline']=='PANNs_Cnn6_linearSVM')
assert panns['external_final']['correct']==58 and panns['held_out']['correct']==367
assert read('power-status.json')['power_watts'] is None
assert not read('latency-memory.json')['capture_chain_validated']
for line in (out/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split('  ',1);assert '..' not in Path(name).parts and not Path(name).is_absolute()
    assert sha(out/name)==digest,name
print('PASS: 65 physical-board predictions, timing summaries, input/model identity, host comparison and evidence checksums')
