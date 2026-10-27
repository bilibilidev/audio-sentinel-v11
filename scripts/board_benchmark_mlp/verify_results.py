"""Offline audit of published physical Tiny MLP results; no board/audio needed."""
import csv,hashlib,json,math,re
from pathlib import Path

repo=Path(__file__).resolve().parents[2]
out=repo/'results/board-mlp-20261027'
read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary=read('summary.json');manifest=read('input-manifest.json');labels=manifest['labels']
model=repo/'model/mlp-optimized-20261024'
deployment=json.loads((model/'deployment.json').read_text())
assert digest(model/'final_all_esc50.npz')==summary['model_sha256']==deployment['model_sha256']
for name,expected in deployment['header_sha256'].items():assert digest(repo/'app/audiodetect'/name)==expected,name
def pct(values,p):
    values=sorted(values);i=(len(values)-1)*p;lo=math.floor(i);hi=math.ceil(i)
    return values[lo]+(values[hi]-values[lo])*(i-lo)
def check_times(value):
    raw=value['raw_values'];assert len(raw)==value['count']
    assert math.isclose(sum(raw)/len(raw),value['mean'])
    assert math.isclose(pct(raw,.95),value['p95'])
for name,inputs in [('clean65.json',manifest['records']),('nand-clean65.json',manifest['records']),
                    *[('noise'+str(n)+'.json',manifest['noise'][str(n)]) for n in [20,10,5]]]:
    result=read(name);rows=result['records'];assert len(rows)==result['total']==65
    assert len({r['id'] for r in rows})==65
    matrix=[[0]*5 for _ in labels]
    for r,source in zip(rows,inputs):
        assert r['id']==source['id'] and r['input_sha256']==source['sha256'] and r['samples']==source['samples']
        assert r['truth']==source['reference_label'] and r['model_sha256']==summary['model_sha256']
        assert r['correct']==(r['truth']==r['prediction'])
        assert math.isclose(r['audio_seconds'],r['samples']/16000)
        assert set(r['probabilities'])==set(labels) and math.isclose(sum(r['probabilities'].values()),1,abs_tol=1e-6)
        assert r['prediction']==max(labels,key=r['probabilities'].get)
        assert r['alerted']==(r['prediction']!='background' and r['confidence']>.8)
        matrix[labels.index(r['truth'])][labels.index(r['prediction'])]+=1
        if 'host_reference_probabilities' in source:
            delta=max(abs(r['probabilities'][label]-source['host_reference_probabilities'][i]) for i,label in enumerate(labels))
            assert math.isclose(delta,r['max_probability_abs_delta_vs_linux'],abs_tol=1e-15)
            assert r['linux_prediction_match']==(r['prediction']==source['host_reference_prediction'])
    assert matrix==result['confusion_matrix']
    assert sum(r['correct'] for r in rows)==result['correct'] and math.isclose(result['accuracy'],result['correct']/65)
    for label in labels:
        truth=[r for r in rows if r['truth']==label];negative=[r for r in rows if r['truth']!=label];p=result['per_class'][label]
        assert p['correct']==sum(r['correct'] for r in truth) and p['total']==len(truth)
        assert p['false_positives']==sum(r['prediction']==label for r in negative)
        assert math.isclose(p['one_vs_rest_fpr'],p['false_positives']/len(negative))
        assert p['correct_alarm_count']==sum(r['correct'] and r['alerted'] for r in truth)
    check_times(result['board_command_seconds'])
    assert result['board_command_seconds']['raw_values']==[r['board_command_seconds'] for r in rows]
clean=read('clean65.json');nand=read('nand-clean65.json')
assert [r['prediction'] for r in clean['records']]==[r['prediction'] for r in nand['records']]
assert clean['correct']==57 and clean['per_class']['baby_cry']['correct']==17
assert summary['classification']['accuracy']==clean['accuracy'] and summary['classification']['per_class']==clean['per_class']
assert clean['per_class']['baby_cry']['correct_alarm_count']==4 and clean['linux_prediction_matches']==65
assert len(list(csv.DictReader((out/'predictions.csv').open(encoding='utf-8'))))==65
assert len(list(csv.DictReader((out/'errors.csv').open(encoding='utf-8'))))==65-clean['correct']
lat=read('latency.json');check_times(lat['board_command_seconds']);assert lat['warmups']==3 and lat['board_command_seconds']['count']==30
assert lat['filesystem']=='tmpfs RAM' and nand['filesystem']=='YAFFS NAND'
capture=read('capture.json');assert len(capture['records'])==5 and capture['passed'] and not capture['microphone_accuracy_measured']
for r in capture['records']:assert (r['rate'],r['channels'],r['sample_width'],r['frames'])==(16000,1,2,48000)
for k in ['capture_seconds','inference_seconds','sum_of_separate_command_seconds','host_sequential_wall_seconds']:check_times(capture[k])
aux=read('auxiliary.json');assert len(aux['keywords'])==11 and all(x['passed'] for x in aux['keywords'])
assert all(x['passed'] for x in aux['input_validation']) and len(aux['input_validation'])==8
assert not aux['network']['http_success'] and not aux['network']['mqtt_success']
soak=read('soak.json');assert soak['iterations']==200 and len(soak['rows'])==200 and all(x['passed'] for x in soak['rows'])
assert not summary['system_acceptance_pass'] and not summary['nand_write_acceptance'] and not summary['one_hour_stability_tested']
assert summary['power_watts'] is None and summary['energy_joules'] is None
for row in read('publication-processing.json')['files']:assert digest(out/row['public'])==row['public_sha256'],row['public']
for line in (out/'SHA256SUMS').read_text().splitlines():
    expected,name=line.split('  ',1);p=Path(name)
    assert not p.is_absolute() and '..' not in p.parts and digest(out/p)==expected,name
assert not any(p.suffix.lower() in {'.wav','.flac','.mp3','.img','.elf','.pem','.key','.npz'} for p in out.rglob('*') if p.is_file())
for p in out.rglob('*'):
    if p.is_file():assert not re.search(r'C:[\\/]+Users[\\/]+libin|/home/sentinel',p.read_text(encoding='utf-8'),re.I),p
print('PASS: physical MLP clean/NAND/noise65, FPR/alarm counts, timings, microphone5, short loop200, failure boundaries, model/header and evidence hashes')
