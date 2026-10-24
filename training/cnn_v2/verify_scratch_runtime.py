"""Execute native Linux C accuracy and streaming checks (standard library only)."""
import csv, hashlib, io, json, math, pathlib, platform, re, statistics, subprocess, time, wave
root = pathlib.Path(__file__).resolve().parent
out = root/'c-results'; out.mkdir(exist_ok=True)
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
commands = []
def save(name, obj):
    (out/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
def run(cmd, **kw):
    commands.append(cmd)
    return subprocess.check_output(cmd, cwd=root, timeout=180, **kw)
def build(cmd, name):
    commands.append(cmd)
    with (out/name).open('w') as log:
        subprocess.run(cmd, cwd=root, check=True, stdout=log, stderr=subprocess.STDOUT)
def score(rows):
    m = [[0]*5 for _ in range(5)]
    for r in rows: m[r['truth']][r['prediction']] += 1
    return dict(correct=sum(m[i][i] for i in range(5)), total=len(rows),
                accuracy=sum(m[i][i] for i in range(5))/len(rows), confusion_matrix=m,
                per_class_recall={name:m[i][i]/sum(m[i]) if sum(m[i]) else None for i,name in enumerate(labels)})
def c_rows(binary, inputs, stem):
    paths = [r['path'] for r in inputs]
    start = time.perf_counter(); text = run([str(binary), *paths], text=True)
    elapsed = time.perf_counter()-start
    (out/(stem+'.tsv')).write_text(text)
    rows = list(csv.DictReader(io.StringIO(text), delimiter='\t')); assert len(rows)==len(inputs)
    for row, source in zip(rows, inputs): assert row['file']==source['path']
    return rows, elapsed
manifest = read(root/'manifest.json'); labels=manifest['labels']
for name, h in read(root/'runtime-inputs.json')['files'].items(): assert sha(root/name)==h, name
official = read(root/'esc-manifest.json'); python = read(root/'python-results.json')
pooled = []; external = []; max_delta = 0.
for model in manifest['models']:
    name = model['model']; d = root/'models'/name; binary=d/'cnn_stream'
    build(['cc','-std=c11','-O2','-Wall','-Wextra','-pedantic','-I',str(d),str(d/'cnn_stream_v2.c'),'cnn_cli.c','-lm','-o',str(binary)], name+'-build.log')
    reference = next(r['int8']['predictions'] for r in python['models'] if r['model']==name)
    rows, elapsed = c_rows(binary, manifest['inputs'],name+'-external')
    records=[]
    for row, source, ref in zip(rows,manifest['inputs'],reference):
        assert source['id']==ref['id']; probs=[float(row['p'+str(i)]) for i in range(5)]
        delta=max(abs(p-q) for p,q in zip(probs,ref['probabilities'])); max_delta=max(delta,max_delta)
        assert delta<.0001 and row['prediction']==ref['prediction'], (name,source['id'],delta)
        records.append(dict(id=source['id'],truth=labels.index(source['reference_label']),prediction=labels.index(row['prediction']),probabilities=probs,python_max_abs_delta=delta))
    external.append(dict(model=name,model_sha256=model['model_sha256'],binary_sha256=sha(binary),elapsed_seconds=elapsed,**score(records),records=records))
    if name.startswith('fold_'):
        holdout=[r for r in official['inputs'] if r['fold']==int(name[-1])]
        refs={r['filename']:r for r in official['references'][name]}
        rows, elapsed=c_rows(binary,holdout,name+'-official-holdout')
        for row, source in zip(rows, holdout):
            ref=refs[source['filename']]; probs=[float(row['p'+str(i)]) for i in range(5)]
            delta=max(abs(p-q) for p,q in zip(probs,ref['probabilities'])); max_delta=max(delta,max_delta)
            assert delta<.0001 and labels.index(row['prediction'])==ref['prediction'], (name,source['filename'],delta)
            pooled.append(dict(model=name,filename=source['filename'],truth=source['truth'],prediction=ref['prediction'],probabilities=probs,python_max_abs_delta=delta))
    print('C VERIFIED',name,external[-1]['correct'],'/65',flush=True)

app=root/'audiodetect'
src=root/'app-source'
build(['cc','-DAED_USE_CNN','-std=c11','-O2','-Wall','-Wextra','-pedantic',*[str(src/f) for f in ['audiodetect_main.c','audio_classifier.c','cnn_stream_v2.c','wav_reader.c','keyword_detect.c','alarm_report.c']],'-lm','-o',str(app)],'app-build.log')
app_records=[]
for record, source in zip(external[-1]['records'],manifest['inputs']):
    text=run([str(app),source['path'],'--debug'],text=True)
    (out/('app-'+source['id']+'.log')).write_text(text)
    values=[float(re.search(r'^  '+name+r': ([0-9.eE+-]+)$',text,re.M)[1]) for name in labels]
    assert max(abs(p-q) for p,q in zip(values,record['probabilities']))<.000001
    assert re.search(r'^Result: (\w+)$',text,re.M)[1]==labels[record['prediction']]
    app_records.append(dict(id=source['id'],probabilities=values))

parity=root/'parity'; binary=root/'models/final_all_esc50/cnn_stream'; cases=read(parity/'references.json')
checks=[]; full=None
for name, ref in cases.items():
    outputs=[]; traces=[]
    for chunk in (1,160,997,4096):
        trace=out/f'{name}-{chunk}.tsv'
        text=run([str(binary),'--chunk',str(chunk),'--trace',str(trace),str(parity/(name+'.wav'))],text=True)
        outputs.append(text); traces.append(trace.read_bytes())
        rows=list(csv.DictReader(io.StringIO(trace.read_text()),delimiter='\t'))
        assert len(rows)==len(ref['LOGITS'])
        delta=max(abs(float(row['l'+str(i)])-v[i]) for row,v in zip(rows,ref['LOGITS']) for i in range(5))
        assert delta<.0003,(name,chunk,delta)
        state=None; edelta=0.
        for row, v in zip(rows,ref['LOGITS']):
            state=v[:] if state is None else [.98*p+.02*q for p,q in zip(state,v)]
            edelta=max(edelta,max(abs(float(row['e'+str(i)])-state[i]) for i in range(5)))
        assert edelta<.0003,(name,chunk,edelta)
        result=list(csv.DictReader(io.StringIO(text),delimiter='\t'))[0]
        assert max(abs(float(result['p'+str(i)])-ref['PROB'][i]) for i in range(5))<.0001
        emit=[int(r['end_sample']) for r in rows if r['emit']=='1']
        assert all(b-a==1600 for a,b in zip(emit,emit[1:]))
        checks.append(dict(input=name,chunk=chunk,frames=len(rows),max_logit_delta=delta,max_ema_delta=edelta))
    assert len(set(traces))==len(set(outputs))==1
    if name=='input':full=rows
    if name=='prefix':assert full[:len(rows)]==rows
with wave.open(str(parity/'input.wav'),'rb') as w: pcm=w.readframes(w.getnframes())
raw=run([str(binary),'--stdin-raw','--chunk','997'],input=pcm).decode()
wav=run([str(binary),'--live',str(parity/'input.wav')],text=True)
assert [json.loads(x) for x in raw.splitlines()]==[json.loads(x) for x in wav.splitlines()]
(out/'stdin-raw.jsonl').write_text(raw)
san=root/'cnn_stream_sanitized'; d=root/'models/final_all_esc50'
build(['cc','-std=c11','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer','-I',str(d),str(d/'cnn_stream_v2.c'),'cnn_cli.c','-lm','-o',str(san)],'sanitizer-build.log')
with (out/'sanitizer-run.log').open('w') as log:
    cmd=[str(san),'--chunk','997',*[str(parity/(n+'.wav')) for n in ('input','short','silence')]]; commands.append(cmd)
    subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=30)
save('summary.json',dict(status='PASS',runtime='native Linux x86_64 shared C core; not a full OpenVela system simulator boot',
     platform=platform.platform(),compiler=run(['cc','--version'],text=True),
     official_cv={**score(pooled), 'records':pooled},external=external,max_python_c_probability_delta=max_delta,
     app_stream_predictions_equal=65,streaming_checks=checks,callback_partition_invariant=True,prefix_causality=True,
     raw_stdin_matches_wav=True,sanitizers='ASan + UBSan PASS',app_binary_sha256=sha(app),
     final_model_sha256=manifest['models'][-1]['model_sha256'],
     external_note='65 previously reviewed development recordings; no model selection in this run; not a new blind test'))
save('commands.json',commands)
print('ALL LINUX C CHECKS PASS; CV',score(pooled)['correct'],'/390; external',external[-1]['correct'],'/65; delta',max_delta,flush=True)
