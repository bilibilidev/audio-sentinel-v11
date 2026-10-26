"""Actual Linux C acceptance; Python standard library only, no package installation."""
import argparse,csv,hashlib,io,json,os,pathlib,platform,re,shutil,statistics,subprocess,time,wave
p=argparse.ArgumentParser(description=__doc__)
for n in ('repo','inputs','reference','exports','study','output'):p.add_argument('--'+n,type=pathlib.Path,required=True)
a=p.parse_args();R=a.repo.resolve();I=a.inputs.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=True)
assert platform.system()=='Linux';assert not (O/'results.json').exists()
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
commands=[]
def run(cmd,log=None,**kw):
    cmd=[str(v) for v in cmd];commands.append(cmd);result=subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kw)
    if log:log.write_bytes(result.stdout+b'\nSTDERR:\n'+result.stderr)
    return result.stdout.decode('utf-8')
def build(cmd,log):return run(cmd,log)
def score(rows):
    m=[[0]*5 for _ in range(5)]
    for v in rows:m[v['truth']][v['prediction']]+=1
    correct=sum(m[i][i] for i in range(5));recall=[m[i][i]/sum(m[i]) if sum(m[i]) else None for i in range(5)]
    return dict(correct=correct,total=len(rows),accuracy=correct/len(rows),macro_recall=statistics.mean(v for v in recall if v is not None),confusion_matrix=m,per_class_recall=dict(zip(labels,recall)),records=rows)
def tsv(binary,rows,dest):
    for row in rows:assert sha(I/row['path'])==row['sha256']
    text=run([binary,*[r['path'] for r in rows]],cwd=I);dest.write_text(text,encoding='utf-8');actual=list(csv.DictReader(io.StringIO(text),delimiter='\t'));assert len(actual)==len(rows)
    records=[]
    for row,v in zip(rows,actual):
        assert v['file']==row['path'];prob=[float(v['p'+str(i)]) for i in range(5)];pred=labels.index(v['prediction']);assert pred==max(range(5),key=prob.__getitem__)
        records.append(dict(id=row['id'],filename=row.get('filename'),truth=row.get('truth',labels.index(row['reference_label']) if 'reference_label' in row else None),prediction=pred,probabilities=prob,input_sha256=row['sha256']))
    return score(records)
labels=['background','cough','glass_break','baby_cry','dog_bark'];esc=read(I/'esc-manifest.json')['inputs'];ext=read(I/'manifest.json')['inputs']
environment=dict(uname=platform.uname()._asdict(),gcc=run(['gcc','--version']),python=platform.python_version(),ldd=run(['ldd','--version']),cpu=run(['lscpu']),memory=run(['free','-b']))
save(O/'environment.json',environment)
run(['sh',R/'scripts/verify_submission.sh'],O/'integrity.log',cwd=R)
run(['sh',R/'scripts/build_audio_sentinel_app.sh'],O/'default-build.log',cwd=R,env={**os.environ,'OUT_DIR':str(O/'default')})
default=O/'default/audiodetect'
run(['python3',R/'scripts/verify_primary_app.py','--inputs',I,'--audit',a.reference,'--output',O/'app','--compiler','gcc'],O/'app-check.log')
app=read(O/'app/verification.json');assert app['external_correct']['mlp']==57
# Fresh Linux compilation of all five formerly INNER-selected folds and final.
selected=read(a.reference/'results.json')['models']['mlp']['models'];formal=[];outer=[];appdir=R/'app/audiodetect'
for item in selected:
    name=item['model'];src=a.exports/('mlp-'+name);d=O/'selected'/name;d.mkdir(parents=True)
    for f in ('weights.h','harness.c'):shutil.copyfile(src/f,d/f)
    binary=d/'infer';build(['gcc','-std=c11','-O2','-Wall','-Wextra','-I',d,'-I',appdir,d/'harness.c',*[appdir/n for n in ('feature_extractor.c','wav_reader.c','keyword_detect.c')],'-lm','-o',binary],d/'build.log')
    result=dict(model=name,binary_sha256=sha(binary))
    if name.startswith('fold_'):
        result['official']=tsv(binary,[r for r in esc if r['fold']==int(name[-1])],d/'official.tsv');outer+=result['official']['records']
    result['external']=tsv(binary,ext,d/'external.tsv');formal.append(result)
formal_result=dict(official=score(outer),external_final=formal[-1]['external'],models=formal)
save(O/'formal-selected.json',formal_result)
print('LINUX formal C',formal_result['official']['correct'],formal_result['external_final']['correct'],flush=True)
# Standard-library feature dump provides the real Linux frontend for host-side
# NumPy/Torch neural parity checks; host Python is not mislabeled as Linux.
dump=O/'feature_dump';build(['gcc','-std=c11','-O2','-I',appdir,R/'scripts/legacy_mlp_features.c',*[appdir/n for n in ('feature_extractor.c','wav_reader.c','keyword_detect.c')],'-lm','-o',dump],O/'feature-build.log')
run([dump,*[r['path'] for r in esc+ext]],O/'features.log',cwd=I)
# All controlled ablations are independently compiled and evaluated in Linux C.
study=read(a.study/'frozen.json');study_results=[]
for variant in dict.fromkeys(v['variant'] for v in study['models']):
    models=[];outer=[]
    for item in [v for v in study['models'] if v['variant']==variant]:
        name=item['model'];src=a.study/variant/name;d=O/'ablations'/variant/name;d.mkdir(parents=True)
        for f in ('weights.h','harness.c'):shutil.copyfile(src/f,d/f)
        assert sha(d/'weights.h')==item['header_sha256'];binary=d/'infer'
        build(['gcc','-std=c11','-O2','-Wall','-Wextra','-I',d,'-I',appdir,d/'harness.c',*[appdir/n for n in ('feature_extractor.c','wav_reader.c','keyword_detect.c')],'-lm','-o',binary],d/'build.log')
        result=dict(model=name,parameters=item['parameters'],feature_dim=item['feature_dim'],model_sha256=item['model_sha256'],binary_sha256=sha(binary))
        if name.startswith('fold_'):
            result['official']=tsv(binary,[r for r in esc if r['fold']==int(name[-1])],d/'official.tsv');outer+=result['official']['records']
        result['external']=tsv(binary,ext,d/'external.tsv');models.append(result)
    value=dict(variant=variant,official=score(outer),external_final=models[-1]['external'],external_fold_mean=statistics.mean(v['external']['accuracy'] for v in models[:5]),models=models)
    study_results.append(value);save(O/'ablation-progress.json',study_results);print('LINUX ablation',variant,value['official']['correct'],value['external_final']['correct'],flush=True)
save(O/'ablations.json',study_results)
# Fresh controlled noise robustness (held-out fixed-full fold models only).
noise=[]
for level,rows in read(I/'noise-manifest.json').items():
    pooled=[]
    for fold in range(1,6):pooled+=tsv(O/f'ablations/full/fold_{fold}/infer',[r for r in rows if r['fold']==fold],O/f'noise-{level}-fold{fold}.tsv')['records']
    noise.append(dict(condition=level,**score(pooled)))
save(O/'noise.json',noise)
# Sanitizers exercise all65 variable-length files, digital silence and malformed WAVs.
san=O/'audiodetect-sanitized';source=['audiodetect_main.c','audio_classifier.c','wav_reader.c','keyword_detect.c','alarm_report.c','feature_extractor.c','mlp_model_infer.c']
build(['gcc','-std=c11','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer','-fno-pie','-no-pie',*[appdir/n for n in source],'-lm','-o',san],O/'sanitizer-build.log')
sanenv={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'}
for s in ext:run([san,I/s['path'],'--debug'],O/('san-'+s['id']+'.log'),env=sanenv,timeout=30)
negative=[]
for name,content in [('invalid',b'not a WAV'),('truncated',b'RIFF\xff\xff\xff\xffWAVEfmt ')]:
    f=O/(name+'.wav');f.write_bytes(content);res=subprocess.run([str(san),str(f)],capture_output=True,env=sanenv);assert res.returncode!=0 and b'AddressSanitizer' not in res.stderr and b'runtime error:' not in res.stderr
    negative.append(dict(case=name,exit_code=res.returncode,stderr=res.stderr.decode()))
# Chunk sizes test different WAV reads; no claim of continuous microphone streaming.
chunks=[];reference={v['id']:v for v in app['records']['mlp']}
for chunk in (1,997,4096):
    d=O/f'chunk-{chunk}';d.mkdir();code=(appdir/'audio_classifier.c').read_text();code=code.replace('uint32_t bytes;int16_t buffer[160];',f'uint32_t bytes;int16_t buffer[{chunk}];').replace('if(n>160)n=160;',f'if(n>{chunk})n={chunk};');(d/'audio_classifier.c').write_text(code)
    binary=d/'audiodetect';build(['gcc','-std=c11','-O2','-I',appdir,*[appdir/n for n in source if n!='audio_classifier.c'],d/'audio_classifier.c','-lm','-o',binary],d/'build.log')
    delta=0.
    for s in ext:
        text=run([binary,I/s['path'],'--debug']);pred=re.search(r'^Result: (\w+)$',text,re.M)[1];prob=[float(re.search(r'^  '+l+r': ([0-9.eE+-]+)$',text,re.M)[1]) for l in labels]
        v=reference[s['id']];assert labels[v['prediction']]==pred;delta=max(delta,max(abs(x-y) for x,y in zip(prob,v['probabilities'])))
    assert delta<1e-7;chunks.append(dict(samples_per_read=chunk,recordings=65,max_probability_delta=delta))
# Fixed3s CLI process time plus separately instrumented in-process WAV classification.
with wave.open(str(I/esc[0]['path']),'rb') as f:pcm=f.readframes(48000)
fixed=O/'fixed3s.wav'
with wave.open(str(fixed),'wb') as f:f.setparams((1,2,16000,0,'NONE','not compressed'));f.writeframes(pcm)
bench=O/'bench.c';bench.write_text('''#define _POSIX_C_SOURCE 200809L
#include "audio_classifier.h"
#include <time.h>
#include <stdio.h>
int main(int argc,char **argv){aed_result_t r;for(int i=0;i<33;i++){struct timespec a,b;clock_gettime(CLOCK_MONOTONIC,&a);if(aed_classify_wav(argv[1],&r))return 1;clock_gettime(CLOCK_MONOTONIC,&b);if(i>=3)printf("%.9f\\n",(b.tv_sec-a.tv_sec)*1000.+(b.tv_nsec-a.tv_nsec)/1000000.);}return 0;}
''')
build(['gcc','-std=c11','-O2','-I',appdir,bench,*[appdir/n for n in ('audio_classifier.c','wav_reader.c','keyword_detect.c','feature_extractor.c','mlp_model_infer.c')],'-lm','-o',O/'bench'],O/'bench-build.log')
core=[float(x) for x in run([O/'bench',fixed]).split()]
wall=[]
for _ in range(3):run([default,fixed,'--debug'])
for _ in range(30):
    start=time.perf_counter();run([default,fixed,'--debug']);wall.append((time.perf_counter()-start)*1000)
variable=[]
for s in ext:
    start=time.perf_counter();run([default,I/s['path'],'--debug']);variable.append(dict(id=s['id'],ms=(time.perf_counter()-start)*1000))
def distribution(values):
    v=sorted(values);i=(len(v)-1)*.95;lo=int(i);p95=v[lo]+(v[min(lo+1,len(v)-1)]-v[lo])*(i-lo)
    return dict(n=len(v),mean_ms=statistics.mean(v),median_ms=statistics.median(v),p95_ms=p95,min_ms=v[0],max_ms=v[-1],raw_ms=values)
run(['/usr/bin/time','-f','%M','-o',O/'rss-kib.txt',default,fixed,'--debug'],O/'time-stdout.log')
performance=dict(fixed3s_core=distribution(core),fixed3s_cli_process=distribution(wall),variable65_cli_process=distribution([v['ms'] for v in variable]),variable_raw=variable,
    core_scope='File reading+DSP+asinh+MLP+aggregation inside one persistent process,3warmup30measured; excludes stdout/process spawn/acquisition.',
    cli_scope='Wall time of freshCLI process including startup/file/DSP/model/stdout capture;3warmup30measured. Not sound-to-alarm latency.',
    fixed_input_sha256=sha(fixed),peak_cli_rss_kib=int((O/'rss-kib.txt').read_text()),binary_bytes=default.stat().st_size,size=run(['size','-A',default]),
    power_measured=False,microphone_tested=False,full_openvela_tested=False,board_tested=False,co_load='Host also running CPU-only training/baseline jobs; virtualized wall timings subject to scheduling')
save(O/'performance.json',performance)
save(O/'results.json',dict(status='PASS',runtime='Actual Ubuntu22.04 x86_64 native C, not fullOpenVela or board',formal=dict(official_correct=formal_result['official']['correct'],external_correct=formal_result['external_final']['correct']),ablations=len(study_results),sanitized_external_recordings=65,negative_wav_cases=negative,chunk_checks=chunks,performance=performance,production_model_unchanged=True))
save(O/'commands.json',commands);print('LINUX SUITE PASS',flush=True)
