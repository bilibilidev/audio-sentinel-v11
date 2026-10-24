"""Re-evaluate saved held-out tiny MLP models, never retraining deployment weights."""
import argparse,csv,hashlib,io,json,os,pathlib,platform,shutil,subprocess,sys,time
import numpy as np
repo=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo/'training'))
from export_to_c import float_header,int8_header
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--folds',type=pathlib.Path,required=True)
p.add_argument('--inputs',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--compiler',default='cc')
p.add_argument('--cflag',action='append',default=[])
a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
assert not (out/'results.json').exists(),'Keep completed results immutable; use a new output directory'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
labels=['background','cough','glass_break','baby_cry','dog_bark']
def score(rows):
    m=[[0]*5 for _ in range(5)]
    for r in rows:m[r['truth']][r['prediction']]+=1
    correct=sum(m[i][i] for i in range(5))
    return dict(correct=correct,total=len(rows),accuracy=correct/len(rows),confusion_matrix=m,
                per_class={label:dict(correct=m[i][i],total=sum(m[i]),recall=m[i][i]/sum(m[i])) for i,label in enumerate(labels)})
historical=read(a.folds/'summary.json');esc=read(a.inputs/'esc-manifest.json')['inputs']
assert len(esc)==390 and len({r['filename'] for r in esc})==390
protected={name:sha(repo/name) for name in ['model/tiny_mlp_mfcc92_robust_simulator.npz','app/audiodetect/model_weights.h','app/audiodetect/model_weights_int8.h','simulator/generated/model_weights_int8.h','app/audiodetect/cnn_weights_v2.h']}
for row in esc:assert sha(a.inputs/row['path'])==row['sha256']
commands=[];models=[];fold_results=[];pooled={m:[] for m in ('legacy','full','mean')};mismatches=[];start=time.perf_counter()
appnames=['audio_classifier.h','audio_classifier_mlp.c','model_infer.h','feature_extractor.c','feature_extractor.h','wav_reader.c','wav_reader.h','keyword_detect.c','keyword_detect.h','keyword_templates.h']
sources={f'app/audiodetect/{n}':sha(repo/'app/audiodetect'/n) for n in appnames}
sources.update({n:sha(repo/n) for n in ['simulator/model_infer_int8.c','simulator/model_infer_int8.h','scripts/legacy_mlp_evaluation.c','scripts/evaluate_legacy_mlp_cv.py']})
for fold in range(1,6):
    prior=next(r for r in historical['folds'] if r['fold']==fold)
    model=a.folds/'models'/f'fold_{fold}.npz';assert sha(model)==prior['model_sha256'] and prior['source_overlap']==0
    d=out/f'fold_{fold}';app=d/'app';sim=d/'simulator';app.mkdir(parents=True);(sim/'generated').mkdir(parents=True)
    for name in appnames:shutil.copyfile(repo/'app/audiodetect'/name,app/name)
    for name in ['model_infer_int8.c','model_infer_int8.h']:shutil.copyfile(repo/'simulator'/name,sim/name)
    shutil.copyfile(repo/'scripts/legacy_mlp_evaluation.c',d/'harness.c')
    with np.load(model,allow_pickle=False) as z:
        assert int(z['TEST_FOLD'])==fold and z['LABEL_NAMES'].tolist()==labels
        (app/'model_weights.h').write_text(float_header(z,labels),encoding='utf-8',newline='\n')
        (sim/'generated/model_weights_int8.h').write_text(int8_header(z,labels,'SIMULATOR_MODEL_WEIGHTS_INT8_H',False),encoding='utf-8',newline='\n')
    binary=d/('mlp_cv.exe' if platform.system()=='Windows' else 'mlp_cv')
    cmd=[a.compiler,*a.cflag,'-std=c11','-O2','-Wall','-Wextra','-I',str(sim),'-I',str(app),str(d/'harness.c'),*[str(app/n) for n in ['audio_classifier_mlp.c','wav_reader.c','feature_extractor.c','keyword_detect.c']],str(sim/'model_infer_int8.c'),'-lm','-o',str(binary)]
    commands.append(cmd)
    with (d/'build.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    selected=[r for r in esc if r['fold']==fold];references={pathlib.PurePosixPath(r['source']).name:r for r in prior['predictions']}
    assert len(selected)==len(references)==prior['test_sources']
    results={}
    for mode in pooled:
        cmd=[str(binary),mode,*[str((a.inputs/r['path']).resolve()) for r in selected]];commands.append(cmd)
        text=subprocess.check_output(cmd,text=True,timeout=120);(d/(mode+'.tsv')).write_text(text,encoding='utf-8',newline='\n')
        rows=list(csv.DictReader(io.StringIO(text),delimiter='\t'));assert len(rows)==len(selected)
        records=[]
        for source,row in zip(selected,rows):
            pred=labels.index(row['prediction']);ref=references[source['filename']];assert ref['truth']==source['truth']
            record=dict(fold=fold,filename=source['filename'],truth=source['truth'],prediction=pred,confidence=float(row['confidence']),
                 probabilities=[float(row['p'+str(i)]) for i in range(5)],windows=int(row['windows']),quiet_windows=int(row['quiet_windows']),input_sha256=source['sha256'])
            records.append(record)
            if mode=='mean' and pred!=ref['int8']:mismatches.append(dict(filename=source['filename'],c=pred,python=ref['int8']))
        assert all(r['windows']==(3 if mode=='legacy' else 5) for r in records)
        pooled[mode]+=records;results[mode]={**score(records),'records':records}
    models.append(dict(fold=fold,model_sha256=sha(model),binary_sha256=sha(binary),source_overlap=0,training_recordings=prior['training_sources'],test_recordings=len(selected)))
    fold_results.append(dict(fold=fold,modes=results))
    print('FOLD',fold,{mode:results[mode]['correct'] for mode in results},'/',len(selected),flush=True)
assert protected=={name:sha(repo/name) for name in protected}
summary=dict(status='PASS',runtime=platform.platform(),models=models,folds=fold_results,labels=labels,
     official_cv={mode:score(rows) for mode,rows in pooled.items()},historical_python=historical['recording_level'],
     mean_c_python_prediction_mismatches=mismatches,compiler=subprocess.check_output([a.compiler,'--version'],text=True),
     modes=dict(legacy='Original App, first3seconds, one-second windows, quiet gate and event-max/mean',full='All5seconds, same App gate and event-max/mean in evaluation-only adapter',mean='All5seconds, mean of5 one-second probability vectors, no quiet gate or event-max; original offline CV aggregation'),
     source_hashes=sources,protected_files=protected,protected_unchanged=True,retrained=False,input_recordings=390,
     input_manifest_sha256=sha(a.inputs/'esc-manifest.json'),saved_fold_summary_sha256=sha(a.folds/'summary.json'),elapsed_seconds=time.perf_counter()-start)
save('results.json',summary);save('commands.json',commands)
print('CV COMPLETE',json.dumps(summary['official_cv']),'; mean parity mismatches',len(mismatches),flush=True)
