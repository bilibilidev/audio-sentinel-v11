"""Frozen paired MLP/CNN evaluation; no training or candidate selection here."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,csv,datetime,hashlib,io,json,pathlib,platform,shutil,subprocess,sys,time
import numpy as np
import torch
repo=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo/'training'));sys.path.insert(0,str(repo/'training/cnn_v2'))
from export_to_c import float_header,int8_header
from extract_features_mfcc import extract_features_mfcc_from_pcm
from audio_utils import read_wav_mono
from train_edge_mfcc import quantize_dequantize
from sweep_edge_models import forward
from cnn_v2 import all_features,load,predict,LABELS
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-root',type=pathlib.Path,required=True)
p.add_argument('--inputs',type=pathlib.Path,required=True)
p.add_argument('--compiler',required=True)
p.add_argument('--cflag',action='append',default=[])
p.add_argument('--output-name',default='evaluation')
a=p.parse_args();base=a.run_root.resolve();assert pathlib.Path(a.output_name).name==a.output_name
out=base/a.output_name;out.mkdir(exist_ok=True)
assert not (out/'results.json').exists(),'Keep completed paired results immutable'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def json_default(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
def save(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=json_default)+'\n',encoding='utf-8',newline='\n')
frozen={kind:read(base/(kind+'-training')/'training-frozen.json') for kind in ('mlp','cnn')}
for kind in frozen:
    for item in frozen[kind]['models']:
        assert sha(base/(kind+'-training/models')/(item['model']+'.npz'))==item['model_sha256']
assert not (out/'joint-freeze.json').exists()
save('joint-freeze.json',dict(created=datetime.datetime.now().astimezone().isoformat(),external_opened=False,
     training_frozen_sha256={kind:sha(base/(kind+'-training/training-frozen.json')) for kind in frozen},
     external_role='Previously reviewed development65; no tuning on this run; not new blind data',
     decision='Compare all recorded results; do not select a seed, epoch, threshold or aggregation on external scores'))
manifest=read(a.inputs/'manifest.json');esc=read(a.inputs/'esc-manifest.json')['inputs'];external=manifest['inputs']
assert len(external)==65 and len(esc)==390
for row in [*external,*esc]:assert sha(a.inputs/row['path'])==row['sha256']
torch.set_num_threads(2);torch.set_num_interop_threads(1)
features=[];window_features=[];external_counts=[]
for row in external:
    pcm=read_wav_mono(a.inputs/row['path']);features.append(all_features(pcm)[:26]);parts=[]
    for offset in range(0,len(pcm),16000):
        chunk=np.zeros(16000,dtype=np.int16);actual=pcm[offset:offset+16000];chunk[:len(actual)]=actual
        parts.append(extract_features_mfcc_from_pcm(chunk,include_delta=True))
    window_features.extend(parts);external_counts.append(len(parts))
window_features=np.stack(window_features).astype(np.float32)
with np.load(base/'mlp-training/clean-features.npz',allow_pickle=False) as z:
    clean=z['X'].copy();fold_ids=z['FOLDS'].copy();sources=z['SOURCES'].copy();clean_y=z['Y'].copy()
commands=[];all_results={'mlp':[],'cnn':[]};official={kind:{m:[] for m in (('legacy','full','mean') if kind=='mlp' else ('mean_logits',))} for kind in all_results}
suffix='.exe' if platform.system()=='Windows' else ''
def scores(rows):
    matrix=[[0]*5 for _ in range(5)]
    for row in rows:matrix[row['truth']][row['prediction']]+=1
    correct=sum(matrix[i][i] for i in range(5));recall={label:matrix[i][i]/sum(matrix[i]) for i,label in enumerate(LABELS)}
    return dict(correct=correct,total=len(rows),accuracy=correct/len(rows),macro_recall=sum(recall.values())/5,per_class_recall=recall,confusion_matrix=matrix)
def build(cmd,log):
    commands.append(cmd)
    with log.open('w') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
def execute(binary,mode,rows,log):
    cmd=[str(binary),*([mode] if mode else []),*[str((a.inputs/r['path']).resolve()) for r in rows]];commands.append(cmd)
    text=subprocess.check_output(cmd,text=True,timeout=120);log.write_text(text,encoding='utf-8',newline='\n')
    parsed=list(csv.DictReader(io.StringIO(text),delimiter='\t'));assert len(parsed)==len(rows)
    result=[]
    for s,r in zip(rows,parsed):
        assert pathlib.Path(r['file'])==(a.inputs/s['path']).resolve()
        truth=s.get('truth',LABELS.index(s['reference_label']) if 'reference_label' in s else None)
        result.append(dict(id=s.get('id',s.get('filename')),filename=s.get('filename'),truth=truth,prediction=LABELS.index(r['prediction']),
              probabilities=[float(r['p'+str(i)]) for i in range(5)],confidence=float(r['confidence'])))
    return result
def average_windows(probs,counts):
    result=[];offset=0
    for count in counts:result.append(probs[offset:offset+count].mean(0));offset+=count
    assert offset==len(probs)
    return np.stack(result)
app=repo/'app/audiodetect';feature_binary=out/('feature_dump'+suffix)
build([a.compiler,*a.cflag,'-std=c11','-O2','-Wall','-Wextra','-I',str(app),str(repo/'scripts/legacy_mlp_features.c'),*[str(app/n) for n in ('feature_extractor.c','keyword_detect.c','wav_reader.c')],'-lm','-o',str(feature_binary)],out/'feature-build.log')
cmd=[str(feature_binary),*[r['path'] for r in [*external,*esc]]];commands.append(cmd)
text=subprocess.check_output(cmd,cwd=a.inputs.resolve(),text=True,timeout=120);(out/'native-features.txt').write_text(text,encoding='utf-8')
native={i:[] for i in range(455)}
for row in text.splitlines():
    values=row.split();i=int(values[0]);assert int(values[1])==len(native[i]);native[i].append([float(x) for x in values[2:]])
assert all(len(native[i])==external_counts[i] for i in range(65)) and all(len(native[i])==5 for i in range(65,455))
native_external=np.asarray([x for i in range(65) for x in native[i]],dtype=np.float32)
native_esc={row['filename']:np.asarray(native[65+i],dtype=np.float32) for i,row in enumerate(esc)}
for kind in ('mlp','cnn'):
    for item in frozen[kind]['models']:
        name=item['model'];d=out/(kind+'-'+name);d.mkdir();model=base/(kind+'-training/models')/(name+'.npz')
        binary=d/('inference'+suffix)
        if kind=='mlp':
            app=d/'app';sim=d/'simulator';app.mkdir();(sim/'generated').mkdir(parents=True)
            names=['audio_classifier.h','audio_classifier_mlp.c','model_infer.h','feature_extractor.c','feature_extractor.h','wav_reader.c','wav_reader.h','keyword_detect.c','keyword_detect.h','keyword_templates.h']
            for n in names:shutil.copyfile(repo/'app/audiodetect'/n,app/n)
            for n in ('model_infer_int8.c','model_infer_int8.h'):shutil.copyfile(repo/'simulator'/n,sim/n)
            shutil.copyfile(repo/'scripts/legacy_mlp_evaluation.c',d/'harness.c')
            with np.load(model,allow_pickle=False) as z:
                mean=z['FEATURE_MEAN'].copy();std=z['FEATURE_STD'].copy();weights=[dict(W=z['W1'].copy(),B=z['B1'].copy()),dict(W=z['W2'].copy(),B=z['B2'].copy())]
                (app/'model_weights.h').write_text(float_header(z,LABELS),encoding='utf-8',newline='\n')
                (sim/'generated/model_weights_int8.h').write_text(int8_header(z,LABELS,'SIMULATOR_MODEL_WEIGHTS_INT8_H',False),encoding='utf-8',newline='\n')
            cmd=[a.compiler,*a.cflag,'-std=c11','-O2','-Wall','-Wextra','-I',str(sim),'-I',str(app),str(d/'harness.c'),*[str(app/n) for n in ['audio_classifier_mlp.c','wav_reader.c','feature_extractor.c','keyword_detect.c']],str(sim/'model_infer_int8.c'),'-lm','-o',str(binary)]
            build(cmd,d/'build.log');modes=('legacy','full','mean');references={}
            for mode,w in [('float',weights),('int8',quantize_dequantize(weights))]:
                references[mode]=average_windows(forward((window_features-mean)/std,w)[0],external_counts)
            references['int8_native_frontend']=average_windows(forward((native_external-mean)/std,quantize_dequantize(weights))[0],external_counts)
        else:
            cmd=[sys.executable,str(repo/'training/cnn_v2/export_c.py'),str(model),str(d)];commands.append(cmd)
            with (d/'export.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log)
            for n in ('cnn_stream_v2.c','cnn_stream_v2.h'):shutil.copyfile(repo/'app/audiodetect'/n,d/n)
            shutil.copyfile(repo/'simulator/cnn_v2/cnn_cli.c',d/'harness.c')
            cmd=[a.compiler,*a.cflag,'-D__USE_MINGW_ANSI_STDIO=1','-std=c11','-O2','-Wall','-Wextra','-I',str(d),str(d/'cnn_stream_v2.c'),str(d/'harness.c'),'-lm','-o',str(binary)]
            build(cmd,d/'build.log');modes=('mean_logits',);references={}
            for mode in ('float','int8'):
                network,mean,std=load(model,quantized=mode=='int8');references[mode]=predict(network,features,mean,std)
        result=dict(model=name,model_sha256=item['model_sha256'],binary_sha256=sha(binary),external={},official={})
        for mode in modes:
            rows=execute(binary,mode if kind=='mlp' else None,external,d/('external-'+mode+'.tsv'))
            result['external'][mode]={**scores(rows),'records':rows}
            if mode==modes[-1]:
                delta=float(np.max(np.abs(np.asarray([r['probabilities'] for r in rows])-references['int8'])))
                result['python_c_max_probability_delta']=delta
                if kind=='mlp':
                    ndelta=float(np.max(np.abs(np.asarray([r['probabilities'] for r in rows])-references['int8_native_frontend'])))
                    assert ndelta<.0001,(kind,name,'same C frontend',ndelta)
                    result['same_c_frontend_network_probability_delta']=ndelta
                    result['c_vs_python_frontend_prediction_mismatches']=int(np.sum(np.asarray([r['prediction'] for r in rows])!=references['int8'].argmax(1)))
                else:assert delta<.0001,(kind,name,delta)
            if name.startswith('fold_'):
                fold=int(name[-1]);holdout=[r for r in esc if r['fold']==fold]
                test=execute(binary,mode if kind=='mlp' else None,holdout,d/('official-'+mode+'.tsv'))
                result['official'][mode]={**scores(test),'records':test};official[kind][mode]+=test
                if mode==modes[-1]:
                    if kind=='mlp':
                        expected={source['filename']:forward((native_esc[source['filename']]-mean)/std,quantize_dequantize(weights))[0].mean(0) for source in holdout}
                    else:
                        receipt=read(base/'cnn-training/receipts'/(name+'.json'));expected={r['filename']:r['probabilities'] for r in receipt['int8']['predictions']}
                    delta=max(max(abs(p-q) for p,q in zip(r['probabilities'],expected[r['filename']])) for r in test)
                    assert delta<.0001,(kind,name,'official',delta)
                    result['official_c_python_max_probability_delta']=delta
        result['external_float_reference']=dict(**scores([dict(truth=LABELS.index(s['reference_label']),prediction=int(prob.argmax())) for s,prob in zip(external,references['float'])]),
             probabilities=references['float'].tolist())
        all_results[kind].append(result);save('progress.json',dict(kind=kind,completed_model=name))
        print(kind,name,{mode:(result['external'][mode]['correct'],result['external'][mode]['per_class_recall']['baby_cry']) for mode in modes},flush=True)
save('results.json',dict(status='PASS',runtime=platform.platform(),compiler=subprocess.check_output([a.compiler,'--version'],text=True),
     official_cv={kind:{mode:scores(rows) for mode,rows in modes.items()} for kind,modes in official.items()},models=all_results,
     input_hashes_verified=455,external_role=manifest['external_role'],joint_freeze_sha256=sha(out/'joint-freeze.json'),
     note='Actual native C for all saved fold models and independent all390 fits. Original models and board unchanged.'))
save('commands.json',commands)
print('PAIRED EVALUATION COMPLETE',flush=True)
