"""Evaluate the six frozen scratch fits without tuning or replacing deployment."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,hashlib,json,pathlib,shutil,subprocess,sys,tarfile
import numpy as np
import torch
from cnn_v2 import LABELS,all_features,load,predict,read_wav_mono,metrics,REPO
from audio_utils import write_wav_mono

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--training',type=pathlib.Path,required=True)
p.add_argument('--review',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
assert not (out/'python-results.json').exists(),'Preserve completed evaluation'
read=lambda f:json.loads(f.read_text(encoding='utf-8'))
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def save(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
frozen=read(a.training/'training-frozen.json');protocol=read(a.training/'protocol.json')
assert frozen['status']=='SIX_RANDOM_INITIALIZATION_FITS_COMPLETE'
assert frozen['protocol_sha256']==sha(a.training/'protocol.json')
assert protocol['external_used_for_selection_in_this_run'] is False
assert read(a.review/'review-acceptance-20261019.json')['confirmed_recordings']==65
records=read(a.review/'manifest.json')['records'];assert len(records)==65
features=[];inputs=[];(out/'inputs').mkdir(exist_ok=True)
torch.set_num_threads(2);torch.set_num_interop_threads(1)
for row in records:
    src=a.review/row['audio_path'];assert sha(src)==row['sha256']
    pcm=read_wav_mono(src,16000);features.append(all_features(pcm)[:26])
    dest=out/'inputs'/(row['id']+'.wav');write_wav_mono(dest,pcm,16000)
    inputs.append(dict(id=row['id'],reference_label=row['reference_label'],path=dest.relative_to(out).as_posix(),sha256=sha(dest),original_sha256=row['sha256'],samples=len(pcm)))
truth=np.asarray([LABELS.index(r['reference_label']) for r in records]);results=[]
for item in frozen['models']:
    name=item['model'];path=a.training/'models'/(name+'.npz');assert sha(path)==item['model_sha256']
    modelout=out/'models'/name;modelout.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(REPO/'training/cnn_v2/export_c.py'),str(path),str(modelout)],stdout=subprocess.DEVNULL,check=True)
    for filename in ['cnn_stream_v2.c','cnn_stream_v2.h']:shutil.copyfile(REPO/'app/audiodetect'/filename,modelout/filename)
    result=dict(model=name,model_sha256=item['model_sha256'])
    for mode in ('float','int8'):
        network,mean,std=load(path,quantized=(mode=='int8'));prob=predict(network,features,mean,std);pred=prob.argmax(1)
        result[mode]=metrics(truth,pred)
        result[mode]['predictions']=[dict(id=row['id'],reference_label=row['reference_label'],prediction=LABELS[int(pr)],correct=int(pr)==int(t),confidence=float(v[pr]),probabilities=v.tolist()) for row,t,pr,v in zip(records,truth,pred,prob)]
        if mode=='int8':np.savez_compressed(modelout/'python-reference.npz',PROBABILITIES=prob,IDS=np.asarray([r['id'] for r in records]))
    results.append(result)
    print('FROZEN EXTERNAL',name,'float',result['float']['correct'],'int8',result['int8']['correct'],'baby',result['int8']['per_class_recall']['baby_cry'],flush=True)
manifest=dict(labels=LABELS,inputs=inputs,models=[{k:r[k] for k in ('model','model_sha256')} for r in results],user_confirmed_recordings=65,external_training=False,external_used_for_model_selection=False,external_role='Previously reviewed development cohort; NOT a new blind test. No candidate/epoch/seed selected on these scores in this run.',training_frozen_sha256=sha(a.training/'training-frozen.json'),dataset_manifest_sha256=sha(a.review/'manifest.json'),probability_parity_tolerance=.0001,final_aggregation='Mean of ALL causal frame logits, then Softmax')
save('manifest.json',manifest);save('python-results.json',dict(models=results,manifest=manifest,mean_fold_accuracy={mode:float(np.mean([r[mode]['accuracy'] for r in results[:5]])) for mode in ('float','int8')},fold_external_note='Five ESC-trained models evaluated on the same external65; not external five-fold CV'))
shutil.copyfile(REPO/'simulator/cnn_v2/cnn_cli.c',out/'cnn_cli.c')
app=out/'app-source';app.mkdir(exist_ok=True)
for filename in ['audiodetect_main.c','audio_classifier.c','audio_classifier.h','cnn_stream_v2.c','cnn_stream_v2.h','wav_reader.c','wav_reader.h','keyword_detect.c','keyword_detect.h','keyword_templates.h','alarm_report.c','alarm_report.h']:
    shutil.copyfile(REPO/'app/audiodetect'/filename,app/filename)
final=out/'models/final_all_esc50'
for filename in ('cnn_weights_v2.h','cnn_model_config_v2.h'):shutil.copyfile(final/filename,app/filename)
with tarfile.open(out.with_suffix('.tar.gz'),'w:gz') as tar:
    for file in out.rglob('*'):
        if file.is_file():tar.add(file,arcname=file.relative_to(out).as_posix())
print('FROZEN EVALUATION COMPLETE',flush=True)
