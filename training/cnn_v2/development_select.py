"""Explicit development-set candidate selection. Not a blind external test."""
import os

import datetime,hashlib,json,pathlib,shutil,time
import numpy as np
from finetune_v1 import CNNV2,configs,views,y,folds,rows,adapt,export,load,predict,metrics,REPO,load_model
REPO=pathlib.Path(__file__).resolve().parents[2]
ROOT=pathlib.Path(os.environ.get('CNN_V2_RUN_DIR',str(REPO/'out/cnn-v2-training'))).resolve()
ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/'artifacts/development-round4';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (OUT/'summary.json').exists(),'Preserve completed selection'
protocol={'registered':datetime.datetime.now().astimezone().isoformat(),'candidates':configs,'training':'Same 390 ESC-50 only; v1 final warm start; same seed5242 and fixed20 epochs for every candidate',
          'external_used_for_gradient_training':False,'external_used_for_model_selection':True,
          'selection':'INT8 feasibility: overall >= frozen v1 52/65 and baby > frozen v1 12/20. Among feasible choose highest macro recall, then accuracy, then baby precision. If none feasible do not replace v1.',
          'external_role':'Repeatedly seen user-confirmed DEVELOPMENT set. This is explicitly optimistic development evidence, not a held-out blind-test claim.',
          'cv':'After selecting recipe on external development set, fit the same fixed recipe to each corresponding v1 four-fold initializer and score untouched ESC-50 fold; report as fixed-recipe CV, not nested selection CV.'}
if not (OUT/'protocol.json').exists():save(OUT/'protocol.json',protocol)
else:assert json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))['candidates']==configs
# Train all candidates first; no candidate parameters or seed changed from scores.
parent=REPO/'model/cnn-v1/final_all_esc50.npz';old,mean,std=load_model(parent)
for config in configs:
    path=OUT/'candidates'/config['name']/'final_all_esc50.npz'
    if path.exists():continue
    if config['name']=='phone_no_anchor':
        path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'artifacts/warmstart-round3/selected/final_all_esc50.npz',path)
    else:
        initial=CNNV2(config);initial.load_state_dict(old.state_dict());model=adapt(initial.eval(),mean,std,np.arange(390),config,5242)
        export(path,model,mean,std,TRAINING_RECORDINGS=np.asarray(390),BASE_MODEL_SHA256=np.asarray(sha(parent)),DEVELOPMENT_SELECTION=np.asarray(True))
    print('FROZEN CANDIDATE',config['name'],sha(path),flush=True)
from cnn_v2 import all_features,select_features,read_wav_mono,LABELS
review=pathlib.Path(os.environ['EXTERNAL_AUDIO_DIR'])
manifest=json.loads((review/'manifest.json').read_text(encoding='utf-8'));records=manifest['records']
accept=json.loads((review/'review-acceptance-20261019.json').read_text(encoding='utf-8'));assert accept['confirmed_recordings']==65
features=[]
for r in records:
    path=review/r['audio_path'];assert sha(path)==r['sha256'];features.append(all_features(read_wav_mono(path,16000))[:26])
truth=np.asarray([LABELS.index(r['reference_label']) for r in records]);evaluated=[]
for config in configs:
    path=OUT/'candidates'/config['name']/'final_all_esc50.npz';item={'config':config,'model_sha256':sha(path)}
    for mode in ['float','int8']:
        model,mean,std=load(path,mode=='int8');probs=predict(model,features,mean,std);pred=probs.argmax(1);item[mode]=metrics(truth,pred)
        item[mode]['predictions']=[{'id':r['id'],'truth':r['reference_label'],'prediction':LABELS[int(p)],'probabilities':v.tolist()} for r,p,v in zip(records,pred,probs)]
    matrix=np.asarray(item['int8']['confusion_matrix']);item['baby_precision']=float(matrix[3,3]/max(matrix[:,3].sum(),1));item['feasible']=bool(item['int8']['correct']>=52 and matrix[3,3]>12)
    evaluated.append(item);print('DEVELOPMENT',config['name'],item['int8']['correct'],'/65 baby',matrix[3,3],'/20 precision',item['baby_precision'],'feasible',item['feasible'],flush=True)
save(OUT/'candidate-results.json',evaluated)
feasible=[e for e in evaluated if e['feasible']]
if not feasible:
    save(OUT/'not-accepted.json',{'reason':'No candidate meets preregistered v1 overall+baby constraints','candidates':evaluated});raise SystemExit('NO ACCEPTED CANDIDATE; retain v1')
chosen=max(feasible,key=lambda e:(e['int8']['balanced_accuracy_present_classes'],e['int8']['accuracy'],e['baby_precision']));config=chosen['config']
selection={'final_config':config,'final_epochs':20,'selected_without_external_audio':False,'external_used_for_model_selection':True,'external_training':False,
           'config_selection_basis':'Explicit repeated development-set selection; final blind evaluation still needed','selected_development_metrics':chosen['int8'],'per_outer_fold':[]}
save(OUT/'selection.json',selection)
models=OUT/'selected';models.mkdir(exist_ok=True);shutil.copyfile(OUT/'candidates'/config['name']/'final_all_esc50.npz',models/'final_all_esc50.npz')
pooled={m:[] for m in ['float','int8']};truth_cv=[];results=[]
for fold in range(1,6):
    path=models/f'fold_{fold}.npz';receipt=path.with_suffix('.json');train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
    if receipt.exists():item=json.loads(receipt.read_text(encoding='utf-8'));assert sha(path)==item['model_sha256']
    else:
        parent=REPO/'model/cnn-v1'/f'fold_{fold}.npz';old,mean,std=load_model(parent);initial=CNNV2(config);initial.load_state_dict(old.state_dict())
        assert set(rows[i]['src_file'] for i in train).isdisjoint(set(rows[i]['src_file'] for i in test))
        model=adapt(initial.eval(),mean,std,train,config,5000+fold);export(path,model,mean,std,TEST_FOLD=np.asarray(fold),BASE_MODEL_SHA256=np.asarray(sha(parent)))
        item={'fold':fold,'config':config,'model_sha256':sha(path),'base_model_sha256':sha(parent),'source_overlap':0}
        for mode in ['float','int8']:
            model,mean,std=load(path,mode=='int8');probs=predict(model,views[0,test],mean,std);pred=probs.argmax(1);item[mode]=metrics(y[test],pred)
            item[mode]['predictions']=[{'filename':rows[i]['filename'],'truth':int(y[i]),'prediction':int(p),'probabilities':v.tolist()} for i,p,v in zip(test,pred,probs)]
        save(receipt,item);print('FIXED RECIPE CV',fold,item['int8']['correct'],'/',len(test),flush=True)
    truth_cv.extend(y[test].tolist())
    for mode in pooled:pooled[mode].extend(r['prediction'] for r in item[mode]['predictions'])
    selection['per_outer_fold'].append({'fold':fold,'config':config,'model_sha256':sha(path)});results.append(item)
selection['fixed_config_cv']={mode:metrics(truth_cv,p) for mode,p in pooled.items()};save(OUT/'selection.json',selection)
save(OUT/'summary.json',{'protocol':protocol,'selection':selection,'candidates':evaluated,'outer_folds':results,'final_model_sha256':sha(models/'final_all_esc50.npz'),
                       'external_used':True,'external_used_for_training':False,'external_used_for_selection':True})
print('DEVELOPMENT SELECTION COMPLETE',config['name'],flush=True)
