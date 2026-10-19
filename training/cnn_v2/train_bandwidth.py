"""Round 2: anti-aliased phone bandwidth, fixed epochs and nested selection.

Registered after round 1 failed development validation. External recordings
remain excluded from gradients, normalization and this program's selection.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='4'
import datetime,hashlib,json,pathlib,shutil,time
import numpy as np
import torch
from scipy.signal import resample_poly
from cnn_v2 import CNNV2,CONFIGS,all_features,select_features,export,load,predict,metrics,read_wav_mono,resample_linear,REPO
REPO=pathlib.Path(__file__).resolve().parents[2]
ROOT=pathlib.Path(os.environ.get('CNN_V2_RUN_DIR',str(REPO/'out/cnn-v2-training'))).resolve()
ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/'artifacts/bandwidth-robust-round2';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);start=time.monotonic();EPOCHS=80
base={**CONFIGS[0],'augmentation':'none'}
configs=[{**base,'name':'gentle_bandwidth','augmentation':'bandwidth'},
         {**base,'name':'lowband_clean','feature':'lowband'},
         {**base,'name':'lowband_bandwidth','feature':'lowband','augmentation':'bandwidth'},
         {**base,'name':'hybrid_clean','feature':'hybrid'},
         {**base,'name':'logmel_clean_80'}]
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol={'registered':datetime.datetime.now().astimezone().isoformat(),'round1_summary_sha256':sha(ROOT/'artifacts/training-suite/summary.json'),
          'reason':'Round 1 final 45/65 external, baby 11/20, worse than v1; not accepted for release. Its 16k->8k linear downsampling aliased high frequencies. Correct augmentation uses anti-aliasing before 8k reduction; fixed 80 epochs avoid early-epoch refit instability.',
          'configs':configs,'epochs':EPOCHS,'selection':'Five official outer folds; next fold cyclically validates inner models trained on other three folds. Select config per outer using inner utility only; retrain selected config on four folds. Final config by mean inner utility, fit all 390.',
          'utility':'0.4*(clean macro recall + phone macro recall) + 0.1*(clean baby recall + phone baby recall); equal clean and synthetic-phone inner validation',
          'augmentation':'Only clean / anti-aliased 8k phone view, 50/50; no clipping, masking, gain or Gaussian feature noise',
          'phone_view':'scipy.signal.resample_poly(...,1,2,window=(kaiser,5.0)); clip/round to PCM16 8k; original v11 linear 8k->16k conversion',
          'lowband':'First 19 original v11 mel filters, nonzero support ends at 3781.25 Hz; discard unavailable >4k phone information',
          'seeds':'Inner 1000+fold, outer 2000+fold, final4242; one seed per fit, no external seed hunting',
          'external_training':False,'external_opened_by_this_program':False,'normalization':'Clean train folds only',
          'optimizer':'AdamW .002, wd .005, cosine80 to .00005, label smoothing .05, inverse sqrt class weights, dropout .15, batch32',
          'external_role':'Previously seen development-validation; not untouched final test'}
assert not (OUT/'summary.json').exists()
if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))['configs']==configs
else:save(OUT/'protocol.json',protocol)
rows=json.loads((REPO/'training/cnn/esc50-selection.json').read_text(encoding='utf-8'))['records']
source=pathlib.Path(os.environ['ESC50_DATASET'])
cache=OUT/'features.npz'
prior=ROOT/'artifacts/bandwidth-round2/features.npz'
if not cache.exists() and prior.exists():shutil.copyfile(prior,cache)
if not cache.exists():
    clean=[];phone=[]
    for r in rows:
        p=source/'audio'/r['filename'];assert sha(p)==r['raw_sha256'];pcm=read_wav_mono(p,16000)
        low=np.clip(np.round(resample_poly(pcm.astype(np.float32),1,2,window=('kaiser',5.0))),-32768,32767).astype(np.int16)
        restored=resample_linear(low,8000,16000);clean.append(all_features(pcm));phone.append(all_features(restored))
    np.savez_compressed(cache,X=np.stack([clean,phone]))
with np.load(cache,allow_pickle=False) as z:views=z['X']
assert views.shape==(2,390,31,498)
y=np.asarray([r['label_id'] for r in rows]);folds=np.asarray([int(r['fold']) for r in rows]);sources=np.asarray([r['src_file'] for r in rows])
def fit(config,indices,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);v=select_features(views,config['feature']).astype(np.float32);clean=v[0,indices]
    mean=clean.mean((0,2),dtype=np.float64).astype(np.float32).reshape(1,-1,1);std=clean.std((0,2),dtype=np.float64).astype(np.float32).reshape(1,-1,1);std[std<1e-4]=1
    x=torch.from_numpy((v[:,indices]-mean)/std);target=torch.from_numpy(y[indices]);weights=1/np.sqrt(np.maximum(np.bincount(y[indices],minlength=5),1));weights/=weights.mean()
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32),label_smoothing=.05)
    model=CNNV2(config);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.005);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80,eta_min=.00005)
    for epoch in range(80):
        model.train();order=rng.permutation(len(indices))
        for offset in range(0,len(order),32):
            batch=order[offset:offset+32];view=rng.integers(0,2,len(batch)) if config['augmentation']=='bandwidth' else np.zeros(len(batch),dtype=int)
            opt.zero_grad(set_to_none=True);loss=criterion(model(x[view,batch]),target[batch]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        schedule.step()
    return model.eval(),mean,std
inner=[]
for config in configs:
    scores=[]
    for fold in range(1,6):
        receipt=OUT/'inner'/config['name']/f'fold_{fold}.json'
        if receipt.exists():item=json.loads(receipt.read_text(encoding='utf-8'))
        else:
            val=fold%5+1;train=np.flatnonzero((folds!=fold)&(folds!=val));test=np.flatnonzero(folds==val)
            assert set(sources[train]).isdisjoint(set(sources[test]))
            model,mean,std=fit(config,train,1000+fold)
            path=receipt.with_suffix('.npz');export(path,model,mean,std,OUTER_FOLD_EXCLUDED=np.asarray(fold),VALIDATION_FOLD=np.asarray(val))
            scores_by_view=[]
            for view in [0,1]:
                p=predict(model,select_features(views[view,test],config['feature']),mean,std);scores_by_view.append(metrics(y[test],p.argmax(1)))
            m,phone=scores_by_view
            item={'fold':fold,'validation_fold':val,'clean_metrics':m,'synthetic_phone_metrics':phone,'model_sha256':sha(path),
                  'utility':.4*(m['balanced_accuracy_present_classes']+phone['balanced_accuracy_present_classes'])+.1*(m['per_class_recall']['baby_cry']+phone['per_class_recall']['baby_cry'])}
            save(receipt,item);print('INNER',config['name'],fold,'utility',round(item['utility'],4),'elapsed',round(time.monotonic()-start,1),flush=True)
        scores.append(item);save(OUT/'progress.json',{'phase':'inner','config':config['name'],'fold':fold,'completed':len(inner)*5+len(scores),'total_inner':25})
    inner.append({'config':config,'scores':scores,'mean_utility':float(np.mean([s['utility'] for s in scores]))})
chosen=[];final=max(inner,key=lambda e:e['mean_utility'])['config']
for fold in range(1,6):
    entry=max(inner,key=lambda e:e['scores'][fold-1]['utility']);chosen.append({'fold':fold,'config':entry['config'],'inner_utility':entry['scores'][fold-1]['utility']})
selection={'per_outer_fold':chosen,'final_config':final,'final_epochs':80,'selected_without_external_audio':True,'config_selection_basis':'Round2 equal clean/synthetic-phone inner utility only, fixed80, no external score/seed selection'}
save(OUT/'selection.json',selection) # Freeze before outer inference or external audio.
models=OUT/'selected';models.mkdir(exist_ok=True);truth=[];pooled={m:[] for m in ['float','int8']};outer=[]
for entry in chosen:
    fold=entry['fold'];config=entry['config'];path=models/f'fold_{fold}.npz';receipt=path.with_suffix('.json');train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
    if receipt.exists():item=json.loads(receipt.read_text(encoding='utf-8'));assert sha(path)==item['model_sha256']
    else:
        assert set(sources[train]).isdisjoint(set(sources[test]));model,mean,std=fit(config,train,2000+fold);export(path,model,mean,std,TEST_FOLD=np.asarray(fold),EPOCHS=np.asarray(80))
        item={'fold':fold,'config':config,'model_sha256':sha(path)}
        for mode in ['float','int8']:
            model,mean,std=load(path,mode=='int8');p=predict(model,select_features(views[0,test],config['feature']),mean,std);pred=p.argmax(1);item[mode]=metrics(y[test],pred)
            item[mode]['predictions']=[{'filename':r['filename'],'truth':int(t),'prediction':int(v),'probabilities':probs.tolist()} for r,t,v,probs in zip(np.asarray(rows,dtype=object)[test],y[test],pred,p)]
            phone=predict(model,select_features(views[1,test],config['feature']),mean,std)
            item[mode]['synthetic_phone_stress']=metrics(y[test],phone.argmax(1))
        save(receipt,item);print('OUTER',fold,config['name'],item['int8']['accuracy'],'baby',item['int8']['per_class_recall']['baby_cry'],flush=True)
    truth.extend(y[test].tolist())
    for mode in pooled:pooled[mode].extend(r['prediction'] for r in item[mode]['predictions'])
    outer.append(item)
selection['nested_cv']={mode:metrics(truth,p) for mode,p in pooled.items()};save(OUT/'selection.json',selection)
model,mean,std=fit(final,np.arange(390),4242);export(models/'final_all_esc50.npz',model,mean,std,TRAINING_RECORDINGS=np.asarray(390),EPOCHS=np.asarray(80))
save(OUT/'summary.json',{'protocol':protocol,'inner_candidates':inner,'outer_folds':outer,'selection':selection,'final_model_sha256':sha(models/'final_all_esc50.npz'),'elapsed_seconds':time.monotonic()-start,'external_used':False})
print('ROUND2 COMPLETE',json.dumps(selection),flush=True)
