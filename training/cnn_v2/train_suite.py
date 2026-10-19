"""Nested selection + controlled ablations, using only fixed ESC-50 recordings.

The 65 external recordings are deliberately not opened by this program.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='4'
import copy,datetime,hashlib,json,pathlib,shutil,time
import numpy as np
import torch
from cnn_v2 import CNNV2,CONFIGS,select_features,gain_vector,export,load,predict,metrics,LABELS

REPO=pathlib.Path(__file__).resolve().parents[2]
ROOT=pathlib.Path(os.environ.get('CNN_V2_RUN_DIR',str(REPO/'out/cnn-v2-training'))).resolve()
ROOT.mkdir(parents=True,exist_ok=True)
OUT=ROOT/'artifacts/training-suite';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
MAX_EPOCHS=80;MIN_EPOCHS=32;PATIENCE=16;BATCH=32
start=time.monotonic()
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

protocol={'created':datetime.datetime.now().astimezone().isoformat(),'configs':CONFIGS,'outer':'Five official ESC-50 folds; original source IDs disjoint',
          'inner':'For each outer test fold, the next official fold cyclically is validation; only remaining three folds train inner models',
          'selection':'Per outer fold select config/epoch using its inner validation only; refit that config on four outer training folds; pool untouched outer predictions',
          'ablation':'Every fixed variant uses the same inner split/seed/optimizer and its own inner-selected epoch before outer refitting',
          'final':'Select configuration by mean inner-validation utility; median inner-selected epoch, refit all 390 ESC-50 recordings',
          'utility':'0.8 * five-class macro recall + 0.2 * baby recall; external scores are never used for selection',
          'normalization':'Clean training folds only, separately recomputed for inner and outer fits',
          'phone_augmentation':'Random training-only view among clean, 16k->8k->16k, gain/clipping+8k, 8k+15..30dB Gaussian noise; feature gain +/-12dB, normalized noise .02 and feature/time masking',
          'external_audio_used_for_training':False,'external_audio_opened_by_this_program':False,
          'max_epochs':MAX_EPOCHS,'min_epochs_before_early_stop':MIN_EPOCHS,'early_stopping_patience':PATIENCE,'validation_every_epochs':4,
          'batch_size':BATCH,'lr':.002,'weight_decay':.005,'label_smoothing':.05,'dropout':.15,'class_weight_power':.5,
          'seeds':'Inner 1000+outer_fold; outer 2000+outer_fold; final 4242','torch':torch.__version__,'numpy':np.__version__,
          'not_an_untouched_external_test':'The user-confirmed 65 are an existing development-validation cohort; improvements there are developmental evidence'}
if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))['configs']==CONFIGS
else:save(OUT/'protocol.json',protocol)
assert not (OUT/'summary.json').exists(),'Preserve completed suite'
with np.load(ROOT/'artifacts/esc50-features.npz',allow_pickle=False) as z:
    all_views,y,folds,files,sources=[z[k].copy() for k in ['X','Y','FOLDS','FILES','SOURCES']]
assert all_views.shape==(4,390,31,498)

def fit(config,views,indices,seed,epochs,validation=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    clean=views[0,indices]
    mean=clean.mean(axis=(0,2),dtype=np.float64).astype(np.float32).reshape(1,clean.shape[1],1)
    std=clean.std(axis=(0,2),dtype=np.float64).astype(np.float32).reshape(1,clean.shape[1],1);std[std<1e-4]=1
    tensor=torch.from_numpy((views[:,indices]-mean)/std);targets=torch.from_numpy(y[indices])
    counts=np.bincount(y[indices],minlength=5);weights=1/np.sqrt(np.maximum(counts,1));weights/=weights.mean()
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32),label_smoothing=.05)
    model=CNNV2(config);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.005)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=MAX_EPOCHS,eta_min=.00005)
    gain=torch.from_numpy(gain_vector(config['feature']).reshape(1,-1,1)/std)
    best_utility=-1;best_epoch=0;best_state=None;history=[];best_metrics=None
    for epoch in range(1,epochs+1):
        model.train();order=rng.permutation(len(indices));loss_sum=0
        for offset in range(0,len(order),BATCH):
            subset=order[offset:offset+BATCH]
            view=rng.integers(0,4,len(subset)) if config['augmentation']=='phone' else np.zeros(len(subset),dtype=int)
            batch=tensor[view,subset].clone()
            if config['augmentation']!='none':
                batch+=torch.empty((len(subset),1,1)).uniform_(-2.7631,2.7631)*gain
                batch+=torch.randn_like(batch)*.02
                for sample in batch:
                    width=int(rng.integers(0,min(3,sample.shape[0]-1)+1))
                    if width:
                        lo=int(rng.integers(0,sample.shape[0]-width+1));sample[lo:lo+width]=0
                    if rng.random()<.3:
                        width=int(rng.integers(1,31));lo=int(rng.integers(0,sample.shape[1]-width+1));sample[:,lo:lo+width]=0
            opt.zero_grad(set_to_none=True);loss=loss_fn(model(batch),targets[subset]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();loss_sum+=float(loss.detach())*len(subset)
        sched.step()
        if validation is not None and epoch%4==0:
            probs=predict(model,views[0,validation],mean,std);m=metrics(y[validation],probs.argmax(axis=1))
            utility=.8*m['balanced_accuracy_present_classes']+.2*m['per_class_recall']['baby_cry']
            history.append({'epoch':epoch,'loss':loss_sum/len(indices),'utility':utility,'accuracy':m['accuracy'],'macro_recall':m['balanced_accuracy_present_classes'],'baby_recall':m['per_class_recall']['baby_cry']})
            if utility>best_utility+1e-8:
                best_utility=utility;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());best_metrics=m
            if epoch>=MIN_EPOCHS and epoch-best_epoch>=PATIENCE:break
    if validation is not None:
        model.load_state_dict(best_state)
    return model.eval(),mean,std,{'selected_epoch':best_epoch if validation is not None else epochs,'utility':best_utility if validation is not None else None,
                                  'validation_metrics':best_metrics,'history':history,'epochs_executed':epoch}

variants=[]
for config in CONFIGS:
    views=select_features(all_views,config['feature']).astype(np.float32)
    entry={'config':config,'folds':[]};predictions={mode:[] for mode in ['float','int8']};truth=[]
    for fold in range(1,6):
        path=OUT/'variants'/config['name']/f'fold_{fold}.npz';receipt=path.with_suffix('.json')
        if receipt.exists():
            item=json.loads(receipt.read_text(encoding='utf-8'));assert sha(path)==item['model_sha256']
            print('RESUME',config['name'],fold,flush=True)
        else:
            inner_val=fold%5+1
            inner_train=np.flatnonzero((folds!=fold)&(folds!=inner_val));validation=np.flatnonzero(folds==inner_val)
            outer_train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
            assert set(sources[inner_train]).isdisjoint(set(sources[validation]))
            assert set(sources[outer_train]).isdisjoint(set(sources[test]))
            _,_,_,inner=fit(config,views,inner_train,1000+fold,MAX_EPOCHS,validation)
            chosen_epoch=inner['selected_epoch']
            model,mean,std,_=fit(config,views,outer_train,2000+fold,chosen_epoch)
            export(path,model,mean,std,TEST_FOLD=np.asarray(fold),EPOCHS=np.asarray(chosen_epoch))
            item={'fold':fold,'inner_validation_fold':inner_val,'inner':inner,'train_recordings':len(outer_train),
                  'test_recordings':len(test),'source_overlap':0,'model_sha256':sha(path),'parameter_count':sum(p.numel() for p in model.parameters())}
            for mode in ['float','int8']:
                model,mean,std=load(path,quantized=mode=='int8');probs=predict(model,views[0,test],mean,std);pred=probs.argmax(axis=1)
                item[mode]=metrics(y[test],pred)
                item[mode]['predictions']=[{'filename':str(files[i]),'truth':int(y[i]),'prediction':int(p),'probabilities':v.tolist()} for i,p,v in zip(test,pred,probs)]
            save(receipt,item)
            print(config['name'],'fold',fold,'epoch',chosen_epoch,'outer accuracy',round(item['int8']['accuracy'],4),'baby',item['int8']['per_class_recall']['baby_cry'],'elapsed',round(time.monotonic()-start,1),flush=True)
        entry['folds'].append(item)
        for mode in predictions:predictions[mode].extend(r['prediction'] for r in item[mode]['predictions'])
        truth.extend(r['truth'] for r in item['int8']['predictions'])
        save(OUT/'progress.json',{'variant':config['name'],'fold':fold,'completed_fits':sum(len(v['folds']) for v in variants)+len(entry['folds']),'total_outer_fits':len(CONFIGS)*5,'elapsed_seconds':time.monotonic()-start})
    entry['cv']={mode:metrics(truth,pred) for mode,pred in predictions.items()}
    entry['mean_inner_utility']=float(np.mean([f['inner']['utility'] for f in entry['folds']]))
    save(OUT/'variants'/config['name']/'summary.json',entry);variants.append(entry)

selected=[];nested_truth=[];nested_pred={mode:[] for mode in ['float','int8']}
selected_dir=OUT/'selected';selected_dir.mkdir(exist_ok=True)
for fold in range(1,6):
    chosen=max(variants,key=lambda v:v['folds'][fold-1]['inner']['utility'])
    item=chosen['folds'][fold-1]
    shutil.copy2(OUT/'variants'/chosen['config']['name']/f'fold_{fold}.npz',selected_dir/f'fold_{fold}.npz')
    selected.append({'fold':fold,'config':chosen['config'],'selected_epoch':item['inner']['selected_epoch'],'inner_utility':item['inner']['utility'],'model_sha256':item['model_sha256']})
    nested_truth.extend(r['truth'] for r in item['int8']['predictions'])
    for mode in nested_pred:nested_pred[mode].extend(r['prediction'] for r in item[mode]['predictions'])
best=max(variants,key=lambda v:v['mean_inner_utility'])
epochs=int(np.median([f['inner']['selected_epoch'] for f in best['folds']]))
selection={'per_outer_fold':selected,'final_config':best['config'],'final_epochs':epochs,
           'selected_without_external_audio':True,'config_selection_basis':'Mean inner utility only; no external inference loaded',
           'nested_cv':{mode:metrics(nested_truth,pred) for mode,pred in nested_pred.items()}}
save(OUT/'selection.json',selection) # Freeze choice BEFORE any external evaluation.
views=select_features(all_views,best['config']['feature']).astype(np.float32)
model,mean,std,_=fit(best['config'],views,np.arange(390),4242,epochs)
export(selected_dir/'final_all_esc50.npz',model,mean,std,TRAINING_RECORDINGS=np.asarray(390),EPOCHS=np.asarray(epochs))
summary={'protocol':protocol,'variants':variants,'selection':selection,'final_model_sha256':sha(selected_dir/'final_all_esc50.npz'),
         'elapsed_seconds':time.monotonic()-start,'external_used':False}
save(OUT/'summary.json',summary)
print('SUITE COMPLETE',json.dumps({'selected':best['config'],'epochs':epochs,'nested_cv':selection['nested_cv'],'seconds':summary['elapsed_seconds']}),flush=True)
