"""Warm-start v1, adapt gently to phone bandwidth, retain clean knowledge.

Inner baselines train on THREE folds from scratch, never initialize them from
four-fold checkpoints that have already seen the inner validation fold.
Outer/final initialization uses the corresponding frozen v1 checkpoints.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import copy,datetime,hashlib,json,pathlib,time
import numpy as np
import torch
from torch.nn import functional as F
from cnn_v2 import CNNV2,CONFIGS,export,load,predict,metrics,REPO
from causal_cnn import load_model
REPO=pathlib.Path(__file__).resolve().parents[2]
ROOT=pathlib.Path(os.environ.get('CNN_V2_RUN_DIR',str(REPO/'out/cnn-v2-training'))).resolve()
ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/'artifacts/warmstart-round3';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);start=time.monotonic()
base={**CONFIGS[0],'name':'v1_base','augmentation':'legacy','pooling':'mean'}
configs=[{**base,'name':'phone_anchor','phone':True,'lr':.0001,'baby_boost':1.5,'anchor':.5,'epochs':20},
         {**base,'name':'phone_strong_anchor','phone':True,'lr':.0001,'baby_boost':1.5,'anchor':1.0,'epochs':20},
         {**base,'name':'phone_no_anchor','phone':True,'lr':.0001,'baby_boost':1.5,'anchor':0.0,'epochs':20},
         {**base,'name':'phone_faster','phone':True,'lr':.0003,'baby_boost':1.5,'anchor':.5,'epochs':20},
         {**base,'name':'clean_anchor','phone':False,'lr':.0001,'baby_boost':1.5,'anchor':.5,'epochs':20}]
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=json.loads((REPO/'training/cnn/esc50-selection.json').read_text(encoding='utf-8'))['records'];y=np.asarray([r['label_id'] for r in rows]);folds=np.asarray([int(r['fold']) for r in rows]);sources=np.asarray([r['src_file'] for r in rows])
with np.load(ROOT/'artifacts/bandwidth-robust-round2/features.npz',allow_pickle=False) as z:views=z['X'][:,:,:26,:].copy()
with np.load(ROOT/'artifacts/esc50-features.npz',allow_pickle=False) as z:assert np.array_equal(views[0],z['X'][0,:,:26,:])
protocol={'registered':datetime.datetime.now().astimezone().isoformat(),'configs':configs,'base_recipe':'Exact v1 architecture/mean pooling, 80-epoch legacy gain/masks/noise training',
          'inner':'For outer fold f, train v1 baseline on the other three folds excluding next-fold validation. Fine-tune each candidate on those three only. Equal clean/anti-aliased-phone inner utility selects candidate.',
          'outer':'Initialize matching frozen v1 four-fold model, retaining its clean train-only normalization; fine-tune selected recipe on four folds, then score untouched outer fold',
          'final':'Average inner utility selects recipe; initialize immutable v1 final trained on390, fine-tune20 on the same390',
          'anchor':'KL with frozen starting teacher on clean TRAINING inputs only, temperature2; teacher never trained on validation/test folds',
          'augmentation':'Clean/anti-aliased8k view50/50 for phone candidates; +/-6dB gain and normalized Gaussian noise .01; no additional masks',
          'utility':'0.4*(clean macro+synthetic-phone macro)+0.1*(clean baby+synthetic-phone baby)',
          'external_training':False,'external_opened':False,'external_role':'Previously seen development-validation, not untouched final test',
          'feature_cache_sha256':sha(ROOT/'artifacts/bandwidth-robust-round2/features.npz'),
          'v1_initializers':{p.name:sha(p) for p in (REPO/'model/cnn-v1').glob('*.npz')},
          'seed':'Inner baseline1000+fold; inner adaptation3000+fold; outer adaptation5000+fold; final5242; no seed search'}
if __name__=='__main__':assert not (OUT/'summary.json').exists()
if not (OUT/'protocol.json').exists():save(OUT/'protocol.json',protocol)
else:assert json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))['configs']==configs
def criterion(indices,boost=1):
    weights=1/np.sqrt(np.maximum(np.bincount(y[indices],minlength=5),1));weights[3]*=boost;weights/=weights.mean()
    return torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32),label_smoothing=.05)
def train_base(indices,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);v=views[0,indices]
    mean=v.mean((0,2),dtype=np.float64).astype(np.float32).reshape(1,26,1);std=v.std((0,2),dtype=np.float64).astype(np.float32).reshape(1,26,1);std[std<1e-4]=1
    x=torch.from_numpy((v-mean)/std);target=torch.from_numpy(y[indices]);norm=torch.from_numpy(std);model=CNNV2(base);loss_fn=criterion(indices)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.005);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80,eta_min=.00005)
    for _ in range(80):
        model.train();order=rng.permutation(len(indices))
        for offset in range(0,len(order),32):
            ix=order[offset:offset+32];batch=x[ix].clone();batch+=torch.empty((len(ix),1,1)).uniform_(-2.7631,2.7631)/norm;batch+=torch.randn_like(batch)*.02
            for sample in batch:
                width=int(rng.integers(0,4))
                if width:lo=int(rng.integers(0,27-width));sample[lo:lo+width]=0
                if rng.random()<.3:width=int(rng.integers(1,31));lo=int(rng.integers(0,499-width));sample[:,lo:lo+width]=0
            opt.zero_grad(set_to_none=True);loss=loss_fn(model(batch),target[ix]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        schedule.step()
    return model.eval(),mean,std
def adapt(initial,mean,std,indices,config,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=CNNV2(config);model.load_state_dict(initial.state_dict());teacher=copy.deepcopy(initial).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    x=torch.from_numpy((views[:,indices]-mean)/std);target=torch.from_numpy(y[indices]);norm=torch.from_numpy(std)
    with torch.no_grad():teacher_probs=torch.cat([F.softmax(teacher(batch)/2,dim=1) for batch in x[0].split(32)])
    loss_fn=criterion(indices,config['baby_boost']);opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.005);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20,eta_min=config['lr']/10)
    for _ in range(20):
        model.train();order=rng.permutation(len(indices))
        for offset in range(0,len(order),32):
            ix=order[offset:offset+32];view=rng.integers(0,2,len(ix)) if config['phone'] else np.zeros(len(ix),dtype=int);batch=x[view,ix].clone()
            batch+=torch.empty((len(ix),1,1)).uniform_(-1.38155,1.38155)/norm;batch+=torch.randn_like(batch)*.01
            opt.zero_grad(set_to_none=True);loss=loss_fn(model(batch),target[ix])
            if config['anchor']:
                loss+=config['anchor']*4*F.kl_div(F.log_softmax(model(x[0,ix])/2,dim=1),teacher_probs[ix],reduction='batchmean')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        schedule.step()
    return model.eval()
def validation(model,mean,std,test):
    vals=[]
    for view in [0,1]:
        p=predict(model,views[view,test],mean,std);vals.append(metrics(y[test],p.argmax(1)))
    a,b=vals;return {'clean':a,'synthetic_phone':b,'utility':.4*(a['balanced_accuracy_present_classes']+b['balanced_accuracy_present_classes'])+.1*(a['per_class_recall']['baby_cry']+b['per_class_recall']['baby_cry'])}
if __name__=='__main__':
    inner=[]
    for fold in range(1,6):
        vfold=fold%5+1;train=np.flatnonzero((folds!=fold)&(folds!=vfold));test=np.flatnonzero(folds==vfold)
        assert set(sources[train]).isdisjoint(set(sources[test]))
        path=OUT/'inner'/f'baseline_{fold}.npz'
        if path.exists():initial,mean,std=load(path)
        else:initial,mean,std=train_base(train,1000+fold);export(path,initial,mean,std)
        items=[]
        for config in configs:
            path=OUT/'inner'/config['name']/f'fold_{fold}.npz';receipt=path.with_suffix('.json')
            if receipt.exists():item=json.loads(receipt.read_text(encoding='utf-8'));assert sha(path)==item['model_sha256']
            else:
                model=adapt(initial,mean,std,train,config,3000+fold);export(path,model,mean,std);item={'fold':fold,'config':config,**validation(model,mean,std,test),'model_sha256':sha(path)};save(receipt,item)
                print('INNER3',fold,config['name'],round(item['utility'],4),'elapsed',round(time.monotonic()-start,1),flush=True)
            items.append(item)
        inner.append(items);save(OUT/'progress.json',{'phase':'inner','fold':fold,'total_folds':5})
    chosen=[max(items,key=lambda a:a['utility']) for items in inner]
    final=max(configs,key=lambda c:np.mean([next(i for i in items if i['config']['name']==c['name'])['utility'] for items in inner]))
    selection={'per_outer_fold':[{'fold':i+1,'config':item['config'],'inner_utility':item['utility']} for i,item in enumerate(chosen)],'final_config':final,'final_epochs':20,'selected_without_external_audio':True,'config_selection_basis':'Clean+synthetic-phone inner only, v1 warm start with clean anchor'}
    save(OUT/'selection.json',selection)
    models=OUT/'selected';models.mkdir(exist_ok=True);truth=[];pooled={m:[] for m in ['float','int8']};outer=[]
    def initializer(name,config):
        p=REPO/'model/cnn-v1'/(name+'.npz');old,mean,std=load_model(p);model=CNNV2(config);model.load_state_dict(old.state_dict());return model.eval(),mean,std,p
    for fold,item in enumerate(chosen,1):
        config=item['config'];train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold);path=models/f'fold_{fold}.npz';receipt=path.with_suffix('.json')
        if receipt.exists():result=json.loads(receipt.read_text(encoding='utf-8'));assert sha(path)==result['model_sha256']
        else:
            initial,mean,std,parent=initializer(f'fold_{fold}',config);model=adapt(initial,mean,std,train,config,5000+fold);export(path,model,mean,std,TEST_FOLD=np.asarray(fold),BASE_MODEL_SHA256=np.asarray(sha(parent)))
            result={'fold':fold,'config':config,'model_sha256':sha(path),'base_model_sha256':sha(parent)}
            for mode in ['float','int8']:
                model,mean,std=load(path,mode=='int8');p=predict(model,views[0,test],mean,std);pred=p.argmax(1);result[mode]=metrics(y[test],pred)
                result[mode]['predictions']=[{'filename':rows[i]['filename'],'truth':int(y[i]),'prediction':int(v),'probabilities':probs.tolist()} for i,v,probs in zip(test,pred,p)]
            save(receipt,result);print('OUTER3',fold,config['name'],result['int8']['accuracy'],'baby',result['int8']['per_class_recall']['baby_cry'],flush=True)
        truth.extend(y[test].tolist())
        for mode in pooled:pooled[mode].extend(r['prediction'] for r in result[mode]['predictions'])
        outer.append(result)
    selection['nested_cv']={mode:metrics(truth,p) for mode,p in pooled.items()};save(OUT/'selection.json',selection)
    initial,mean,std,parent=initializer('final_all_esc50',final);model=adapt(initial,mean,std,np.arange(390),final,5242);export(models/'final_all_esc50.npz',model,mean,std,TRAINING_RECORDINGS=np.asarray(390),BASE_MODEL_SHA256=np.asarray(sha(parent)))
    save(OUT/'summary.json',{'protocol':protocol,'inner':inner,'outer_folds':outer,'selection':selection,'final_model_sha256':sha(models/'final_all_esc50.npz'),'elapsed_seconds':time.monotonic()-start,'external_used':False})
    print('ROUND3 COMPLETE',json.dumps(selection),flush=True)
