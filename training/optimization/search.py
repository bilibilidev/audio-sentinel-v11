"""ESC-only nested candidate search for compact MLP and causal CNN.

Outer folds never select configurations/epochs. External files are not read.
Every inner, outer and final fit starts from fresh random weights.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,copy,datetime,hashlib,json,math,pathlib,sys,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'training'));sys.path.insert(0,str(REPO/'training/cnn_v2'))
from cnn_v2 import CNNV2,select_features,gain_vector,export,metrics,LABELS
from train_edge_mfcc import save_model

MLP_BASE=dict(kind='mlp',channels=64,feature='mfcc92',loss='window',augmentation='none',dropout=.1,
              lr=.002,wd=.001,smoothing=.02,epochs=240,ema=0.,transform='none')
MLP_CONFIGS=[dict(name='window64_clean',dropout=0.),
 dict(name='window96_mild',channels=96,augmentation='mild',wd=.0003),
 dict(name='window128_mild',channels=128,augmentation='mild',dropout=.15),
 dict(name='mil64_clean',loss='bag'),
 dict(name='mil96_mild',channels=96,loss='bag',augmentation='mild'),
 dict(name='mil96_asinh',channels=96,loss='bag',augmentation='mild',transform='asinh40',wd=.0003),
 dict(name='window96_asinh',channels=96,augmentation='mild',transform='asinh40',dropout=.15),
 dict(name='mil128_asinh_ema',channels=128,loss='bag',augmentation='mild',transform='asinh40',dropout=.15,ema=.98)]
MLP_CONFIGS=[{**MLP_BASE,**c} for c in MLP_CONFIGS]
CNN_BASE=dict(kind='cnn',feature='logmel',channels=32,dilations=[1,2,4,8,16],pooling='mean',
              augmentation='none',dropout=.05,lr=.0015,wd=.0001,smoothing=.01,epochs=160,ema=0.,loss='bag')
CNN_CONFIGS=[dict(name='clean32_mean'),
 dict(name='mild32_mean',augmentation='mild',dropout=.1,wd=.001,smoothing=.02),
 dict(name='hybrid32_lme',feature='hybrid',pooling='mixed_lme',augmentation='mild',wd=.001),
 dict(name='clean48_lme_ema',channels=48,pooling='mixed_lme',dropout=.1,wd=.001,ema=.98),
 dict(name='lowband32_phone',feature='lowband',augmentation='phone_mild',dropout=.1,wd=.001),
 dict(name='medium32_mixup',dilations=[1,2,4,8,8],augmentation='mixup',dropout=.1,wd=.001)]
CNN_CONFIGS=[{**CNN_BASE,**c} for c in CNN_CONFIGS]

class TinyMLP(nn.Module):
    def __init__(self,c):
        super().__init__();self.config=c
        self.fc1=nn.Linear(92,c['channels']);self.drop=nn.Dropout(c['dropout']);self.fc2=nn.Linear(c['channels'],5)
    def frames(self,x):return self.fc2(self.drop(F.relu(self.fc1(x))))
    def forward(self,x):return torch.softmax(self.frames(x),dim=-1).mean(dim=1)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def transform(x,c,scale):
    if c.get('transform')=='asinh40':
        x=x.copy();x[...,:40]=np.arcsinh(x[...,:40]/scale)
    return x
def normalize_fit(views,indices,c):
    clean=views[0,indices]
    if c['kind']=='mlp':
        scale=np.median(np.abs(clean[...,:40]),axis=(0,1)).astype(np.float32)
        scale=np.maximum(scale,np.float32(1e-6));mapped=transform(views,c,scale)
        train=mapped[0,indices];mean=train.mean((0,1),dtype=np.float64).astype(np.float32);std=train.std((0,1),dtype=np.float64).astype(np.float32)
        std[std<1e-6]=1
    else:
        scale=np.ones(40,dtype=np.float32);mapped=views
        mean=clean.mean((0,2),dtype=np.float64).astype(np.float32).reshape(1,clean.shape[1],1)
        std=clean.std((0,2),dtype=np.float64).astype(np.float32).reshape(1,clean.shape[1],1);std[std<1e-4]=1
    return mapped,mean,std,scale
def probabilities(model,x,c):
    model.eval();result=[]
    with torch.no_grad():
        for part in np.array_split(x,max(1,(len(x)+31)//32)):
            v=model(torch.from_numpy(part))
            result.append(v.numpy() if c['kind']=='mlp' else torch.softmax(v,dim=1).numpy())
    return np.concatenate(result)
def qmodel(model,c):
    result=copy.deepcopy(model)
    with torch.no_grad():
        for p in result.parameters():
            scale=max(float(p.abs().max()),1e-10)/127;p.copy_(p.div(scale).round().clamp(-127,127).mul(scale))
    return result.eval()

def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    configs=MLP_CONFIGS if args.kind=='mlp' else CNN_CONFIGS
    torch.set_num_threads(2 if args.kind=='cnn' else 1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    audit=args.audit.resolve();cache=args.cnn_cache.resolve()
    rows=json.loads((audit/'mlp-training/input-audit.json').read_text(encoding='utf-8'))['rows'][::5]
    files=np.asarray([r['filename'] for r in rows]);source=np.asarray([r['src_file'] for r in rows]);y=np.asarray([r['label_id'] for r in rows]);folds=np.asarray([r['fold'] for r in rows])
    assert len(rows)==390 and len(set(files))==390
    with np.load(audit/'mlp-training/clean-features.npz',allow_pickle=False) as z:
        mx=z['X'].reshape(390,5,92).copy();assert np.array_equal(z['Y'].reshape(390,5)[:,0],y)
    if args.kind=='mlp':
        with np.load(audit/'mlp-training/augmented-features.npz',allow_pickle=False) as z:aug=z['X_AUG'].reshape(4,390,5,92).copy()
        allviews=np.concatenate([mx[None],aug])
    else:
        with np.load(cache,allow_pickle=False) as z:
            lookup={str(f):i for i,f in enumerate(z['FILES'])};order=[lookup[str(f)] for f in files]
            assert np.array_equal(z['Y'][order],y) and np.array_equal(z['FOLDS'][order],folds)
            allviews=z['X'][:,order].copy()
    protocol=dict(created=datetime.datetime.now().astimezone().isoformat(),kind=args.kind,candidates=configs,
       outer='Official fold held out; all seconds and takes/src_file disjoint',inner='Next official fold cyclically is validation; remaining3 folds train',
       selection='0.6 accuracy +0.3 macro recall +0.1 baby recall on INNER validation only; epochs checked every10; ties choose earlier config/epoch',
       final='Best mean INNER utility across5 inner splits; median selected epoch; fresh random fit on all390',
       early_stop=dict(minimum_epochs=60 if args.kind=='mlp' else 50,patience=50 if args.kind=='mlp' else 40),
       normalization='Clean inner/outer TRAINING only; asinh scale also training-only',
       external_opened=False,pretrained=False,finetuned=False,teacher=False,source_sha256=sha(pathlib.Path(__file__)),
       source_data=dict(cnn_cache_sha256=sha(cache),mlp_clean_sha256=sha(audit/'mlp-training/clean-features.npz'),mlp_aug_sha256=sha(audit/'mlp-training/augmented-features.npz')),
       seed_base=2026102400,torch=torch.__version__,numpy=np.__version__,python=sys.version)
    if (out/'protocol.json').exists():
        old=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        assert all(old[k]==v for k,v in protocol.items() if k!='created')
    else:write(out/'protocol.json',protocol)
    protected={n:sha(REPO/n) for n in ['app/audiodetect/mlp_weights.h','app/audiodetect/cnn_weights_v2.h','model/mlp-retrained-20261024/final_all_esc50.npz']}
    assert not (out/'frozen.json').exists(),'Preserve completed search'
    start=time.monotonic();inner_records=[];outer_records=[];pooled={m:[] for m in ('float','int8')}

    def fit(c,indices,seed,epochs,validation=None,tag=''):
        views=allviews if c['kind']=='mlp' else select_features(allviews,c['feature']).astype(np.float32)
        mapped,mean,std,scale=normalize_fit(views,indices,c)
        tensor=torch.from_numpy((mapped[:,indices]-mean)/std);target=torch.from_numpy(y[indices])
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=TinyMLP(c) if c['kind']=='mlp' else CNNV2(c)
        if c['kind']=='cnn':model.dropout.p=c['dropout']
        counts=np.bincount(y[indices],minlength=5);weights=1/np.sqrt(np.maximum(counts,1));weights/=weights.mean()
        classw=torch.tensor(weights,dtype=torch.float32)
        opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['wd'])
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c['epochs'],eta_min=c['lr']*.025)
        ema=copy.deepcopy(model) if c['ema'] else None;best=-1.;best_epoch=0;best_state=None;best_metrics=None;history=[]
        gain=torch.from_numpy(gain_vector(c['feature']).reshape(1,-1,1)/std) if c['kind']=='cnn' else None
        for epoch in range(1,epochs+1):
            model.train();order=rng.permutation(len(indices));loss_sum=0.
            for off in range(0,len(order),32):
                subset=order[off:off+32]
                view=np.zeros(len(subset),dtype=int)
                if c['augmentation']!='none':
                    chance=np.where(y[indices[subset]]==0,.1,.3) if c['kind']=='mlp' else np.full(len(subset),.25 if c['augmentation']=='phone_mild' else 0.)
                    use=rng.random(len(subset))<chance
                    view[use]=rng.integers(1,len(views),int(use.sum()))
                batch=tensor[view,subset].clone();labels=target[subset]
                if c['kind']=='cnn' and c['augmentation'] in ('mild','phone_mild','mixup'):
                    batch+=torch.empty((len(subset),1,1)).uniform_(-1.38155,1.38155)*gain
                    batch+=torch.randn_like(batch)*.01
                    for sample in batch:
                        if rng.random()<.2:
                            width=int(rng.integers(1,11));lo=int(rng.integers(0,sample.shape[1]-width+1));sample[:,lo:lo+width]=0
                opt.zero_grad(set_to_none=True)
                if c['kind']=='mlp':
                    logits=model.frames(batch)
                    if c['loss']=='window':loss=F.cross_entropy(logits.reshape(-1,5),labels[:,None].expand(-1,5).reshape(-1),weight=classw,label_smoothing=c['smoothing'])
                    else:
                        lp=torch.logsumexp(F.log_softmax(logits,dim=-1),dim=1)-math.log(5)
                        costs=-(1-c['smoothing'])*lp[torch.arange(len(labels)),labels]-c['smoothing']*lp.mean(1)
                        w=classw[labels];loss=(costs*w).sum()/w.sum()
                else:
                    if c['augmentation']=='mixup':
                        lam=float(rng.beta(.2,.2));perm=torch.randperm(len(batch));logits=model(lam*batch+(1-lam)*batch[perm])
                        loss=lam*F.cross_entropy(logits,labels,weight=classw,label_smoothing=c['smoothing'])+(1-lam)*F.cross_entropy(logits,labels[perm],weight=classw,label_smoothing=c['smoothing'])
                    else:loss=F.cross_entropy(model(batch),labels,weight=classw,label_smoothing=c['smoothing'])
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();loss_sum+=float(loss.detach())*len(subset)
                if ema is not None:
                    with torch.no_grad():
                        for e,v in zip(ema.parameters(),model.parameters()):e.mul_(c['ema']).add_(v,alpha=1-c['ema'])
            sched.step();evaluate=ema if ema is not None else model
            if validation is not None and epoch%10==0:
                prob=probabilities(evaluate,(mapped[0,validation]-mean)/std,c);m=metrics(y[validation],prob.argmax(1))
                utility=.6*m['accuracy']+.3*m['balanced_accuracy_present_classes']+.1*m['per_class_recall']['baby_cry']
                history.append(dict(epoch=epoch,loss=loss_sum/len(indices),utility=utility,accuracy=m['accuracy'],macro=m['balanced_accuracy_present_classes']))
                if utility>best+1e-9:best=utility;best_epoch=epoch;best_state=copy.deepcopy(evaluate.state_dict());best_metrics=m
                minimum=60 if c['kind']=='mlp' else 50;patience=50 if c['kind']=='mlp' else 40
                if epoch>=minimum and epoch-best_epoch>=patience:break
            if epoch%10==0:write(out/'progress.json',dict(stage=tag,epoch=epoch,max_epochs=epochs,elapsed_seconds=time.monotonic()-start))
        result=copy.deepcopy(ema if ema is not None else model).eval()
        if validation is not None:result.load_state_dict(best_state)
        return result,mean,std,scale,dict(selected_epoch=best_epoch if validation is not None else epochs,utility=best if validation is not None else None,validation_metrics=best_metrics,history=history,executed_epochs=epoch)

    def save_model_file(path,model,c,mean,std,scale,fold):
        if args.kind=='cnn':export(path,model,mean,std,TEST_FOLD=np.asarray(fold),RANDOM_INITIALIZATION=np.asarray(True))
        else:
            weights=[dict(W=model.fc1.weight.detach().numpy().T,B=model.fc1.bias.detach().numpy()),dict(W=model.fc2.weight.detach().numpy().T,B=model.fc2.bias.detach().numpy())]
            save_model(path,weights,mean,std,dict(TEST_FOLD=np.asarray(fold),CONFIG=np.asarray(json.dumps(c,sort_keys=True)),TIME_ASINH_SCALE=scale,RANDOM_INITIALIZATION=np.asarray(True)))

    for fold in range(1,6):
        valfold=fold%5+1;tr=np.flatnonzero((folds!=fold)&(folds!=valfold));va=np.flatnonzero(folds==valfold)
        outertr=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
        assert set(source[tr]).isdisjoint(source[va]) and set(source[outertr]).isdisjoint(source[test])
        candidates=[]
        for config in configs:
            c=dict(config);receipt=out/'inner'/f'fold_{fold}'/(c['name']+'.json')
            if receipt.exists():item=json.loads(receipt.read_text(encoding='utf-8'))
            else:
                model,mean,std,scale,info=fit(c,tr,2026102400+fold,c['epochs'],va,f'inner fold{fold} {c["name"]}')
                item=dict(config=c,fold=fold,inner_validation_fold=valfold,source_overlap=0,**info);write(receipt,item)
                print('INNER',args.kind,fold,c['name'],'epoch',info['selected_epoch'],'acc',round(info['validation_metrics']['accuracy'],4),'utility',round(info['utility'],4),flush=True)
            candidates.append(item);inner_records.append(item)
        chosen=max(candidates,key=lambda r:r['utility']);c=chosen['config'];epochs=chosen['selected_epoch']
        path=out/'models'/f'fold_{fold}.npz';receipt=out/'outer'/f'fold_{fold}.json'
        if receipt.exists():item=json.loads(receipt.read_text(encoding='utf-8'));assert sha(path)==item['model_sha256']
        else:
            model,mean,std,scale,info=fit(c,outertr,2026102500+fold,epochs,tag=f'outer fold{fold} {c["name"]}')
            path.parent.mkdir(exist_ok=True);save_model_file(path,model,c,mean,std,scale,fold)
            views=allviews if args.kind=='mlp' else select_features(allviews,c['feature']).astype(np.float32)
            x=(transform(views[0,test],c,scale)-mean)/std
            item=dict(fold=fold,config=c,selected_epoch=epochs,model_sha256=sha(path),source_overlap=0)
            for mode,network in [('float',model),('int8',qmodel(model,c))]:
                prob=probabilities(network,x,c);pred=prob.argmax(1);item[mode]=metrics(y[test],pred)
                item[mode]['predictions']=[dict(filename=str(f),truth=int(t),prediction=int(p),probabilities=v.tolist()) for f,t,p,v in zip(files[test],y[test],pred,prob)]
            write(receipt,item);print('OUTER',args.kind,fold,c['name'],'float',item['float']['correct'],'int8',item['int8']['correct'],'/',len(test),flush=True)
        outer_records.append(item)
        for mode in pooled:pooled[mode]+=item[mode]['predictions']
    ranking=[dict(config=c,mean_inner_utility=float(np.mean([r['utility'] for r in inner_records if r['config']['name']==c['name']])),median_epoch=int(np.median([r['selected_epoch'] for r in inner_records if r['config']['name']==c['name']]))) for c in configs]
    selected=max(ranking,key=lambda r:r['mean_inner_utility']);c=selected['config'];epochs=selected['median_epoch']
    model,mean,std,scale,info=fit(c,np.arange(390),2026103400,epochs,tag='final '+c['name'])
    path=out/'models/final_all_esc50.npz';save_model_file(path,model,c,mean,std,scale,0)
    summary={mode:metrics(np.asarray([r['truth'] for r in rows]),np.asarray([r['prediction'] for r in rows])) for mode,rows in pooled.items()}
    assert protected=={n:sha(REPO/n) for n in protected}
    write(out/'frozen.json',dict(status='NESTED_SELECTION_AND_FINAL_FIT_FROZEN',kind=args.kind,official_nested_cv=summary,outer=outer_records,inner_ranking=ranking,
        final_config=c,final_epochs=epochs,final_model_sha256=sha(path),protocol_sha256=sha(out/'protocol.json'),external_opened=False,old_default_unchanged=True,elapsed_seconds=time.monotonic()-start))
    print('FROZEN',args.kind,json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['mlp','cnn'],required=True);p.add_argument('--audit',type=pathlib.Path,required=True)
    p.add_argument('--cnn-cache',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);run(p.parse_args())
