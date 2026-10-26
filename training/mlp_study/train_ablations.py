"""Fixed-recipe controlled ablations of the formal Tiny MLP (fresh fits, no selection)."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,copy,hashlib,json,pathlib,sys,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'training'),str(REPO/'training/optimization')]
from search import normalize_fit,transform,probabilities,qmodel
from train_edge_mfcc import save_model
from export_to_c import int8_header,c_float_array
LABELS=['background','cough','glass_break','baby_cry','dog_bark']
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
assert not (out/'frozen.json').exists()
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
base=dict(kind='mlp',channels=96,loss='window',augmentation='mild',dropout=.15,lr=.002,wd=.001,smoothing=.02,epochs=240,ema=0.,transform='asinh40',features='all92',class_weight=True)
configs=[{**base,**c} for c in [dict(name='full'),dict(name='time_only',features='time40'),dict(name='mfcc_only',features='mfcc52',transform='none'),
    dict(name='no_delta',features='no_delta66'),dict(name='no_asinh',transform='none'),dict(name='no_augmentation',augmentation='none'),
    dict(name='hidden64',channels=64),dict(name='hidden128',channels=128),dict(name='no_dropout',dropout=0.),dict(name='no_class_weight',class_weight=False)]]
audit=a.audit.resolve()/'mlp-training';rows=read(audit/'input-audit.json')['rows'][::5]
files=np.array([r['filename'] for r in rows]);sources=np.array([r['src_file'] for r in rows]);y=np.array([r['label_id'] for r in rows]);folds=np.array([r['fold'] for r in rows])
with np.load(audit/'clean-features.npz',allow_pickle=False) as z:clean=z['X'].reshape(390,5,92).copy();assert np.array_equal(z['Y'].reshape(390,5)[:,0],y)
with np.load(audit/'augmented-features.npz',allow_pickle=False) as z:aug=z['X_AUG'].reshape(4,390,5,92).copy()
views=np.concatenate([clean[None],aug]);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
protected={n:sha(REPO/n) for n in ['model/mlp-optimized-20261024/final_all_esc50.npz','app/audiodetect/mlp_weights.h','app/audiodetect/cnn_weights_v2.h']}
protocol=dict(date='2026-10-26',source_sha256=sha(pathlib.Path(__file__)),configs=configs,epochs=60,scheduler_horizon=240,
    policy='Fixed formal-recipe60epochs for EVERY variant/fold, no epoch/config selection, same initialization seed and data order. One changed factor per variant.',
    train='390 existing ESC records only, official folds/src_file grouping; all5 seconds and augmentation sources restricted to training folds.',
    external_audio_read_during_training=False,external_role='Existing65 development recordings already used in prior project selection, not a blind test.',
    seeds=dict(outer_base=2026102500,final=2026103400),pretrained=False,finetuned=False,teacher=False,
    distinction='Fixed-recipe CV is NEW, not previous333/390 nested-selected CV with varying widths/data budgets.',
    environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,threads=1),
    input_hashes={n:sha(audit/n) for n in ['clean-features.npz','augmented-features.npz','input-audit.json']})
save(out/'protocol.json',protocol);start=time.monotonic();receipts=[]
class Model(nn.Module):
    def __init__(self,c,n):
        super().__init__();self.fc1=nn.Linear(n,c['channels']);self.drop=nn.Dropout(c['dropout']);self.fc2=nn.Linear(c['channels'],5)
    def frames(self,x):return self.fc2(self.drop(F.relu(self.fc1(x))))
    def forward(self,x):return torch.softmax(self.frames(x),dim=-1).mean(1)
for c in configs:
    feature_indices=np.arange(92) if c['features']=='all92' else (np.arange(40) if c['features']=='time40' else np.arange(40,92) if c['features']=='mfcc52' else np.arange(66))
    for fold in [1,2,3,4,5,0]:
        name='final_all_esc50' if not fold else f'fold_{fold}';folder=out/c['name']/name;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'receipt.json').exists():receipts.append(read(folder/'receipt.json'));continue
        tr=np.flatnonzero(folds!=fold) if fold else np.arange(390);test=np.flatnonzero(folds==fold) if fold else np.array([],dtype=int)
        assert not set(sources[tr])&set(sources[test])
        mapped,mean,std,scale=normalize_fit(views,tr,c);mapped=mapped[...,feature_indices];mean=mean[feature_indices];std=std[feature_indices]
        x=torch.from_numpy((mapped[:,tr]-mean)/std);target=torch.from_numpy(y[tr])
        seed=2026102500+fold if fold else 2026103400;torch.manual_seed(seed);rng=np.random.default_rng(seed);model=Model(c,len(feature_indices))
        counts=np.bincount(y[tr],minlength=5);weights=1/np.sqrt(np.maximum(counts,1));weights/=weights.mean();classw=torch.tensor(weights,dtype=torch.float32) if c['class_weight'] else None
        opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['wd']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=240,eta_min=c['lr']*.025)
        history=[]
        for epoch in range(1,61):
            model.train();order=rng.permutation(len(tr));loss_sum=0.
            for off in range(0,len(order),32):
                subset=order[off:off+32];view=np.zeros(len(subset),dtype=int)
                if c['augmentation']!='none':
                    use=rng.random(len(subset))<np.where(y[tr[subset]]==0,.1,.3);view[use]=rng.integers(1,5,int(use.sum()))
                batch=x[view,subset];labels=target[subset]
                opt.zero_grad(set_to_none=True);logits=model.frames(batch)
                loss=F.cross_entropy(logits.reshape(-1,5),labels[:,None].expand(-1,5).reshape(-1),weight=classw,label_smoothing=c['smoothing'])
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();loss_sum+=float(loss.detach())*len(subset)
            sched.step()
            if epoch%10==0:
                history.append(dict(epoch=epoch,training_loss=loss_sum/len(tr)));save(out/'progress.json',dict(variant=c['name'],fold=fold,epoch=epoch,elapsed_seconds=time.monotonic()-start))
        model.eval();weights=[dict(W=model.fc1.weight.detach().numpy().T,B=model.fc1.bias.detach().numpy()),dict(W=model.fc2.weight.detach().numpy().T,B=model.fc2.bias.detach().numpy())]
        npz=folder/'model.npz';save_model(npz,weights,mean,std,dict(CONFIG=np.array(json.dumps(c,sort_keys=True)),TIME_ASINH_SCALE=scale,FEATURE_INDICES=feature_indices,TEST_FOLD=np.array(fold),EPOCHS=np.array(60),RANDOM_INITIALIZATION=np.array(True)))
        with np.load(npz,allow_pickle=False) as z:header=int8_header(z,LABELS,'STUDY_WEIGHTS_H',True)
        extra='#define STUDY_ASINH '+str(int(c['transform']=='asinh40'))+'\n'+c_float_array('TIME_ASINH_SCALE',scale,'[40]')
        extra+='static const int FEATURE_INDICES[] = {'+','.join(map(str,feature_indices))+'};\n'
        header=header.replace('#include <stdint.h>','#include <stdint.h>\n'+extra);(folder/'weights.h').write_text(header,encoding='utf-8',newline='\n')
        receipt=dict(variant=c['name'],model=name,config=c,seed=seed,training_recordings=len(tr),training_source_ids=len(set(sources[tr])),source_overlap=0,parameters=sum(p.numel() for p in model.parameters()),feature_dim=len(feature_indices),model_sha256=sha(npz),header_sha256=sha(folder/'weights.h'),history=history)
        if len(test):
            features=(mapped[0,test]-mean)/std
            for mode,m in [('float',model),('int8',qmodel(model,c))]:
                prob=probabilities(m,features,c);receipt[mode]=[dict(filename=str(f),truth=int(t),prediction=int(p),probabilities=v.tolist()) for f,t,p,v in zip(files[test],y[test],prob.argmax(1),prob)]
        if not fold and c['name']=='full':
            with np.load(REPO/'model/mlp-optimized-20261024/final_all_esc50.npz',allow_pickle=False) as official,np.load(npz,allow_pickle=False) as fresh:
                receipt['official_parameter_bit_identical']=all(np.array_equal(official[n],fresh[n]) for n in ('W1','B1','W2','B2','FEATURE_MEAN','FEATURE_STD','TIME_ASINH_SCALE'))
                assert receipt['official_parameter_bit_identical'],'Full recipe does not reproduce official model parameters'
        save(folder/'receipt.json',receipt);receipts.append(receipt);print('FIT',c['name'],name,{k:sum(r['truth']==r['prediction'] for r in receipt.get(k,[])) for k in ['float','int8']},flush=True)
assert protected=={n:sha(REPO/n) for n in protected}
save(out/'frozen.json',dict(status='ALL_60_FRESH_FITS_FROZEN',protocol_sha256=sha(out/'protocol.json'),models=receipts,production_model_unchanged=True,external_opened=False,elapsed_seconds=time.monotonic()-start));print('ABLATIONS FROZEN',flush=True)
