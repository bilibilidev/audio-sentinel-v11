"""Bounded final student experiment: wider feature student + v1 clean anchor."""
import os

import copy,datetime,hashlib,json,pathlib,time
import numpy as np
import torch
from torch.nn import functional as F
from cnn_v2 import CNNV2,CONFIGS,export,load,predict,metrics,REPO
from causal_cnn import load_model
from finetune_v1 import views,y,folds,rows,criterion
from distill_panns import expanded,pooled,target_features
REPO=pathlib.Path(__file__).resolve().parents[2]
ROOT=pathlib.Path(os.environ.get('CNN_V2_RUN_DIR',str(REPO/'out/cnn-v2-training'))).resolve()
ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/'artifacts/distill-anchor-round6';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);start=time.monotonic();base={**CONFIGS[0],'augmentation':'phone_gain_only','pooling':'mean'}
configs=[{**base,'name':'panns_anchor32','channels':32,'feature_weight':1.0,'anchor':1.0,'lr':.0003},
         {**base,'name':'panns_anchor64','channels':64,'feature_weight':1.0,'anchor':1.0,'lr':.0003},
         {**base,'name':'panns_mild64','channels':64,'feature_weight':.3,'anchor':.5,'lr':.0001}]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
protocol={'registered':datetime.datetime.now().astimezone().isoformat(),'configs':configs,'epochs':20,'extra_pretraining':'AudioSet via frozen PANNs clean ESC embeddings',
          'external_training':False,'external_opened_by_trainer':False,'selection':'Equal clean/anti-aliased-phone inner utility on ESC folds only; fixed20; three-fold v1 inner baseline, corresponding v1 four-fold outer baseline',
          'loss':'CE baby weight1.5 + feature_weight*cosine PANNs feature loss + anchor*KL temperature2 to frozen v1 on clean TRAINING clips only',
          'widening':'Function-preserving channel duplication from24, divide incoming weights by duplication counts; widths32/64',
          'augmentation':'Clean/anti-aliased8k50/50, +/-6dB gain, normalized Gaussian noise.01',
          'bound':'Final registered family for this run; preserve conservative ESC-only candidate if no simultaneous development gain',
          'external_role':'Previously seen development cohort, not blind. Extra teacher pretraining overlap not fully auditable.'}
assert not (OUT/'summary.json').exists()
if not (OUT/'protocol.json').exists():save(OUT/'protocol.json',protocol)
else:assert json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))['configs']==configs
def fit(initial,mean,std,indices,config,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=expanded(initial,config).eval();teacher=copy.deepcopy(initial).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    x=torch.from_numpy((views[:,indices]-mean)/std);labels=torch.from_numpy(y[indices]);embedding=target_features[indices];norm=torch.from_numpy(std)
    with torch.no_grad():
        delta=float((model(x[0,:3])-teacher(x[0,:3])).abs().max());old_probs=torch.cat([F.softmax(teacher(b)/2,dim=1) for b in x[0].split(32)])
    assert delta<1e-4
    projector=torch.nn.Linear(config['channels'],512);loss_fn=criterion(indices,1.5)
    opt=torch.optim.AdamW([{'params':model.parameters(),'lr':config['lr']},{'params':projector.parameters(),'lr':.001}],weight_decay=.005);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20,eta_min=.00001)
    for _ in range(20):
        model.train();order=rng.permutation(len(indices))
        for offset in range(0,len(order),32):
            ix=order[offset:offset+32];view=rng.integers(0,2,len(ix));batch=x[view,ix].clone();batch+=torch.empty((len(ix),1,1)).uniform_(-1.38155,1.38155)/norm;batch+=torch.randn_like(batch)*.01
            opt.zero_grad(set_to_none=True);latent=pooled(model,batch);logits=model.head(latent[:,:,None])[:,:,0]
            loss=loss_fn(logits,labels[ix])+config['feature_weight']*(1-F.cosine_similarity(projector(latent),embedding[ix],dim=1)).mean()
            loss+=config['anchor']*4*F.kl_div(F.log_softmax(model(x[0,ix])/2,dim=1),old_probs[ix],reduction='batchmean')
            loss.backward();torch.nn.utils.clip_grad_norm_([*model.parameters(),*projector.parameters()],5);opt.step()
        schedule.step()
    return model.eval(),delta
inner=[]
for config in configs:
    items=[]
    for fold in range(1,6):
        val=fold%5+1;train=np.flatnonzero((folds!=fold)&(folds!=val));test=np.flatnonzero(folds==val);path=OUT/'inner'/config['name']/f'fold_{fold}.npz';receipt=path.with_suffix('.json')
        if receipt.exists():item=json.loads(receipt.read_text(encoding='utf-8'));assert sha(path)==item['model_sha256']
        else:
            initial,mean,std=load(ROOT/'artifacts/warmstart-round3/inner'/f'baseline_{fold}.npz');model,delta=fit(initial,mean,std,train,config,10000+fold);export(path,model,mean,std,EXTRA_AUDIOSET_PRETRAINING=np.asarray(True))
            vals=[]
            for view in [0,1]:p=predict(model,views[view,test],mean,std);vals.append(metrics(y[test],p.argmax(1)))
            a,b=vals;item={'fold':fold,'config':config,'clean':a,'synthetic_phone':b,'utility':.4*(a['balanced_accuracy_present_classes']+b['balanced_accuracy_present_classes'])+.1*(a['per_class_recall']['baby_cry']+b['per_class_recall']['baby_cry']),
                          'model_sha256':sha(path),'initial_logits_max_delta':delta}
            save(receipt,item);print('INNER6',config['name'],fold,round(item['utility'],4),'elapsed',round(time.monotonic()-start,1),flush=True)
        items.append(item)
    inner.append({'config':config,'folds':items,'mean_utility':float(np.mean([i['utility'] for i in items]))})
chosen=[max(inner,key=lambda e:e['folds'][f-1]['utility']) for f in range(1,6)];final=max(inner,key=lambda e:e['mean_utility'])['config']
selection={'final_config':final,'final_epochs':20,'selected_without_external_audio':True,'external_used_for_model_selection':False,'extra_pretraining':'AudioSet/PANNs features',
           'config_selection_basis':'ESC clean+synthetic-phone inner only','per_outer_fold':[]}
save(OUT/'selection.json',selection);models=OUT/'selected';models.mkdir(exist_ok=True);truth=[];pooled_pred={mode:[] for mode in ['float','int8']};outer=[]
for fold,entry in enumerate(chosen,1):
    config=entry['config'];train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold);path=models/f'fold_{fold}.npz';receipt=path.with_suffix('.json')
    if receipt.exists():item=json.loads(receipt.read_text(encoding='utf-8'));assert sha(path)==item['model_sha256']
    else:
        old,mean,std=load_model(REPO/'model/cnn-v1'/f'fold_{fold}.npz');initial=CNNV2({**base,'channels':24});initial.load_state_dict(old.state_dict());model,delta=fit(initial.eval(),mean,std,train,config,11000+fold)
        export(path,model,mean,std,TEST_FOLD=np.asarray(fold),EXTRA_AUDIOSET_PRETRAINING=np.asarray(True));item={'fold':fold,'config':config,'model_sha256':sha(path),'initial_logits_max_delta':delta}
        for mode in ['float','int8']:
            model,mean,std=load(path,mode=='int8');p=predict(model,views[0,test],mean,std);pred=p.argmax(1);item[mode]=metrics(y[test],pred)
            item[mode]['predictions']=[{'filename':rows[i]['filename'],'truth':int(y[i]),'prediction':int(v),'probabilities':prob.tolist()} for i,v,prob in zip(test,pred,p)]
        save(receipt,item);print('OUTER6',fold,config['name'],item['int8']['correct'],'baby',item['int8']['per_class_recall']['baby_cry'],flush=True)
    truth.extend(y[test].tolist())
    for mode in pooled_pred:pooled_pred[mode].extend(r['prediction'] for r in item[mode]['predictions'])
    selection['per_outer_fold'].append({'fold':fold,'config':config,'model_sha256':sha(path)});outer.append(item)
selection['nested_cv']={mode:metrics(truth,p) for mode,p in pooled_pred.items()};save(OUT/'selection.json',selection)
old,mean,std=load_model(REPO/'model/cnn-v1/final_all_esc50.npz');initial=CNNV2({**base,'channels':24});initial.load_state_dict(old.state_dict());model,delta=fit(initial.eval(),mean,std,np.arange(390),final,12242)
export(models/'final_all_esc50.npz',model,mean,std,TRAINING_RECORDINGS=np.asarray(390),EXTRA_AUDIOSET_PRETRAINING=np.asarray(True))
save(OUT/'summary.json',{'protocol':protocol,'inner_candidates':inner,'outer_folds':outer,'selection':selection,'final_model_sha256':sha(models/'final_all_esc50.npz'),'elapsed_seconds':time.monotonic()-start,'external_used':False})
print('ROUND6 COMPLETE',json.dumps(selection),flush=True)
