"""Small deployable CNN with frozen AudioSet/PANNs feature supervision.

Only the first390 ESC embeddings enter fitting; the external65 rows from the
baseline cache are explicitly excluded. Extra AudioSet prior is disclosed.
"""
import os

import datetime,hashlib,json,pathlib,time
import numpy as np
import torch
from torch.nn import functional as F
from cnn_v2 import CNNV2,CONFIGS,export,load,predict,metrics,REPO
from causal_cnn import load_model
from finetune_v1 import views,y,folds,rows,criterion
REPO=pathlib.Path(__file__).resolve().parents[2]
ROOT=pathlib.Path(os.environ.get('CNN_V2_RUN_DIR',str(REPO/'out/cnn-v2-training'))).resolve()
ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/'artifacts/distillation-round5';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);start=time.monotonic()
base={**CONFIGS[0],'augmentation':'phone_gain_only','pooling':'mean'}
configs=[{**base,'name':f'panns_feature_c{channels}_w{weight}','channels':channels,'feature_weight':weight,'epochs':40,'lr':.0003} for channels in [24,32] for weight in [.3,1.0]]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
teacher_path=OUT/'esc-only-teacher-features.npz'
if not teacher_path.exists():
    with np.load(ROOT/'artifacts/baselines/panns-embeddings.npz',allow_pickle=False) as z:
        assert z['X'].shape==(455,512);embedding=z['X'][:390].copy()
    np.savez_compressed(teacher_path,X=embedding,FILES=np.asarray([r['filename'] for r in rows]))
with np.load(teacher_path,allow_pickle=False) as z:
    embedding=z['X'];assert z['FILES'].tolist()==[r['filename'] for r in rows]
assert embedding.shape==(390,512)
target_features=torch.from_numpy(embedding/np.maximum(np.linalg.norm(embedding,axis=1,keepdims=True),1e-8))
protocol={'registered':datetime.datetime.now().astimezone().isoformat(),'configs':configs,'external_training':False,'external_opened_by_trainer':False,
          'extra_pretraining':'Frozen official PANNs Cnn6 AudioSet512D embeddings used as training-only auxiliary targets; not an equal-pretraining-budget comparison with ESC-only methods',
          'teacher_features_sha256':sha(teacher_path),'teacher_checkpoint_sha256':'8a17541944f2623a2a0164e072ac2f915dcedca99968083cc44bfbc7eb682275',
          'teacher_cache_rows_used':'Exactly0:390 ESC records, source ordered by fixed selection; external390:455 excluded',
          'selection':'Clean/anti-aliased-phone ESC inner utility; v1 three-fold inner initializers, corresponding frozen v1 four-fold outer initializers. Final by mean inner utility.',
          'loss':'Class-weighted CE label smoothing.05 baby boost1.5 + feature_weight * (1-cosine(projected CNN mean feature, normalized frozen clean PANNs embedding))',
          'widening':'24->32 duplicates first8 channels and divides next-layer incoming weights by duplication counts; validate initial floating logits unchanged',
          'optimizer':'AdamW CNNlr.0003, auxiliary projectorlr.001, wd.005, cosine40, batch32, dropout.15; projector removed for inference',
          'augmentation':'Clean/anti-aliased8k50/50, +/-6dB gain, normalized Gaussian noise.01; frozen clean semantic target applies to both views',
          'external_role':'Already seen development-validation, not blind; AudioSet pretraining overlap not fully auditable'}
if __name__=='__main__':assert not (OUT/'summary.json').exists()
if not (OUT/'protocol.json').exists():save(OUT/'protocol.json',protocol)
else:assert json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))['configs']==configs
def expanded(initial,config):
    model=CNNV2(config)
    if config['channels']==24:model.load_state_dict(initial.state_dict());return model
    mapping=torch.tensor([i%24 for i in range(config['channels'])],dtype=torch.long);counts=torch.bincount(mapping,minlength=24).float()[mapping]
    with torch.no_grad():
        for i,(old,new) in enumerate(zip(initial.convs,model.convs)):
            weight=old.weight[mapping].clone()
            if i:weight=weight[:,mapping,:]/counts[None,:,None]
            new.weight.copy_(weight);new.bias.copy_(old.bias[mapping])
        model.head.weight.copy_(initial.head.weight[:,mapping,:]/counts[None,:,None]);model.head.bias.copy_(initial.head.bias)
    return model
def pooled(model,x):
    for layer,d in zip(model.convs,model.config['dilations']):x=model.dropout(F.relu(layer(F.pad(x,(4*d,0)))))
    return x.mean(dim=2)
def fit(initial,mean,std,indices,config,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=expanded(initial,config).eval();x=torch.from_numpy((views[:,indices]-mean)/std);target=torch.from_numpy(y[indices]);teacher=target_features[indices]
    with torch.no_grad():init_delta=float((model(x[0,:3])-initial(x[0,:3])).abs().max())
    assert init_delta<1e-4,init_delta
    projector=torch.nn.Linear(config['channels'],512);loss_fn=criterion(indices,1.5);norm=torch.from_numpy(std)
    opt=torch.optim.AdamW([{'params':model.parameters(),'lr':config['lr']},{'params':projector.parameters(),'lr':.001}],weight_decay=.005)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40,eta_min=.00003)
    for _ in range(40):
        model.train();projector.train();order=rng.permutation(len(indices))
        for offset in range(0,len(order),32):
            ix=order[offset:offset+32];view=rng.integers(0,2,len(ix));batch=x[view,ix].clone();batch+=torch.empty((len(ix),1,1)).uniform_(-1.38155,1.38155)/norm;batch+=torch.randn_like(batch)*.01
            opt.zero_grad(set_to_none=True);latent=pooled(model,batch);logits=model.head(latent[:,:,None])[:,:,0]
            feature_loss=(1-F.cosine_similarity(projector(latent),teacher[ix],dim=1)).mean();loss=loss_fn(logits,target[ix])+config['feature_weight']*feature_loss
            loss.backward();torch.nn.utils.clip_grad_norm_([*model.parameters(),*projector.parameters()],5);opt.step()
        schedule.step()
    return model.eval(),init_delta
if __name__=='__main__':
    inner=[]
    for config in configs:
        items=[]
        for fold in range(1,6):
            val=fold%5+1;train=np.flatnonzero((folds!=fold)&(folds!=val));test=np.flatnonzero(folds==val);receipt=OUT/'inner'/config['name']/f'fold_{fold}.json';path=receipt.with_suffix('.npz')
            if receipt.exists():item=json.loads(receipt.read_text(encoding='utf-8'));assert sha(path)==item['model_sha256']
            else:
                initial,mean,std=load(ROOT/'artifacts/warmstart-round3/inner'/f'baseline_{fold}.npz')
                assert np.array_equal(mean,views[0,train].mean((0,2),dtype=np.float64).astype(np.float32).reshape(1,26,1))
                model,delta=fit(initial,mean,std,train,config,7000+fold);export(path,model,mean,std,EXTRA_AUDIOSET_PRETRAINING=np.asarray(True))
                vals=[]
                for view in [0,1]:p=predict(model,views[view,test],mean,std);vals.append(metrics(y[test],p.argmax(1)))
                a,b=vals;item={'fold':fold,'config':config,'clean':a,'synthetic_phone':b,'utility':.4*(a['balanced_accuracy_present_classes']+b['balanced_accuracy_present_classes'])+.1*(a['per_class_recall']['baby_cry']+b['per_class_recall']['baby_cry']),
                    'model_sha256':sha(path),'initial_logits_preserved_max_delta':delta}
                save(receipt,item);print('INNER5',config['name'],fold,round(item['utility'],4),'elapsed',round(time.monotonic()-start,1),flush=True)
            items.append(item)
        inner.append({'config':config,'folds':items,'mean_utility':float(np.mean([i['utility'] for i in items]))});save(OUT/'progress.json',{'completed_configs':len(inner),'total_configs':len(configs)})
    chosen=[max(inner,key=lambda e:e['folds'][f-1]['utility']) for f in range(1,6)];final=max(inner,key=lambda e:e['mean_utility'])['config']
    selection={'final_config':final,'final_epochs':40,'selected_without_external_audio':True,'external_used_for_model_selection':False,'extra_pretraining':'AudioSet via frozen PANNs feature supervision',
              'config_selection_basis':'ESC clean+synthetic-phone inner utility only; extra AudioSet frozen teacher knowledge','per_outer_fold':[]}
    save(OUT/'selection.json',selection);models=OUT/'selected';models.mkdir(exist_ok=True);truth=[];pooled_pred={mode:[] for mode in ['float','int8']};results=[]
    for fold,entry in enumerate(chosen,1):
        config=entry['config'];train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold);path=models/f'fold_{fold}.npz';receipt=path.with_suffix('.json')
        if receipt.exists():item=json.loads(receipt.read_text(encoding='utf-8'));assert sha(path)==item['model_sha256']
        else:
            old,mean,std=load_model(REPO/'model/cnn-v1'/f'fold_{fold}.npz');initial=CNNV2({**base,'channels':24});initial.load_state_dict(old.state_dict())
            model,delta=fit(initial.eval(),mean,std,train,config,8000+fold);export(path,model,mean,std,TEST_FOLD=np.asarray(fold),EXTRA_AUDIOSET_PRETRAINING=np.asarray(True));item={'fold':fold,'config':config,'model_sha256':sha(path),'initial_logits_preserved_max_delta':delta}
            for mode in ['float','int8']:
                model,mean,std=load(path,mode=='int8');p=predict(model,views[0,test],mean,std);pred=p.argmax(1);item[mode]=metrics(y[test],pred)
                item[mode]['predictions']=[{'filename':rows[i]['filename'],'truth':int(y[i]),'prediction':int(v),'probabilities':prob.tolist()} for i,v,prob in zip(test,pred,p)]
            save(receipt,item);print('OUTER5',fold,config['name'],item['int8']['correct'],'baby',item['int8']['per_class_recall']['baby_cry'],flush=True)
        truth.extend(y[test].tolist())
        for mode in pooled_pred:pooled_pred[mode].extend(r['prediction'] for r in item[mode]['predictions'])
        selection['per_outer_fold'].append({'fold':fold,'config':config,'model_sha256':sha(path)});results.append(item)
    selection['nested_cv']={mode:metrics(truth,p) for mode,p in pooled_pred.items()};save(OUT/'selection.json',selection)
    old,mean,std=load_model(REPO/'model/cnn-v1/final_all_esc50.npz');initial=CNNV2({**base,'channels':24});initial.load_state_dict(old.state_dict())
    model,delta=fit(initial.eval(),mean,std,np.arange(390),final,9242);export(models/'final_all_esc50.npz',model,mean,std,TRAINING_RECORDINGS=np.asarray(390),EXTRA_AUDIOSET_PRETRAINING=np.asarray(True))
    save(OUT/'summary.json',{'protocol':protocol,'inner_candidates':inner,'outer_folds':results,'selection':selection,'final_model_sha256':sha(models/'final_all_esc50.npz'),'elapsed_seconds':time.monotonic()-start,'external_used':False})
    print('ROUND5 COMPLETE',json.dumps(selection),flush=True)
