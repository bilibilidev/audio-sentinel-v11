"""Fixed-recipe supervised training from random initialization, with no v1 input.

Five official ESC-50 holdouts plus a full-390 deployment fit. The external set
is not opened until every fit and its selection-free protocol have been frozen.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import argparse,datetime,hashlib,json,pathlib,subprocess,sys,time
import numpy as np
import torch
from cnn_v2 import CNNV2,export,load,predict,metrics,LABELS

REPO=pathlib.Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--features',type=pathlib.Path,required=True)
parser.add_argument('--output',type=pathlib.Path,required=True)
parser.add_argument('--seed-base',type=int,default=20261024)
args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
config=dict(name='scratch_logmel24_mean',feature='logmel',channels=24,dilations=[1,2,4,8,16],augmentation='legacy',pooling='mean',epochs=80,initialization='random')
selection_path=REPO/'training/cnn/esc50-selection.json'
protocol=dict(created=datetime.datetime.now().astimezone().isoformat(),name='CNN v2 from-scratch fixed recipe',config=config,
    initialization='Fresh torch random initialization for EACH fold and final fit; no v1 checkpoint, teacher, optimizer state or pretrained weights',
    epochs=80,batch_size=32,optimizer='AdamW',lr=.002,weight_decay=.005,minimum_lr=.00005,
    scheduler='CosineAnnealingLR T_max=80',label_smoothing=.05,dropout=.15,class_weight_power=.5,baby_weight_multiplier=1.5,
    augmentation='ESC-only clean logmel plus +/-12dB gain, normalized Gaussian noise std .02, feature masks width0..3, time masks probability .3 and width1..30; no phone view',
    normalization='Recomputed on clean TRAINING recordings only for each fit',
    official_folds='Fit each model on four official folds; score the excluded fold only after fixed80 epochs; pooled recording accuracy, not unweighted fold mean',
    epoch_or_config_search=False,early_stopping=False,external_gradient_training=False,external_used_for_selection_in_this_run=False,
    external_role='Previously reviewed development set from earlier work; not a new blind test, even though this run does not select on it',
    seeds={**{f'fold_{i}':args.seed_base+i for i in range(1,6)},'final_all_esc50':args.seed_base+100},
    features_sha256=sha(args.features),selection_sha256=sha(selection_path),script_sha256=sha(pathlib.Path(__file__)),
    network_source_sha256=sha(REPO/'training/cnn_v2/cnn_v2.py'),torch=torch.__version__,numpy=np.__version__,python=sys.version,
    torch_threads=4,interop_threads=1,application_base_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip())
protocol_path=out/'protocol.json'
if protocol_path.exists():
    prior=json.loads(protocol_path.read_text(encoding='utf-8'))
    for key in protocol:
        if key!='created':assert prior[key]==protocol[key],key
    protocol=prior
else:save(protocol_path,protocol)
assert not (out/'training-frozen.json').exists(),'Completed training is immutable; use its model files for evaluation'
selection=json.loads(selection_path.read_text(encoding='utf-8'))['records']
with np.load(args.features,allow_pickle=False) as z:
    assert z['X'].shape==(4,390,31,498)
    raw=z['X'][0,:,:26,:].copy()
    y=z['Y'].copy();folds=z['FOLDS'].copy();files=z['FILES'].copy();sources=z['SOURCES'].copy()
assert np.isfinite(raw).all() and len(selection)==390
assert np.array_equal(y,np.asarray([r['label_id'] for r in selection]))
assert files.tolist()==[r['filename'] for r in selection]
assert np.array_equal(folds,np.asarray([int(r['fold']) for r in selection]))
assert sources.tolist()==[r['src_file'] for r in selection]
assert set(folds.tolist())=={1,2,3,4,5}
start=time.monotonic();receipts=[]
for name,seed in protocol['seeds'].items():
    modelpath=out/'models'/(name+'.npz');receiptpath=out/'receipts'/(name+'.json')
    if receiptpath.exists():
        receipt=json.loads(receiptpath.read_text(encoding='utf-8'));assert sha(modelpath)==receipt['model_sha256']
        receipts.append(receipt);print('RESUME COMPLETED',name,flush=True);continue
    testfold=int(name[-1]) if name.startswith('fold_') else None
    train=np.flatnonzero(folds!=testfold) if testfold else np.arange(390)
    test=np.flatnonzero(folds==testfold) if testfold else np.asarray([],dtype=int)
    assert set(sources[train]).isdisjoint(set(sources[test]))
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    mean=raw[train].mean((0,2),dtype=np.float64).astype(np.float32).reshape(1,26,1)
    std=raw[train].std((0,2),dtype=np.float64).astype(np.float32).reshape(1,26,1);std[std<1e-4]=1
    x=torch.from_numpy((raw[train]-mean)/std);target=torch.from_numpy(y[train]);norm=torch.from_numpy(std)
    counts=np.bincount(y[train],minlength=5);weights=1/np.sqrt(np.maximum(counts,1));weights[3]*=1.5;weights/=weights.mean()
    model=CNNV2(config)
    initial=hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32),label_smoothing=.05)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.005)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=80,eta_min=.00005)
    fitstart=time.monotonic();history=[]
    print('START RANDOM FIT',name,'train',len(train),'test',len(test),'seed',seed,'initial',initial,flush=True)
    for epoch in range(1,81):
        model.train();order=rng.permutation(len(train));total_loss=0.
        for offset in range(0,len(order),32):
            indices=order[offset:offset+32];batch=x[indices].clone()
            batch+=torch.empty((len(indices),1,1)).uniform_(-2.7631,2.7631)/norm
            batch+=torch.randn_like(batch)*.02
            for sample in batch:
                width=int(rng.integers(0,4))
                if width:
                    lo=int(rng.integers(0,27-width));sample[lo:lo+width]=0
                if rng.random()<.3:
                    width=int(rng.integers(1,31));lo=int(rng.integers(0,499-width));sample[:,lo:lo+width]=0
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(batch),target[indices]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            total_loss+=float(loss.detach())*len(indices)
        scheduler.step()
        history.append(dict(epoch=epoch,training_loss=total_loss/len(train),learning_rate=optimizer.param_groups[0]['lr']))
        save(out/'progress.json',dict(phase='training',model=name,epoch=epoch,epochs=80,completed_fits=len(receipts),total_fits=6,fit_seconds=time.monotonic()-fitstart,total_seconds=time.monotonic()-start))
        if epoch==1 or epoch%10==0:print(name,'epoch',epoch,'loss',round(history[-1]['training_loss'],5),'seconds',round(time.monotonic()-fitstart,1),flush=True)
    model.eval()
    export(modelpath,model,mean,std,TRAINING_RECORDINGS=np.asarray(len(train)),EPOCHS=np.asarray(80),RANDOM_INITIALIZATION=np.asarray(True),SEED=np.asarray(seed),TEST_FOLD=np.asarray(testfold or 0))
    receipt=dict(model=name,seed=seed,model_sha256=sha(modelpath),random_initial_state_sha256=initial,pretrained_weights_loaded=False,teacher_used=False,train_recordings=len(train),test_recordings=len(test),source_overlap=0,epochs=80,parameter_count=sum(t.numel() for t in model.parameters()),training_seconds=time.monotonic()-fitstart,history=history)
    assert receipt['parameter_count']==14885
    if len(test):
        for mode in ('float','int8'):
            network,m,s=load(modelpath,quantized=(mode=='int8'));prob=predict(network,raw[test],m,s);pred=prob.argmax(1)
            receipt[mode]=metrics(y[test],pred)
            receipt[mode]['predictions']=[dict(filename=str(files[i]),truth=int(y[i]),prediction=int(pr),probabilities=v.tolist()) for i,pr,v in zip(test,pred,prob)]
        print('FROZEN HOLDOUT',name,receipt['float']['correct'],receipt['int8']['correct'],'/',len(test),flush=True)
    save(receiptpath,receipt);receipts.append(receipt)
pooled={}
for mode in ('float','int8'):
    rows=[r for item in receipts[:5] for r in item[mode]['predictions']]
    assert len(rows)==390 and len({r['filename'] for r in rows})==390
    pooled[mode]=metrics(np.asarray([r['truth'] for r in rows]),np.asarray([r['prediction'] for r in rows]))
save(out/'training-frozen.json',dict(status='SIX_RANDOM_INITIALIZATION_FITS_COMPLETE',protocol_sha256=sha(protocol_path),config=config,models=[{k:r[k] for k in ('model','model_sha256','random_initial_state_sha256','seed','train_recordings','test_recordings','epochs','pretrained_weights_loaded','teacher_used')} for r in receipts],official_cv=pooled,folds=receipts[:5],final_model_sha256=receipts[-1]['model_sha256'],elapsed_seconds=time.monotonic()-start,external_opened=False))
print('TRAINING FROZEN',json.dumps(pooled),flush=True)
