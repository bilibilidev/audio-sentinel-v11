"""Preregistered CNN fit on ESC-50 only; external recordings never enter training."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='4'
import argparse,datetime,hashlib,json,pathlib,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
from causal_cnn import CausalCNN,LABELS,CHANNELS,DILATIONS,logmel,read_wav_mono,export_npz,load_model,metrics
ROOT=pathlib.Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--esc50-dir',type=pathlib.Path,required=True,help='ESC-50 directory containing audio/ and meta/')
parser.add_argument('--output',type=pathlib.Path,default=ROOT/'out/cnn-training')
args=parser.parse_args()
OUT=args.output.resolve()
if OUT.exists() and any(OUT.iterdir()):parser.error('Use a new empty output directory; completed models are preserved')
OUT.mkdir(parents=True,exist_ok=True)

torch.set_num_threads(4)
EPOCHS=80;BATCH=32
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
source=ROOT
protected_names=['model/tiny_mlp_mfcc92_robust_simulator.npz','app/audiodetect/model_weights.h',
 'app/audiodetect/model_weights_int8.h','simulator/generated/model_weights_int8.h']
protected={name:sha(ROOT/name) for name in protected_names}
dataset_root=args.esc50_dir.resolve()
selection_path=pathlib.Path(__file__).with_name('esc50-selection.json')
dataset=json.loads(selection_path.read_text(encoding='utf-8'))
assert sha(dataset_root/'meta/esc50.csv')==dataset['metadata_sha256'],'ESC-50 metadata differs from the recorded selection'
records=dataset['records'];assert len(records)==390
config={'created':datetime.datetime.now().astimezone().isoformat(),'application_head':'b465d17afd43b6a574f5c368ad90ac24d102cbb7',
 'training_data':'Only the same 390 ESC-50 original recordings selected by current v11; five official folds, source separated',
 'external_training':False,'external_hyperparameter_selection':False,'epochs':EPOCHS,'batch_size':BATCH,
 'learning_rate':.002,'weight_decay':.005,'class_weight_power':.5,'label_smoothing':.05,'seed':42,
 'architecture':{'input_logmel_bands':26,'channels':CHANNELS,'kernel':5,'dilations':DILATIONS,'causal':True,
                 'frame_len_samples':400,'hop_samples':160,'receptive_field_frames':125,'parameter_count':sum(p.numel() for p in CausalCNN().parameters())},
 'augmentation':'Training only: +/-12 dB gain shift, up to 3 mel-band mask, up to 30-frame time mask on 30% of samples, normalized Gaussian noise std .02',
 'selection':'One architecture and fixed epoch count decided before any new model external result. No epoch or threshold selection from external recordings.',
 'cv':'Each of five models trains on four official folds and tests on the fifth, including source-ID overlap assertions; no external training',
 'normalization':'Per mel band mean/std of training folds only; final model uses all ESC training recordings only',
 'file_decision':'Mean causal frame logits over original recording, then softmax/argmax; Python and C must use exactly this same rule',
 'streaming':'Maintain STFT/pre-emphasis and causal convolution states across input callbacks; emit every 10 frames (100 ms); no 1 s/3 s capture reset',
 'streaming_decoder':'Prototype EMA of frame logits alpha=.02; no live threshold tuned on external set. File accuracy remains a separate mean-logit metric.',
 'torch':torch.__version__,'numpy':np.__version__,'threads':4,'dataset_manifest_sha256':sha(selection_path),
 'source_scripts_sha256':{p.name:sha(p) for p in pathlib.Path(__file__).parent.glob('*cnn*.py')},
 'protected_deployment_files':protected}
save('configuration.json',config)
start=time.monotonic();feature_path=OUT/'esc50-continuous-logmel.npz'
if feature_path.exists():
    with np.load(feature_path,allow_pickle=False) as z:x=z['X'];y=z['Y'];folds=z['FOLDS']
else:
    arrays=[]
    for i,r in enumerate(records):
        p=dataset_root/'audio'/r['filename'];assert sha(p)==r['raw_sha256']
        arrays.append(logmel(read_wav_mono(p,16000)))
        if (i+1)%50==0:print(f'CONTINUOUS FEATURES {i+1}/390',flush=True)
    x=np.stack(arrays);y=np.asarray([r['label_id'] for r in records],dtype=np.int64);folds=np.asarray([int(r['fold']) for r in records])
    np.savez_compressed(feature_path,X=x,Y=y,FOLDS=folds,FILES=np.asarray([r['filename'] for r in records]))
assert x.shape==(390,26,498) and np.isfinite(x).all()
def fit(indices,seed,label):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    values=x[indices];targets=y[indices]
    mean=values.mean(axis=(0,2),dtype=np.float64).astype(np.float32).reshape(1,26,1)
    std=values.std(axis=(0,2),dtype=np.float64).astype(np.float32).reshape(1,26,1);std[std<1e-4]=1
    tx=torch.from_numpy((values-mean)/std);ty=torch.from_numpy(targets)
    counts=np.bincount(targets,minlength=5);weights=1/np.maximum(counts,1)**.5;weights/=weights.mean()
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32),label_smoothing=.05)
    model=CausalCNN();optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.005)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=.00005)
    band_std=torch.from_numpy(std)
    for epoch in range(EPOCHS):
        model.train();order=rng.permutation(len(indices));loss_sum=0
        for a in range(0,len(order),BATCH):
            subset=order[a:a+BATCH];batch=tx[subset].clone()
            gain=torch.empty((len(subset),1,1)).uniform_(-2.7631,2.7631)
            batch+=gain/band_std
            batch+=torch.randn_like(batch)*.02
            for sample in batch:
                width=int(rng.integers(0,4))
                if width:
                    lo=int(rng.integers(0,27-width));sample[lo:lo+width]=0
                if rng.random()<.3:
                    width=int(rng.integers(1,31));lo=int(rng.integers(0,499-width));sample[:,lo:lo+width]=0
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(batch),ty[subset]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step();loss_sum+=float(loss.detach())*len(subset)
        scheduler.step()
        if (epoch+1)%10==0:
            print(f'{label} epoch {epoch+1}/{EPOCHS} loss={loss_sum/len(indices):.5f} elapsed={time.monotonic()-start:.1f}s',flush=True)
            save('progress.json',{'stage':label,'epoch':epoch+1,'epochs':EPOCHS,'elapsed_seconds':time.monotonic()-start})
    return model.eval(),mean,std
all_truth=[];all_pred={m:[] for m in ['float','int8']};fold_results=[]
for fold in range(1,6):
    train_indices=np.where(folds!=fold)[0];test_indices=np.where(folds==fold)[0]
    assert {records[i]['src_file'] for i in train_indices}.isdisjoint({records[i]['src_file'] for i in test_indices})
    path=OUT/f'fold_{fold}.npz'
    if not path.exists():
        model,mean,std=fit(train_indices,42+fold,f'FOLD {fold}')
        export_npz(path,model,mean,std,TEST_FOLD=np.asarray(fold))
    result={'fold':fold,'model_sha256':sha(path),'training_recordings':len(train_indices),'test_recordings':len(test_indices),'source_overlap':0}
    for mode in ['float','int8']:
        model,mean,std=load_model(path,quantized=mode=='int8')
        with torch.no_grad():probs=torch.softmax(model(torch.from_numpy((x[test_indices]-mean)/std)),dim=1).numpy()
        pred=probs.argmax(axis=1);result[mode]=metrics(y[test_indices],pred)
        result[mode]['predictions']=[{'filename':records[i]['filename'],'reference_label':LABELS[int(y[i])],
            'prediction':LABELS[int(p)],'probabilities':v.tolist()} for i,p,v in zip(test_indices,pred,probs)]
        all_pred[mode].extend(pred.tolist())
    all_truth.extend(y[test_indices].tolist());fold_results.append(result);save(f'fold-{fold}-results.json',result)
    print(f'HELD OUT {fold}: float={result["float"]["accuracy"]:.5f} int8={result["int8"]["accuracy"]:.5f}',flush=True)
final_path=OUT/'final_all_esc50.npz'
if not final_path.exists():
    model,mean,std=fit(np.arange(len(x)),4242,'FINAL ALL ESC50')
    export_npz(final_path,model,mean,std,TRAINING_RECORDINGS=np.asarray(390))
summary={'configuration':config,'folds':fold_results,'official_held_out_cv':{m:metrics(all_truth,all_pred[m]) for m in all_pred},
         'final_model_sha256':sha(final_path),'final_model_training_recordings':390,'elapsed_seconds':time.monotonic()-start,
         'external_training':False,'deployment_unchanged':all(sha(ROOT/p)==h for p,h in protected.items())}
assert summary['deployment_unchanged'];save('summary.json',summary)
print('CNN TRAINING COMPLETE',json.dumps(summary['official_held_out_cv']),flush=True)
