"""One fixed independent MLP retraining run: five holdouts and all-390 fit.

Fresh PCM/features/augmentation, training-only normalization and no external
audio access before freezing. Existing model and C weights are protected.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,collections,csv,datetime,hashlib,json,pathlib,sys,time
import numpy as np
from audio_utils import read_wav_mono
from extract_features_mfcc import extract_features_mfcc_from_pcm
from build_robust_augmented_cache import augment
from train_edge_mfcc import augmented_training_data,quantize_dequantize,save_model
from sweep_edge_models import forward,metrics,recording_predictions,train,LABEL_NAMES

repo=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--esc50',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--seed-base',type=int,default=202610240)
a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
assert not (out/'training-frozen.json').exists(),'Preserve completed run'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
protected={n:sha(repo/n) for n in ['model/tiny_mlp_mfcc92_robust_simulator.npz','app/audiodetect/model_weights.h','app/audiodetect/model_weights_int8.h','simulator/generated/model_weights_int8.h','app/audiodetect/cnn_weights_v2.h']}
selection=json.loads((repo/'training/cnn/esc50-selection.json').read_text(encoding='utf-8'))['records']
protocol=dict(created=datetime.datetime.now().astimezone().isoformat(),architecture=[92,64,5],parameter_count=6277,epochs=120,batch_size=64,
     lr=.003,weight_decay=.0001,class_weight_power=.5,augmentation='hybrid PCM transforms;4 copies;75% augmented events and25% augmented background',
     seed_base=a.seed_base,augmentation_seed=202610240,normalization='Training clean+selected augmented windows only',
     holdout='Official ESC folds; all takes/src_file and all seconds of a recording stay on same side',
     model_selection=False,early_stopping=False,external_opened=False,training='Random initialization independently for every fit; no checkpoints or teacher',
     python=sys.version,numpy=np.__version__,metadata_sha256=sha(a.esc50/'meta/esc50.csv'),
     selection_sha256=sha(repo/'training/cnn/esc50-selection.json'),sources={n:sha(repo/'training'/n) for n in ['retrain_mlp_audit.py','sweep_edge_models.py','train_edge_mfcc.py','build_robust_augmented_cache.py','audio_utils.py','extract_features_mfcc.py','extract_features.py','keyword_detect.py']})
assert not (out/'protocol.json').exists(),'Use fresh directory for a fresh random fit'
save('protocol.json',protocol)
metadata={r['filename']:r for r in csv.DictReader((a.esc50/'meta/esc50.csv').open(encoding='utf-8',newline=''))}
target={'coughing':'cough','glass_breaking':'glass_break','crying_baby':'baby_cry','dog':'dog_bark'}
ordered=sorted(selection,key=lambda r:(r['label_id'],r['filename']))
pcms=[];rows=[];energy=[];sourcefolds=collections.defaultdict(set)
start=time.perf_counter()
for row in ordered:
    m=metadata[row['filename']];assert int(m['fold'])==int(row['fold'])
    assert target.get(m['category'],'background')==row['label']
    assert m['src_file']==row['src_file']
    sourcefolds[m['src_file']].add(int(row['fold']))
    raw=a.esc50/'audio'/row['filename'];assert sha(raw)==row['raw_sha256']
    pcm=read_wav_mono(raw);assert len(pcm)==80000
    for second in range(5):
        part=pcm[second*16000:(second+1)*16000].copy();pcms.append(part)
        rows.append(dict(filename=row['filename'],source=row['label']+'/'+row['filename'],src_file=row['src_file'],second=second,fold=int(row['fold']),label=row['label'],label_id=row['label_id'],pcm_sha256=hashlib.sha256(part.tobytes()).hexdigest()))
        x=part.astype(np.float32)/32768;fr=np.lib.stride_tricks.sliding_window_view(x,400)[::160]
        energy.append(float(np.mean(fr*fr)))
assert all(len(v)==1 for v in sourcefolds.values())
y=np.asarray([r['label_id'] for r in rows],dtype=np.int64);folds=np.asarray([r['fold'] for r in rows]);sources=np.asarray([r['source'] for r in rows]);src_ids=np.asarray([r['src_file'] for r in rows])
clean=[]
for i,pcm in enumerate(pcms):
    clean.append(extract_features_mfcc_from_pcm(pcm,include_delta=True))
    if (i+1)%500==0:print('CLEAN',i+1,'/1950',flush=True)
clean=np.stack(clean).astype(np.float32);assert clean.shape==(1950,92) and np.isfinite(clean).all()
assert np.max(np.abs(clean[:,0]-np.asarray(energy)))<1e-6
np.savez_compressed(out/'clean-features.npz',X=clean,Y=y,FOLDS=folds,SOURCES=sources,SRC_IDS=src_ids)
save('input-audit.json',dict(raw_hashes_verified=390,metadata_labels_verified=390,unique_source_ids=len(sourcefolds),source_ids_crossing_folds=0,recordings=390,windows=1950,
     class_recording_counts=dict(collections.Counter(r['label'] for r in ordered)),
     low_energy_windows={name:dict(quiet=int(np.sum((y==i)&(np.asarray(energy)<2e-5))),total=int(np.sum(y==i))) for i,name in enumerate(LABEL_NAMES)},
     low_energy_definition='Original MLP gate uses mean25ms frame energy<2e-5; low energy is not proof of an incorrect source label',
     window_label_note='ESC labels annotate5-second clips. Copying clip labels to all one-second cuts creates weak window labels; no event timing annotation exists here.',
     rows=rows))
background={f:np.flatnonzero((folds==f)&(y==0)) for f in range(1,6)}
augmented=[];mixes=[]
for copy in range(4):
    rng=np.random.default_rng(202610240+copy);features=[]
    for i,(pcm,label,fold) in enumerate(zip(pcms,y,folds)):
        b=int(rng.choice(background[int(fold)]));assert folds[b]==fold
        features.append(extract_features_mfcc_from_pcm(augment(pcm,int(label),pcms[b],rng,'hybrid'),include_delta=True))
        mixes.append(dict(copy=copy,source_index=i,background_index=b,fold=int(fold)))
    augmented.append(np.stack(features).astype(np.float32));print('AUGMENT',copy+1,'/4',flush=True)
augmented=np.stack(augmented);assert np.isfinite(augmented).all()
np.savez_compressed(out/'augmented-features.npz',X_AUG=augmented)
save('augmentation-audit.json',dict(same_fold_background_pairs_verified=len(mixes),pairs=mixes))
(out/'models').mkdir();receipts=[];pooled={mode:[] for mode in ('float','int8')}
for testfold in [1,2,3,4,5,0]:
    name=f'fold_{testfold}' if testfold else 'final_all_esc50';seed=a.seed_base+(testfold or 100)
    tr=np.flatnonzero(folds!=testfold) if testfold else np.arange(len(y));te=np.flatnonzero(folds==testfold) if testfold else np.asarray([],dtype=int)
    assert set(src_ids[tr]).isdisjoint(set(src_ids[te]))
    x,t=augmented_training_data(clean,y,augmented,tr,.5,.75,.25,seed+1000)
    mean=x.mean(0).astype(np.float32);std=x.std(0).astype(np.float32);std[std<1e-6]=1
    rng=np.random.default_rng(seed);initial=[]
    for fan_in,fan_out in [(92,64),(64,5)]:initial.extend([rng.normal(0,np.sqrt(2/fan_in),(fan_in,fan_out)).astype(np.float32).tobytes(),np.zeros(fan_out,dtype=np.float32).tobytes()])
    initial_sha=hashlib.sha256(b''.join(initial)).hexdigest()
    print('TRAIN RANDOM',name,'seed',seed,'train recordings',len(set(sources[tr])),'test',len(set(sources[te])),flush=True)
    weights=train((x-mean)/std,t,[64],120,64,.003,.0001,seed,.5)
    path=out/'models'/(name+'.npz');save_model(path,weights,mean,std,dict(TEST_FOLD=np.asarray(testfold),SEED=np.asarray(seed),RANDOM_INITIALIZATION=np.asarray(True),EPOCHS=np.asarray(120)))
    receipt=dict(model=name,fold=testfold,seed=seed,model_sha256=sha(path),initial_state_sha256=initial_sha,pretrained_loaded=False,train_recordings=len(set(sources[tr])),test_recordings=len(set(sources[te])),training_windows=len(x),source_overlap=0)
    for mode,w in [('float',weights),('int8',quantize_dequantize(weights))]:
        train_prob=forward((clean[tr]-mean)/std,w)[0]
        receipt[mode+'_train_window_accuracy']=float(np.mean(train_prob.argmax(1)==y[tr]))
        if testfold:
            probs=forward((clean[te]-mean)/std,w)[0];truth,pred=recording_predictions(probs,y[te],sources[te]);names=sorted(set(sources[te]))
            receipt[mode]=metrics(truth,pred)
            receipt[mode]['predictions']=[dict(source=n,truth=int(t),prediction=int(p)) for n,t,p in zip(names,truth,pred)]
            pooled[mode]+=receipt[mode]['predictions']
    receipts.append(receipt);save(name+'-receipt.json',receipt)
    print('FIT COMPLETE',name,{m:receipt.get(m,{}).get('overall') for m in pooled},flush=True)
assert protected=={n:sha(repo/n) for n in protected}
summary={mode:metrics(np.asarray([r['truth'] for r in rows]),np.asarray([r['prediction'] for r in rows])) for mode,rows in pooled.items()}
save('training-frozen.json',dict(status='SIX_RANDOM_FITS_FROZEN',models=receipts,official_cv=summary,protocol_sha256=sha(out/'protocol.json'),
     clean_features_sha256=sha(out/'clean-features.npz'),augmented_features_sha256=sha(out/'augmented-features.npz'),external_opened=False,
     protected_unchanged=True,elapsed_seconds=time.perf_counter()-start))
print('MLP TRAINING FROZEN',json.dumps(summary),flush=True)
