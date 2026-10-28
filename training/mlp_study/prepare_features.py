"""Prepare the fixed390-recording Tiny MLP clean and augmented feature cache."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,collections,csv,datetime,hashlib,json,pathlib,sys,time
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from audio_utils import read_wav_mono
from extract_features_mfcc import extract_features_mfcc_from_pcm
from augmentation import augment
from numerics import LABEL_NAMES

repo=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--esc50',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--seed-base',type=int,default=202610240)
a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
assert not (out/'training-frozen.json').exists(),'Preserve completed run'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
protected={n:sha(repo/n) for n in ['model/mlp-optimized-20261024/final_all_esc50.npz','app/audiodetect/mlp_weights.h']}
selection=json.loads((repo/'training/mlp_study/esc50-selection.json').read_text(encoding='utf-8'))['records']
protocol=dict(created=datetime.datetime.now().astimezone().isoformat(),recordings=390,feature_dim=92,augmentation_seed=202610240,selection_sha256=sha(repo/'training/mlp_study/esc50-selection.json'),python=sys.version,numpy=np.__version__)
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
assert protected=={n:sha(repo/n) for n in protected}
print("FEATURE CACHE READY",str(out),flush=True)
