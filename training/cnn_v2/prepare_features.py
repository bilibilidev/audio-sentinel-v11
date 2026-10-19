"""Stateless ESC-only waveform views; external validation cannot enter training."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='4'
import hashlib,json,pathlib,time
import numpy as np
from cnn_v2 import all_features,read_wav_mono,resample_linear,REPO

REPO=pathlib.Path(__file__).resolve().parents[2]
ROOT=pathlib.Path(os.environ.get('CNN_V2_RUN_DIR',str(REPO/'out/cnn-v2-training'))).resolve()
ROOT.mkdir(parents=True,exist_ok=True)
(ROOT/'artifacts').mkdir(exist_ok=True)
out=ROOT/'artifacts/esc50-features.npz'
assert not out.exists(),'Preserve feature cache'
work=ROOT.parent
selection=json.loads((REPO/'training/cnn/esc50-selection.json').read_text(encoding='utf-8'))
source=pathlib.Path(os.environ['ESC50_DATASET']);rows=selection['records'];rng=np.random.default_rng(20261019)
views=[];start=time.monotonic()
for index,r in enumerate(rows):
    path=source/'audio'/r['filename'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['raw_sha256']
    pcm=read_wav_mono(path,16000)
    low=resample_linear(resample_linear(pcm,16000,8000),8000,16000)
    clip=np.clip(pcm.astype(np.float32)*10**(float(rng.uniform(6,18))/20),-32768,32767).astype(np.int16)
    clip=resample_linear(resample_linear(clip,16000,8000),8000,16000)
    rms=max(float(np.sqrt(np.mean(low.astype(np.float64)**2))),1)
    noisy=np.clip(low.astype(np.float32)+rng.normal(0,rms/10**(float(rng.uniform(15,30))/20),len(low)),-32768,32767).astype(np.int16)
    view=np.stack([all_features(pcm),all_features(low),all_features(clip),all_features(noisy)])
    assert view.shape==(4,31,498) and np.isfinite(view).all();views.append(view)
    if (index+1)%50==0:print('ESC FEATURES',index+1,'/390',flush=True)
x=np.stack(views,axis=1)
np.savez_compressed(out,X=x,Y=np.asarray([r['label_id'] for r in rows]),FOLDS=np.asarray([int(r['fold']) for r in rows]),FILES=np.asarray([r['filename'] for r in rows]),SOURCES=np.asarray([r['src_file'] for r in rows]))
print('Prepared',x.shape,'elapsed',time.monotonic()-start,flush=True)
