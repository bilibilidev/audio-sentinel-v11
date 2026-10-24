"""Replace aliased synthetic phone views while preserving canonical clean inputs."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import argparse,hashlib,json,pathlib,sys
import numpy as np
from scipy.signal import resample_poly
REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'training/cnn_v2'))
from cnn_v2 import all_features,read_wav_mono,resample_linear
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--old-cache',type=pathlib.Path,required=True);p.add_argument('--inputs',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True,parents=True);assert not (out/'features.npz').exists()
records=json.loads((a.inputs/'esc-manifest.json').read_text(encoding='utf-8'))['inputs'];lookup={r['filename']:r for r in records}
with np.load(a.old_cache,allow_pickle=False) as z:
    clean=z['X'][0].copy();meta={k:z[k].copy() for k in ('Y','FOLDS','FILES','SOURCES')}
phone=[];checks=[];clean_delta=0.
for i,filename in enumerate(meta['FILES']):
    row=lookup[str(filename)];path=a.inputs/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    pcm=read_wav_mono(path);features=all_features(pcm);clean_delta=max(clean_delta,float(np.max(np.abs(features-clean[i]))));assert clean_delta<.00001
    low=np.clip(np.round(resample_poly(pcm.astype(np.float32),1,2,window=('kaiser',5.0))),-32768,32767).astype(np.int16)
    restored=resample_linear(low,8000,16000);phone.append(all_features(restored))
    checks.append(dict(filename=str(filename),canonical_input_sha256=row['sha256'],phone_pcm_sha256=hashlib.sha256(restored.tobytes()).hexdigest()))
    if (i+1)%50==0:print('CORRECT PHONE',i+1,'/390',flush=True)
np.savez_compressed(out/'features.npz',X=np.stack([clean,np.stack(phone)]),**meta)
tone=(10000*np.sin(2*np.pi*6000*np.arange(16000)/16000)).astype(np.int16)
old=resample_linear(tone,16000,8000).astype(np.float64)
correct=resample_poly(tone.astype(np.float32),1,2,window=('kaiser',5.0)).astype(np.float64)
# Exclude filter boundary transients when measuring alias suppression.
rms=lambda x:float(np.sqrt(np.mean(x[100:-100]**2)))
receipt=dict(old_cache_sha256=hashlib.sha256(a.old_cache.read_bytes()).hexdigest(),new_cache_sha256=hashlib.sha256((out/'features.npz').read_bytes()).hexdigest(),
    clean_view_max_delta=clean_delta,clean_inputs_unchanged=True,external_audio_opened=False,
    phone_view='16k PCM -> SciPy resample_poly1/2 Kaiser5 anti-alias -> round/clip PCM16 -> canonical linear8k-to16k interpolation',
    diagnostic_tone_hz=6000,legacy_downsample_rms=rms(old),filtered_downsample_rms=rms(correct),
    alias_suppression_db=20*np.log10(rms(old)/max(rms(correct),1e-12)),records=checks)
(out/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print('CORRECTED PHONE CACHE FROZEN; alias suppression dB',receipt['alias_suppression_db'],flush=True)
