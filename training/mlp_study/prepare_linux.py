"""Prepare a portable Linux acceptance directory from frozen models/canonical WAVs."""
import argparse,hashlib,json,pathlib,shutil,sys,wave
import numpy as np
R=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'training'))
from c_arrays import int8_header,c_float_array
p=argparse.ArgumentParser(description=__doc__)
for n in ('inputs','study','output'):p.add_argument('--'+n,type=pathlib.Path,required=True)
a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8',newline='\n')
for n in ('manifest.json','esc-manifest.json','keyword-manifest.json'):
    (out/'inputs').mkdir(exist_ok=True);shutil.copyfile(a.inputs/n,out/'inputs'/n)
for n in ('inputs','esc-inputs','keyword-inputs'):shutil.copytree(a.inputs/n,out/'inputs'/n)
shutil.copytree(a.study,out/'study');frozen=read(out/'study/frozen.json');assert len(frozen['models'])==60
harness=R/'training/mlp_study/study_harness.c'
for item in frozen['models']:
    d=out/'study'/item['variant']/item['model'];assert sha(d/'model.npz')==item['model_sha256'];assert sha(d/'weights.h')==item['header_sha256'];shutil.copyfile(harness,d/'harness.c')
manifest={};esc=read(a.inputs/'esc-manifest.json')['inputs']
for snr in (20,10,5):
    rows=[];d=out/'inputs/noise'/str(snr);d.mkdir(parents=True)
    for idx,row in enumerate(esc):
        assert sha(a.inputs/row['path'])==row['sha256']
        with wave.open(str(a.inputs/row['path']),'rb') as f:pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float64)/32768
        rng=np.random.default_rng(2026102600+snr*1000+idx);noise=rng.standard_normal(len(pcm));noise*=np.sqrt(np.mean(pcm**2)/10**(snr/10))/np.sqrt(np.mean(noise**2))
        result=np.clip(np.rint((pcm+noise)*32768),-32768,32767).astype('<i2');path=d/(row['id']+'.wav')
        with wave.open(str(path),'wb') as f:f.setparams((1,2,16000,0,'NONE','not compressed'));f.writeframes(result.tobytes())
        actual=result.astype(np.float64)/32768-pcm
        rows.append({**row,'id':f'snr{snr}-'+row['id'],'path':f'noise/{snr}/'+path.name,'sha256':sha(path),'clean_sha256':row['sha256'],'target_snr_db':snr,'achieved_snr_db':float(10*np.log10(np.mean(pcm**2)/np.mean(actual**2))),'post_quantization_clipped_samples':int(np.sum((result==32767)|(result==-32768)))})
    manifest[str(snr)]=rows
write(out/'inputs/noise-manifest.json',manifest)
write(out/'preparation.json',dict(source_sha256=sha(pathlib.Path(__file__)),harness_sha256=sha(harness),external_audio_training=False))
print('Linux input directory prepared:',out)
