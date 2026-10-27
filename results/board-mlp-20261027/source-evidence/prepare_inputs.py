import hashlib,json,pathlib,re,shutil,wave
import numpy as np

root=pathlib.Path('board-validation-20261027')
source=pathlib.Path('model-from-scratch-20261024/external-evaluation')
app_logs=pathlib.Path('mlp-study-20261026/native/linux-results/app')
model='2c74151ca171eff88af7efa26b8a4b3dfc37ea42c7e77b0c9b379bd61610f9e9'
labels=['background','cough','glass_break','baby_cry','dog_bark']
inputs=root/'inputs'
assert not inputs.exists()
for d in ['b','n20','n10','n5','k','i']:(inputs/d).mkdir(parents=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,pcm):
    with wave.open(str(path),'wb') as w:
        w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(pcm)
rows=[];noisy={str(s):[] for s in [20,10,5]}
for i,row in enumerate(json.loads((source/'manifest.json').read_text())['inputs']):
    src=source/row['path'];assert sha(src)==row['sha256']
    with wave.open(str(src)) as w:
        assert (w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,16000)
        pcm=w.readframes(w.getnframes());assert len(pcm)//2==row['samples']
    p=inputs/'b'/f'{i:03}.wav';shutil.copyfile(src,p)
    text=(app_logs/f'mlp-{row["id"]}.log').read_text();assert model in text
    prediction=re.search(r'Result:\s*(\w+)',text)[1]
    probs={k:float(v) for k,v in re.findall(r'^\s*(background|cough|glass_break|baby_cry|dog_bark):\s*([\d.eE+-]+)',text,re.M)}
    assert set(probs)==set(labels)
    rows.append(dict(row,board_file=f'b/{i:03}.wav',host_reference_prediction=prediction,
                     host_reference_probabilities=[probs[k] for k in labels]))
    x=np.frombuffer(pcm,dtype='<i2').astype(np.float64)
    rms=float(np.sqrt(np.mean(x*x)))
    z=np.random.default_rng(2026102700+i).normal(size=len(x));z/=np.sqrt(np.mean(z*z))
    for snr in [20,10,5]:
        mixed=x+z*rms/(10**(snr/20));clipped=int(np.sum((mixed>32767)|(mixed< -32768)))
        y=np.rint(np.clip(mixed,-32768,32767)).astype('<i2')
        dest=inputs/f'n{snr}'/f'{i:03}.wav';save(dest,y.tobytes())
        err=np.mean((y.astype(np.float64)-x)**2)
        actual=float(10*np.log10(np.mean(x*x)/err)) if err else None
        noisy[str(snr)].append(dict(id=row['id'],reference_label=row['reference_label'],samples=row['samples'],
                                   board_file=f'n{snr}/{i:03}.wav',sha256=sha(dest),source_sha256=row['sha256'],
                                   target_snr_db=snr,actual_quantized_snr_db=actual,clipped_samples=clipped,seed=2026102700+i))
kw=[]
for i,row in enumerate(json.loads((source/'keyword-manifest.json').read_text())['cases']):
    dest=inputs/'k'/f'{i:02}.wav';src=source/row['path'];assert sha(src)==row['sha256'];shutil.copyfile(src,dest)
    kw.append(dict(row,board_file=f'k/{i:02}.wav'))
for n in [1,123,16000,16001,48000]:save(inputs/'i'/f'z{n}.wav',b'\0\0'*n)
save(inputs/'i'/'empty.wav',b'')
(inputs/'i'/'bad.wav').write_bytes(b'not a WAV file\n')
(inputs/'i'/'trunc.wav').write_bytes((inputs/'b'/'000.wav').read_bytes()[:40])
# Same non-silent 3-second fixture as the actual Linux benchmark.
fixed=pathlib.Path('mlp-study-20261026/native/linux-results/fixed3s.wav')
with wave.open(str(fixed)) as w:
    assert (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes())==(1,2,16000,48000)
shutil.copyfile(fixed,inputs/'f.wav')
manifest=dict(labels=labels,expected_model_sha256=model,operator_declared_image_sha256='fad544406f90ad4fdacf0cb35f5593331746455ddda2252be6427a5ded82d713',
              records=rows,noise=noisy,keywords=kw,fixed_wav_sha256=sha(inputs/'f.wav'),
              data_role='Previously heard development-validation65, includes20baby, used for development/selection; not blind or microphone field accuracy',
              external_gradient_training=False,gaussian_noise_scope='Deterministic synthetic additive Gaussian stress; not recorded room noise')
(root/'input-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
files={str(p.relative_to(inputs)):sha(p) for p in inputs.rglob('*') if p.is_file()}
(root/'input-sha256.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
print('Prepared',len(files),'files;',sum(p.stat().st_size for p in inputs.rglob('*') if p.is_file()),'bytes')
