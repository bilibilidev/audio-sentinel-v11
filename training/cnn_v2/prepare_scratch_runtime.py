"""Prepare hash-checked WAV and reference cases for an isolated native C run."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse, hashlib, json, pathlib, shutil, subprocess, sys, tarfile
import numpy as np
from cnn_v2 import REPO, read_wav_mono, all_features
from audio_utils import write_wav_mono
from keyword_detect import KEYWORDS, synth_keyword_pcm

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--training', type=pathlib.Path, required=True)
p.add_argument('--evaluation', type=pathlib.Path, required=True)
p.add_argument('--esc50', type=pathlib.Path, required=True)
a = p.parse_args()
out = a.evaluation.resolve()
read = lambda f: json.loads(f.read_text(encoding='utf-8'))
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')

frozen = read(a.training/'training-frozen.json')
cache_path = REPO.parent/'model-improvement-v2-20261019/artifacts/esc50-features.npz'
assert sha(cache_path) == read(a.training/'protocol.json')['features_sha256']
with np.load(cache_path, allow_pickle=False) as z:
    cache = z['X'][0,:,:26,:].copy()
esc = out/'esc-inputs'; esc.mkdir(exist_ok=True)
rows = read(REPO/'training/cnn/esc50-selection.json')['records']
inputs = []; feature_delta = 0.
for i, row in enumerate(rows):
    source = a.esc50/'audio'/row['filename']
    assert sha(source) == row['raw_sha256'], row['filename']
    pcm = read_wav_mono(source, 16000)
    dest = esc/f'esc-{i:03d}.wav'; write_wav_mono(dest, pcm, 16000)
    feature_delta = max(feature_delta, float(np.max(np.abs(all_features(pcm)[:26]-cache[i]))))
    inputs.append(dict(id=dest.stem, path=dest.relative_to(out).as_posix(), filename=row['filename'],
                       fold=int(row['fold']), truth=row['label_id'], raw_sha256=row['raw_sha256'], sha256=sha(dest)))
assert feature_delta < .00001, feature_delta
refs = {item['model']: item['int8']['predictions'] for item in frozen['folds']}
save(out/'esc-manifest.json', dict(inputs=inputs, references=refs, cache_feature_max_delta=feature_delta))

parity = out/'parity'
subprocess.run([sys.executable, str(REPO/'training/cnn_v2/prepare_c_check.py'),
                str(a.training/'models/final_all_esc50.npz'), str(esc/'esc-000.wav'), str(parity)], check=True)
cases = {}
for name in ('input','prefix','short','silence'):
    with np.load(parity/(name+'-reference.npz'), allow_pickle=False) as z:
        cases[name] = {k: z[k].tolist() for k in z.files}
save(parity/'references.json', cases)

kwdir = out/'keyword-inputs'; kwdir.mkdir(exist_ok=True)
keywords = []
for k, keyword in enumerate(KEYWORDS):
    for seed in range(3):
        pcm = synth_keyword_pcm(keyword, np.random.default_rng(20261024+seed+k*10))
        if isinstance(pcm, tuple): pcm = pcm[0]
        dest = kwdir/f'kw-{k}-{seed}.wav'; write_wav_mono(dest, pcm, 16000)
        keywords.append(dict(path=dest.relative_to(out).as_posix(), expected=k, synthetic=True, sha256=sha(dest)))
for name, pcm in [('silence', np.zeros(16000,dtype=np.int16)),
                  ('low-noise', np.random.default_rng(19).integers(-20,21,16000,dtype=np.int16))]:
    dest = kwdir/(name+'.wav'); write_wav_mono(dest, pcm, 16000)
    keywords.append(dict(path=dest.relative_to(out).as_posix(), expected=-1, synthetic=True, sha256=sha(dest)))
save(out/'keyword-manifest.json', dict(cases=keywords, scope='Synthetic fixtures and silence only; NOT real speech accuracy'))
for name in ('verify_scratch_runtime.py','verify_aux_features.py'):
    shutil.copyfile(REPO/'training/cnn_v2'/name, out/name)
files = {f.relative_to(out).as_posix(): sha(f) for f in out.rglob('*') if f.is_file()}
save(out/'runtime-inputs.json', dict(files=files, deployment_model_unchanged=True,
     final_model_sha256=frozen['final_model_sha256'], application_base_commit=read(a.training/'protocol.json')['application_base_commit']))
with tarfile.open(out.with_name('runtime-supplement.tar.gz'), 'w:gz') as tar:
    for path in [esc, parity, kwdir, out/'esc-manifest.json', out/'keyword-manifest.json',
                 out/'verify_scratch_runtime.py', out/'verify_aux_features.py', out/'runtime-inputs.json']:
        tar.add(path, arcname=path.relative_to(out).as_posix())
print('Prepared runtime supplement; 390 raw hashes verified; feature cache max delta', feature_delta)
