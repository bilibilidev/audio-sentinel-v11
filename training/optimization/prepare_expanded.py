"""Cache all2000 ESC recordings: extra1610 backgrounds, no external audio."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import argparse,csv,hashlib,json,pathlib,sys,time
import numpy as np
REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'training'));sys.path.insert(0,str(REPO/'training/cnn_v2'))
from audio_utils import read_wav_mono
from extract_features_mfcc import extract_features_mfcc_from_pcm
from cnn_v2 import all_features,LABELS
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
assert not (out/'features.npz').exists()
selection=json.loads((REPO/'training/cnn/esc50-selection.json').read_text(encoding='utf-8'))
meta=a.dataset/'meta/esc50.csv';assert hashlib.sha256(meta.read_bytes()).hexdigest()==selection['metadata_sha256']
selected={r['filename']:r for r in selection['records']}
target={'dog':'dog_bark','crying_baby':'baby_cry','coughing':'cough','glass_breaking':'glass_break'}
rows=sorted(csv.DictReader(meta.open(encoding='utf-8',newline='')),key=lambda r:r['filename'])
source_folds={source:{r['fold'] for r in rows if r['src_file']==source} for source in {r['src_file'] for r in rows}}
conflicts={s for s,folds in source_folds.items() if len(folds)>1}
# Keep the existing390 evaluation cohort unchanged. Remove extra records of
# conflicting source IDs rather than accepting leakage from added backgrounds.
excluded=[r for r in rows if r['src_file'] in conflicts and r['filename'] not in selected]
rows=[r for r in rows if r not in excluded]
mel=[];mlp=[];labels=[];folds=[];sources=[];files=[];mask=[];proof=[];start=time.monotonic()
for i,row in enumerate(rows):
    path=a.dataset/'audio'/row['filename'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if row['filename'] in selected:assert digest==selected[row['filename']]['raw_sha256']
    pcm=read_wav_mono(path);assert len(pcm)==80000
    mel.append(all_features(pcm));mlp.append(np.stack([extract_features_mfcc_from_pcm(pcm[s:s+16000],True) for s in range(0,80000,16000)]))
    labels.append(LABELS.index(target.get(row['category'],'background')));folds.append(int(row['fold']));sources.append(row['src_file']);files.append(row['filename']);mask.append(row['filename'] in selected)
    proof.append(dict(filename=row['filename'],raw_sha256=digest,fold=int(row['fold']),category=row['category'],src_file=row['src_file']))
    if (i+1)%100==0:print('EXPANDED CACHE',i+1,'/2000',round(time.monotonic()-start,1),'s',flush=True)
assert len(rows)==2000-len(excluded) and sum(mask)==390
for source in set(sources):assert len({f for s,f in zip(sources,folds) if s==source})==1
np.savez_compressed(out/'features.npz',X_CNN=np.stack(mel).astype(np.float32),X_MLP=np.stack(mlp).astype(np.float32),Y=np.asarray(labels),FOLDS=np.asarray(folds),SOURCES=np.asarray(sources),FILES=np.asarray(files),EVAL_MASK=np.asarray(mask))
(out/'provenance.json').write_text(json.dumps(dict(recordings=len(rows),extra_background_recordings=len(rows)-390,evaluation_recordings=390,
    local_full_metadata_cross_fold_source_ids=sorted(conflicts),excluded_extra_background_records=excluded,
    external_opened=False,source_groups_crossing_folds=0,metadata_sha256=hashlib.sha256(meta.read_bytes()).hexdigest(),
    cache_sha256=hashlib.sha256((out/'features.npz').read_bytes()).hexdigest(),records=proof),indent=2)+'\n',encoding='utf-8')
print('EXPANDED CACHE FROZEN',flush=True)
