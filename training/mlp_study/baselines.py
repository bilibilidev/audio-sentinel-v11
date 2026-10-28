"""Fresh MFCC SVM/RF and PANNs frozen-feature baselines for the formal MLP study."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import argparse,hashlib,json,pathlib,sys,time,platform
import numpy as np
import torch,sklearn,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'training'))
from extract_features_mfcc import extract_features_mfcc_from_pcm
from audio_utils import read_wav_mono
from numerics import LABELS,metrics
p=argparse.ArgumentParser(description=__doc__)
for n in ('inputs','panns','output'):p.add_argument('--'+n,type=pathlib.Path,required=True)
a=p.parse_args();ROOT=a.output.resolve();OUT=ROOT;OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'summary.json').exists()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs=a.inputs.resolve();esc=read(inputs/'esc-manifest.json')['inputs'];external=read(inputs/'manifest.json')['inputs']
lookup={v['filename']:v for v in esc};selection=read(REPO/'training/mlp_study/esc50-selection.json')['records']
ordered=[lookup[r['filename']] for r in selection]+external
paths=[inputs/r['path'] for r in ordered]
for path,row in zip(paths,ordered):assert sha(path)==row['sha256']
y=np.asarray([r['label_id'] for r in selection]);folds=np.asarray([int(r['fold']) for r in selection]);sources=np.asarray([r['src_file'] for r in selection])
ey=np.asarray([LABELS.index(r['reference_label']) for r in external])
save(OUT/'protocol.json',dict(date='2026-10-26',source_sha256=sha(pathlib.Path(__file__)),
    inputs=[dict(id=r['id'],sha256=r['sha256']) for r in ordered],supervised_records=390,external_records=65,
    mfcc_grids=dict(SVM_C=[.1,1,10,100],RF_depth=[None,12],RF_min_leaf=[1,3],RF_trees=300),panns_svm_C=[.01,.1,1,10],
    selection='Inner validation only0.6acc+0.3macro+0.1baby; next official fold validates, remaining3folds train, refit4folds then score held-out. Final chooses meaninnerutility, fits390.',
    canonical_inputs='All methods use exact same16k monoPCM16 WAV bytes; PANNs resamples them to32k without restoring lost bandwidth.',
    external_role='Already-seen development65, no supervised fitting or parameter tuning; not blind. AudioSet pretraining overlap cannot be fully audited.',
    model_artifacts='Six fitted estimators per baseline retained locally asjoblib; artifact hashes published; no audio/model pickles distributed.',
    environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,sklearn=sklearn.__version__,platform=platform.platform())))
values=[]
for i,path in enumerate(paths):
    values.append(extract_features_mfcc_from_pcm(read_wav_mono(path,16000),include_delta=True)[40:])
    if (i+1)%100==0:print('FRESH MFCC',i+1,'/455',flush=True)
mfcc=np.stack(values);np.savez_compressed(OUT/'mfcc-features.npz',X=mfcc)
assert mfcc.shape==(455,52) and np.isfinite(mfcc).all()
def score(truth,pred):
    m=metrics(truth,pred);return .6*m['accuracy']+.3*m['balanced_accuracy_present_classes']+.1*m['per_class_recall']['baby_cry']
def classifier(name,params):
    if name=='MFCC_SVM':return make_pipeline(StandardScaler(),SVC(C=params['C'],kernel='rbf',gamma='scale',class_weight='balanced'))
    if name=='MFCC_RF':return RandomForestClassifier(n_estimators=300,max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'],class_weight='balanced',random_state=42,n_jobs=2)
    return make_pipeline(StandardScaler(),LinearSVC(C=params['C'],class_weight='balanced',random_state=42,max_iter=10000))

def evaluate(name,features,grid,pretrained):
    results=[];pooled_truth=[];pooled_pred=[]
    for fold in range(1,6):
        train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold);inner_val=fold%5+1
        itrain=np.flatnonzero((folds!=fold)&(folds!=inner_val));ival=np.flatnonzero(folds==inner_val)
        assert set(sources[train]).isdisjoint(set(sources[test]))
        choices=[]
        for param in grid:
            model=classifier(name,param);model.fit(features[itrain],y[itrain]);pred=model.predict(features[ival]);choices.append((score(y[ival],pred),param))
        utility,best=max(choices,key=lambda r:r[0]);model=classifier(name,best);model.fit(features[train],y[train]);pred=model.predict(features[test])
        joblib.dump(model,OUT/(name+f'-fold-{fold}.joblib'));external_pred=model.predict(features[390:]);item={'fold':fold,'chosen_parameters':best,'inner_utility':utility,'source_overlap':0,'held_out':metrics(y[test],pred),'external':metrics(ey,external_pred),
            'held_out_predictions':[{'filename':selection[i]['filename'],'truth':int(y[i]),'prediction':int(p)} for i,p in zip(test,pred)],
            'external_predictions':[{'id':r['id'],'reference_label':r['reference_label'],'prediction':LABELS[int(p)]} for r,p in zip(external,external_pred)]}
        results.append(item);pooled_truth.extend(y[test].tolist());pooled_pred.extend(pred.tolist())
        save(OUT/(name+f'-fold-{fold}.json'),item);print(name,fold,'CV',item['held_out']['accuracy'],'baby',item['held_out']['per_class_recall']['baby_cry'],'external',item['external']['accuracy'],flush=True)
    # Final parameters use aggregate inner validation, not external predictions.
    parameter_scores=[]
    for param in grid:
        values=[]
        for fold in range(1,6):
            inner=fold%5+1;tr=np.flatnonzero((folds!=fold)&(folds!=inner));va=np.flatnonzero(folds==inner)
            m=classifier(name,param);m.fit(features[tr],y[tr]);values.append(score(y[va],m.predict(features[va])))
        parameter_scores.append((float(np.mean(values)),param))
    _,params=max(parameter_scores,key=lambda r:r[0]);model=classifier(name,params);model.fit(features[:390],y)
    joblib.dump(model,OUT/(name+'-final.joblib'));pred=model.predict(features[390:]);final=metrics(ey,pred);final['predictions']=[{'id':r['id'],'reference_label':r['reference_label'],'prediction':LABELS[int(p)]} for r,p in zip(external,pred)]
    value={'baseline':name,'pretrained_extra_data':pretrained,'folds':results,'held_out':metrics(pooled_truth,pooled_pred),
           'external_fold_mean_accuracy':float(np.mean([r['external']['accuracy'] for r in results])), 'final_parameters':params,'external_final':final}
    save(OUT/(name+'-summary.json'),value);return value

baselines=[]
for name,grid in [('MFCC_SVM',[{'C':c} for c in [.1,1,10,100]]),
                  ('MFCC_RF',[{'max_depth':d,'min_samples_leaf':m} for d in [None,12] for m in [1,3]])]:
    baselines.append(evaluate(name,mfcc,grid,False))

panns_dir=a.panns.resolve();sys.path.insert(0,str(panns_dir));from models import Cnn6
import librosa
torch.set_num_threads(2)
emb_path=OUT/'panns-embeddings.npz'
if emb_path.exists():
    with np.load(emb_path,allow_pickle=False) as z:emb=z['X']
else:
    checkpoint=panns_dir/'Cnn6_mAP=0.343.pth';assert sha(checkpoint)==read(panns_dir/'provenance.json')['checkpoint']['sha256']
    model=Cnn6(32000,1024,320,64,50,14000,527)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(state['model']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    embeddings=[];started=time.monotonic()
    with torch.no_grad():
        for i,path in enumerate(paths):
            pcm,_=librosa.load(path,sr=32000,mono=True,dtype=np.float32)
            value=model(torch.from_numpy(pcm[None]))['embedding'][0].numpy();embeddings.append(value)
            if (i+1)%25==0:print('PANNs FROZEN EMBEDDINGS',i+1,'/455 elapsed',round(time.monotonic()-started,1),flush=True)
    emb=np.stack(embeddings);np.savez_compressed(emb_path,X=emb)
assert emb.shape==(455,512) and np.isfinite(emb).all()
baselines.append(evaluate('PANNs_Cnn6_linearSVM',emb,[{'C':c} for c in [.01,.1,1,10]],True))
summary={'baselines':baselines,'feature_protocols':{'MFCC':'52 full-recording MFCC/delta means/std from same canonicalPCM16 inputs as MLP','PANNs':'Official frozen AudioSet Cnn6, canonicalPCM16 resampled to32k bylibrosa,512dims; no extra original high-rate bandwidth'},
         'supervised_training_recordings':390,'external_recordings':65,'external_training':False,'external_labels_user_confirmed':True,
         'selection':'Official held-out folds; next-fold inner validation, same0.6acc+0.3macro+0.1baby utility as MLP selection; scaler/estimator fit on training only',
         'pretraining_limit':'PANNs uses extra AudioSet pretraining. It is a separate transfer-reference track, not an equal-data or deployable-board comparator; upstream pretraining overlap is not exhaustively auditable',
         'cloud_api':'Not run. No recordings uploaded and no API cost incurred.'}
save(OUT/'summary.json',summary);print('BASELINES COMPLETE',flush=True)

save(OUT/'artifact-hashes.json',{p.name:sha(p) for p in OUT.glob('*.joblib')})
