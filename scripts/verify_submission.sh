#!/usr/bin/env sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir"
python3 - <<'PY'
import hashlib,json,pathlib,xml.etree.ElementTree as ET
root=pathlib.Path('.')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
required=['README.md','audio-sentinel-v11.xml','app/audiodetect/audiodetect_main.c','app/audiodetect/audiodetect_ui.c','app/audiodetect/audio_classifier.c','app/audiodetect/feature_extractor.c','app/audiodetect/mlp_model_infer.c','app/audiodetect/keyword_templates.h','scripts/build_audio_sentinel_app.sh','scripts/build_r528s3_image.sh','training/mlp_study/train_ablations.py','training/mlp_study/export_model.py','demos/run_demo.sh']
for n in required:assert (root/n).is_file(),n
models=root/'model/mlp-optimized-20261024';deployment=json.loads((models/'deployment.json').read_text())
assert deployment['architecture']==[92,96,5] and deployment['parameters']==9413
assert deployment['training_recordings']==390
assert sha(models/'final_all_esc50.npz')==deployment['model_sha256']
for name,digest in deployment['header_sha256'].items():assert sha(root/'app/audiodetect'/name)==digest,name
assert deployment['model_sha256'] in (root/'app/audiodetect/mlp_model_config.h').read_text()
assert '#define MLP_TIME_TRANSFORM_ASINH40 1' in (root/'app/audiodetect/mlp_weights.h').read_text()
study=root/'model/mlp-study-20261026';hashes=json.loads((study/'sha256.json').read_text());assert len(hashes)==60
for name,digest in hashes.items():assert sha(study/name)==digest,name
results=root/'results/mlp-study-20261026';summary=json.loads((results/'summary.json').read_text())
assert (summary['fixed_recipe_cv']['correct'],summary['fixed_recipe_cv']['total'])==(337,390)
assert (summary['external_final']['correct'],summary['external_final']['total'])==(57,65)
for variant in json.loads((results/'linux/ablations.json').read_text()):
 rows=[r for m in variant['models'] if 'official' in m for r in m['official']['records']]
 assert len(rows)==390 and len({r['id'] for r in rows})==390
 assert sum(r['truth']==r['prediction'] for r in rows)==variant['official']['correct']
selection=json.loads((root/'training/mlp_study/esc50-selection.json').read_text())['records']
assert len(selection)==390
source_folds={}
for r in selection:source_folds.setdefault(r['src_file'],set()).add(int(r['fold']))
assert all(len(v)==1 for v in source_folds.values())
manifest=ET.parse(root/'audio-sentinel-v11.xml').getroot()
assert any(r.get('fetch')=='https://github.com/bilibilidev/' for r in manifest.findall('remote'))
assert any(p.get('name')=='audio-sentinel-v11' and p.get('revision')=='main' for p in manifest.findall('project'))
print('Tiny MLP source/model checks: PASS')
print('Model SHA256:',deployment['model_sha256'])
print('Controlled models:60; fixed five-fold337/390; external57/65')
PY
