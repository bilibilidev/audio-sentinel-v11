#!/usr/bin/env sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir"

required_files="
README.md
audio-sentinel-v11.xml
app/audiodetect/audiodetect_main.c
app/audiodetect/audiodetect_ui.c
app/audiodetect/audio_classifier.c
app/audiodetect/cnn_stream_v2.c
app/audiodetect/cnn_stream_v2.h
app/audiodetect/cnn_model_config_v2.h
app/audiodetect/cnn_weights_v2.h
app/audiodetect/keyword_detect.c
app/audiodetect/keyword_detect.h
app/audiodetect/keyword_templates.h
app/audiodetect/model_weights.h
app/audiodetect/model_weights_int8.h
simulator/model_infer_int8.c
simulator/generated/model_weights_int8.h
training/export_to_c.py
training/train_edge_mfcc.py
training/extract_features_mfcc.py
training/keyword_detect.py
model/tiny_mlp_mfcc92_robust_simulator.npz
model/cnn-v2/final_all_esc50.npz
model/cnn-v2/sha256.json
model/cnn-v2/deployment.json
model/mlp-retrained-20261024/final_all_esc50.npz
model/mlp-retrained-20261024/deployment.json
model/mlp-optimized-20261024/final_all_esc50.npz
model/mlp-optimized-20261024/deployment.json
app/audiodetect/mlp_model_config.h
app/audiodetect/mlp_weights.h
app/audiodetect/mlp_model_infer.c
scripts/build_audio_sentinel_app.sh
scripts/build_cnn_v2_app.sh
scripts/build_cnn_v2_stream.sh
results/cnn-v2/baselines.json
board/r528s3-gemini-s1/configs/nsh/defconfig
board/r528s3-gemini-s1/src/etc/init.d/rcS.nsh
results/python_cv/metrics.json
"

for file in $required_files; do
  test -s "$file" || { echo "missing or empty: $file" >&2; exit 1; }
done

if grep -R "your-github-login\|hello_app\|hello_quickapp" \
  README.md audio-sentinel-v11.xml app board training model results 2>/dev/null; then
  echo "placeholder content found" >&2
  exit 1
fi

expected=c96333aaae66d324b8b8cec48ffb0ec981ac4e975d3cf9a7d4a2fdba4307c97e
if command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum model/tiny_mlp_mfcc92_robust_simulator.npz | awk '{print $1}')
else
  actual=$(shasum -a 256 model/tiny_mlp_mfcc92_robust_simulator.npz | awk '{print $1}')
fi
test "$actual" = "$expected" || {
  echo "model checksum mismatch: $actual" >&2
  exit 1
}

python3 -m json.tool results/python_cv/metrics.json >/dev/null

# The three generated weight headers must describe the same network.
for header in app/audiodetect/model_weights.h \
              app/audiodetect/model_weights_int8.h \
              simulator/generated/model_weights_int8.h; do
  test "$(grep -c '#define FEATURE_DIM 92' "$header")" = "1" || {
    echo "unexpected FEATURE_DIM in $header" >&2; exit 1; }
  test "$(grep -c '#define HIDDEN_DIM 64' "$header")" = "1" || {
    echo "unexpected HIDDEN_DIM in $header" >&2; exit 1; }
  test "$(grep -c '#define NUM_CLASSES 5' "$header")" = "1" || {
    echo "unexpected NUM_CLASSES in $header" >&2; exit 1; }
done

python3 - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('.')
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
models=root/'model/cnn-v2'
hashes=json.loads((models/'sha256.json').read_text())
assert set(hashes)=={*[f'fold_{i}.npz' for i in range(1,6)],'final_all_esc50.npz'}
for name,expected in hashes.items():assert digest(models/name)==expected,name
deployment=json.loads((models/'deployment.json').read_text())
assert deployment['model_sha256']==hashes['final_all_esc50.npz']
assert deployment['training_recordings']==390
assert deployment['labels']==['background','cough','glass_break','baby_cry','dog_bark']
for name,expected in deployment['header_sha256'].items():
    assert digest(root/'app/audiodetect'/name)==expected,name
assert deployment['model_sha256'] in (root/'app/audiodetect/cnn_model_config_v2.h').read_text()
print('CNN v2 model/header hashes match:',deployment['model_sha256'])
models=root/'model/mlp-retrained-20261024'
deployment=json.loads((models/'deployment.json').read_text())
hashes=json.loads((models/'sha256.json').read_text())
for name,expected in hashes.items():assert digest(models/name)==expected,name
assert deployment['model_sha256']==hashes['final_all_esc50.npz']
assert deployment['training_recordings']==390
assert deployment['architecture']==[92,64,5]
directory=root/deployment.get('header_directory','app/audiodetect')
for name,expected in deployment['header_sha256'].items():assert digest(directory/name)==expected,name
assert deployment['model_sha256'] in (directory/'mlp_model_config.h').read_text()
print('Archived previous MLP model/header hashes match:',deployment['model_sha256'])
models=root/'model/mlp-optimized-20261024'
deployment=json.loads((models/'deployment.json').read_text())
hashes=json.loads((models/'sha256.json').read_text())
assert set(hashes)=={*[f'fold_{i}.npz' for i in range(1,6)],'final_all_esc50.npz'}
for name,expected in hashes.items():assert digest(models/name)==expected,name
assert deployment['model_sha256']==hashes['final_all_esc50.npz']
assert deployment['training_recordings']==390 and deployment['architecture']==[92,96,5]
assert deployment['parameters']==9413
for name,expected in deployment['header_sha256'].items():assert digest(root/'app/audiodetect'/name)==expected,name
assert deployment['model_sha256'] in (root/'app/audiodetect/mlp_model_config.h').read_text()
assert '#define MLP_TIME_TRANSFORM_ASINH40 1' in (root/'app/audiodetect/mlp_weights.h').read_text()
print('Default optimized MLP model/header hashes match:',deployment['model_sha256'])
PY

echo "Submission source tree is complete (integrity check; runtime acceptance is separate)."
echo "Model SHA256: $actual"
