#!/usr/bin/env sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
app_dir="$repo_dir/app/audiodetect"
out_dir=${OUT_DIR:-$repo_dir/out/audio-sentinel}
model=${DEFAULT_MODEL:-mlp}
src_dir="$out_dir/source"
mkdir -p "$src_dir"
for file in audiodetect_main.c audio_classifier.c audio_classifier.h wav_reader.c wav_reader.h \
            keyword_detect.c keyword_detect.h keyword_templates.h alarm_report.c alarm_report.h; do
  cp "$app_dir/$file" "$src_dir/$file"
done
case "$model" in
  mlp)
    for file in feature_extractor.c feature_extractor.h mlp_model_infer.c mlp_model_infer.h mlp_weights.h mlp_model_config.h; do
      cp "$app_dir/$file" "$src_dir/$file"
    done
    backend_flag=""
    ;;
  cnn)
    model_dir=${MODEL_DIR:-$app_dir}
    cp "$app_dir/cnn_stream_v2.c" "$app_dir/cnn_stream_v2.h" "$src_dir/"
    cp "$model_dir/cnn_model_config_v2.h" "$model_dir/cnn_weights_v2.h" "$src_dir/"
    backend_flag="-DAED_USE_CNN"
    ;;
  *) echo "DEFAULT_MODEL must be mlp or cnn" >&2;exit 2;;
esac
# Source files are passed relative to the private source directory, so paths
# containing spaces remain valid; only fixed backend basenames are expanded.
if [ "$model" = mlp ]; then
  set -- feature_extractor.c mlp_model_infer.c
else
  set -- cnn_stream_v2.c
fi
(cd "$src_dir" && "${CC:-cc}" -std=c11 -O2 -Wall -Wextra -pedantic $backend_flag \
  audiodetect_main.c audio_classifier.c wav_reader.c keyword_detect.c alarm_report.c "$@" \
  -lm -o ../audiodetect)
echo "Built Audio Sentinel ($model): $out_dir/audiodetect"
