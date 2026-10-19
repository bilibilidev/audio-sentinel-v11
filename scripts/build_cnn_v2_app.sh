#!/usr/bin/env sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
app_dir="$repo_dir/app/audiodetect"
out_dir=${OUT_DIR:-$repo_dir/out/cnn-v2}
model_dir=${MODEL_DIR:-$app_dir}
mkdir -p "$out_dir"
# MODEL_DIR may point at a private exported fold for numerical validation.
src_dir="$out_dir/source"
mkdir -p "$src_dir"
for file in audiodetect_main.c audio_classifier.c audio_classifier.h \
            cnn_stream_v2.c cnn_stream_v2.h wav_reader.c wav_reader.h \
            keyword_detect.c keyword_detect.h keyword_templates.h \
            alarm_report.c alarm_report.h; do
  cp "$app_dir/$file" "$src_dir/$file"
done
cp "$model_dir/cnn_model_config_v2.h" "$src_dir/"
cp "$model_dir/cnn_weights_v2.h" "$src_dir/"
cc -std=c11 -O2 -Wall -Wextra -pedantic -I "$src_dir" \
  "$src_dir/audiodetect_main.c" "$src_dir/audio_classifier.c" \
  "$src_dir/cnn_stream_v2.c" "$src_dir/wav_reader.c" \
  "$src_dir/keyword_detect.c" "$src_dir/alarm_report.c" \
  -lm -o "$out_dir/audiodetect"
echo "Built shared CNN v2 App CLI: $out_dir/audiodetect"
