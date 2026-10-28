#!/usr/bin/env sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
app_dir="$repo_dir/app/audiodetect"
out_dir=${OUT_DIR:-$repo_dir/out/audio-sentinel}
if [ "${DEFAULT_MODEL:-mlp}" != mlp ]; then
  echo 'This release builds the formal Tiny MLP backend.' >&2
  exit 2
fi
mkdir -p "$out_dir/source"
for file in audiodetect_main.c audio_classifier.c audio_classifier.h wav_reader.c wav_reader.h \
            keyword_detect.c keyword_detect.h keyword_templates.h alarm_report.c alarm_report.h \
            feature_extractor.c feature_extractor.h mlp_model_infer.c mlp_model_infer.h mlp_weights.h mlp_model_config.h; do
  cp "$app_dir/$file" "$out_dir/source/$file"
done
(cd "$out_dir/source" && "${CC:-cc}" -std=c11 -O2 -Wall -Wextra -pedantic \
  audiodetect_main.c audio_classifier.c wav_reader.c keyword_detect.c alarm_report.c \
  feature_extractor.c mlp_model_infer.c -lm -o ../audiodetect)
echo "Built Tiny MLP: $out_dir/audiodetect"
