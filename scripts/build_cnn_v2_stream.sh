#!/usr/bin/env sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
app_dir="$repo_dir/app/audiodetect"
out_dir=${OUT_DIR:-$repo_dir/out/cnn-v2}
model_dir=${MODEL_DIR:-$app_dir}
mkdir -p "$out_dir"
src_dir="$out_dir/stream-source"
mkdir -p "$src_dir"
cp "$app_dir/cnn_stream_v2.c" "$app_dir/cnn_stream_v2.h" "$src_dir/"
cp "$model_dir/cnn_model_config_v2.h" "$model_dir/cnn_weights_v2.h" "$src_dir/"
cc -std=c11 -O2 -Wall -Wextra -pedantic -I "$src_dir" \
  "$src_dir/cnn_stream_v2.c" "$repo_dir/simulator/cnn_v2/cnn_cli.c" \
  -lm -o "$out_dir/cnn_stream"
echo "Built causal PCM/WAV streaming CLI: $out_dir/cnn_stream"
