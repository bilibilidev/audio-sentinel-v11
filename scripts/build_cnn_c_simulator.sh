#!/usr/bin/env sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cnn_dir=$repo_dir/simulator/cnn
out_dir=${OUT_DIR:-$repo_dir/out/cnn-simulator}
mkdir -p "$out_dir"

"${CC:-cc}" -std=c11 -O2 -Wall -Wextra -pedantic \
  -I "$cnn_dir" "$cnn_dir/cnn_stream.c" "$cnn_dir/cnn_cli.c" \
  -lm -o "$out_dir/cnn_stream"

echo "Built Linux host CNN candidate: $out_dir/cnn_stream"
echo "Input: 16000 Hz mono PCM16 WAV, or --stdin-raw PCM16LE."
