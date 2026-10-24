#!/usr/bin/env sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export DEFAULT_MODEL=cnn
export OUT_DIR=${OUT_DIR:-$repo_dir/out/cnn-v2}
exec sh "$repo_dir/scripts/build_audio_sentinel_app.sh"
