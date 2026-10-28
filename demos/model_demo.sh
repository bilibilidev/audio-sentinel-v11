#!/bin/sh
set -eu
demo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${PYTHON:-python3}" "$demo_dir/run_demo.py" --section model "$@"
