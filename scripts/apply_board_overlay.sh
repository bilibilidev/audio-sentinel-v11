#!/usr/bin/env sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 OPENVELA_WORKSPACE_ROOT" >&2
  exit 2
fi

workspace=$1
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
team_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
board_dir=$workspace/vendor/allwinnertech/boards/r528/r528s3-gemini-s1

test -d "$workspace/nuttx"
test -d "$board_dir/configs/nsh"
test -f "$team_dir/board/r528s3-gemini-s1/configs/nsh/defconfig"
test -f "$team_dir/board/r528s3-gemini-s1/src/etc/init.d/rcS.nsh"

# Apply only to the board SDK's LVGL checkout. Refuse unknown source revisions
# instead of silently constructing a different display driver.
lvgl_dir=$workspace/apps/graphics/lvgl/lvgl
lcd_patch=$team_dir/board/r528s3-gemini-s1/patches/lvgl-aligned-buffers.patch
test -d "$lvgl_dir/.git" || test -f "$lvgl_dir/.git"
if git -C "$lvgl_dir" apply --reverse --check "$lcd_patch" 2>/dev/null; then
  echo "LVGL aligned LCD buffers already applied"
elif git -C "$lvgl_dir" apply --check "$lcd_patch"; then
  git -C "$lvgl_dir" apply "$lcd_patch"
else
  echo "LVGL LCD source differs from the validated version; inspect the patch." >&2
  exit 1
fi

cp "$team_dir/board/r528s3-gemini-s1/configs/nsh/defconfig" \
  "$board_dir/configs/nsh/defconfig"
cp "$team_dir/board/r528s3-gemini-s1/src/etc/init.d/rcS.nsh" \
  "$board_dir/src/etc/init.d/rcS.nsh"

echo "Applied Audio Sentinel v11 board overlay to: $board_dir"
