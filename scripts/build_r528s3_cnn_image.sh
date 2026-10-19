#!/usr/bin/env bash
# Build and pack local files only. This script never flashes hardware.
set -eo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: bash $0 ISOLATED_OPENVELA_SDK OUTPUT_PARENT" >&2
  exit 2
fi
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
sdk=$(realpath "$1")
test -f "$sdk/.audio-sentinel-cnn-v2-isolated" || {
  echo "Use a separate SDK copy with the isolation marker; preserve the existing SDK." >&2; exit 2;
}
test "$(realpath "$sdk/apps/audiodetect")" = "$sdk/audio-sentinel-v11/app/audiodetect"
test "$(git -C "$sdk/audio-sentinel-v11" rev-parse HEAD)" = "$(git -C "$repo_dir" rev-parse HEAD)"
git -C "$repo_dir" diff --quiet HEAD -- app model scripts
git -C "$sdk/audio-sentinel-v11" diff --quiet HEAD -- app model scripts
model_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_sha256"])' "$repo_dir/model/cnn-v2/deployment.json")
python3 - "$repo_dir" "$model_sha" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'results/cnn-v2/runtime-acceptance.json'
assert p.exists(),'Final model requires actual Ubuntu C acceptance before release-image building'
r=json.loads(p.read_text());assert r['status']=='PASS' and r['model_sha256']==sys.argv[2]
assert r['actual_linux_c_recordings']==65 and r['app_cli_recordings']==65
PY
test -e /lib/ld-linux.so.2 || {
  echo "Official dragon packer needs approved Ubuntu 32-bit runtime dependencies." >&2; exit 2;
}
out_parent=$(realpath -m "$2")
out="$out_parent/cnn-v2-r528s3-${model_sha:0:12}"
test ! -e "$out" || { echo "Use a fresh output directory: $out" >&2; exit 2; }
mkdir -p "$out/debug"
export PATH="$sdk/prebuilts/gcc/linux-x86_64/arm-none-eabi/bin:$sdk/prebuilts/build-tools/linux-x86_64/bin:$PATH"
{
  uname -a
  arm-none-eabi-gcc --version
  git lfs version
} > "$out/toolchain-versions.txt"
(cd "$repo_dir" && sh scripts/verify_submission.sh) > "$out/source-integrity.log" 2>&1
sh "$repo_dir/scripts/apply_board_overlay.sh" "$sdk" > "$out/overlay.log" 2>&1
(cd "$sdk" && ./build.sh vendor/allwinnertech/boards/r528/r528s3-gemini-s1/configs/nsh/ -j6) > "$out/build.log" 2>&1
cp "$sdk/nuttx/.config" "$out/active.config"
cp "$sdk/nuttx/nuttx.elf" "$out/debug/nuttx.elf"
cp "$sdk/nuttx/vela.bin" "$out/debug/vela.bin"
arm-none-eabi-nm "$out/debug/nuttx.elf" > "$out/symbols.txt"
python3 - "$sdk" "$model_sha" <<'PY'
import hashlib,pathlib,sys
sdk=pathlib.Path(sys.argv[1]);raw=(sdk/'nuttx/vela.bin').read_bytes()
assert sys.argv[2].encode() in raw,'Final model SHA not present in application payload'
assert b'Audio Sentinel CNN v2' in raw
payload=sdk/'vendor/allwinnertech/lichee/board/r528s3/gemini-s1_nand/configs/nsh.fex'
assert hashlib.sha256(raw).digest()==hashlib.sha256(payload.read_bytes()).digest()
config=(sdk/'nuttx/.config').read_text()
for flag in ['CONFIG_ARCH_BOARD_R528S3_GEMINI_S1=y','CONFIG_LCD_ILI9341=y','CONFIG_AE_AUDIODETECT=y','CONFIG_AE_AUDIODETECT_UI=y']:
    assert flag in config,flag
PY
(cd "$sdk" && python3 .repo/repo/repo manifest -r -o "$out/manifest-resolved.xml") > "$out/manifest.log" 2>&1
git -C "$sdk/vendor/allwinnertech" diff > "$out/bsp-local.patch"
pack_root="$sdk/vendor/allwinnertech/lichee"
test ! -f "$pack_root/tools/scripts/.hooks/post-dragon" || {
  echo "Review the optional vendor post-dragon hook before packing." >&2; exit 2;
}
touch "$out/pack-start"
(
  cd "$pack_root"
  source envsetup.sh
  lunch_nuttx r528s3-gemini-s1
  pack
) > "$out/pack.log" 2>&1
# The vendor script can return success after dragon fails; validate fresh output.
python3 - "$pack_root" "$out" "$model_sha" "$repo_dir" <<'PY'
import hashlib,json,pathlib,shutil,subprocess,sys
pack,out=map(pathlib.Path,sys.argv[1:3]);model=sys.argv[3];repo=pathlib.Path(sys.argv[4]);start=(out/'pack-start').stat().st_mtime_ns
candidates=[]
for p in (pack/'out/r528s3/gemini-s1_nand').glob('*.img'):
    if p.stat().st_mtime_ns>=start and p.open('rb').read(8)==b'IMAGEWTY':candidates.append(p)
assert len(candidates)==1,('Expected one fresh Allwinner IMAGEWTY container',candidates)
ap=(out/'debug/vela.bin').read_bytes()
offset=candidates[0].read_bytes().find(ap)
assert offset>=0,'Packed container does not contain the exact newly built AP payload'
name=f'AudioSentinel-CNN-v2-R528S3-GeminiS1-ILI9341-{model[:12]}.img'
image=out/name;shutil.copyfile(candidates[0],image)
digest=hashlib.sha256(image.read_bytes()).hexdigest()
(out/'SHA256SUMS').write_text(digest+'  '+name+'\n')
receipt={'status':'BUILD_AND_PACK_PASS','target':'R528S3 Gemini S1 / 128MB NAND / ILI9341 320x240','image':name,'image_sha256':digest,'image_bytes':image.stat().st_size,
         'model_sha256':model,'application_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
         'exact_ap_payload_offset':offset,'ap_payload_sha256':hashlib.sha256(ap).hexdigest(),
         'flashed_by_this_script':False,'real_board_runtime_verified':False,'historical_image_byte_identity_claimed':False}
(out/'image-acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(image)
PY
