"""Upload frozen reviewed WAVs and verify byte-for-byte roundtrip on new firmware."""
import argparse, hashlib, json, re, time
from pathlib import Path
from board_io import Nsh, adb

parser = argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--image-sha256',required=True)
args = parser.parse_args()
assert re.fullmatch(r'[0-9a-f]{64}',args.image_sha256)
out = args.output
out.mkdir(parents=True,exist_ok=True)
evidence = out / 'evidence'
evidence.mkdir(exist_ok=True)
assert not (evidence/'flash-success.json').exists(), 'Preserve prior setup evidence'
manifest = json.loads((args.data / 'manifest.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
shell = Nsh(evidence / 'test-setup-shell.log')
try:
    text, rc = shell.run('ps')
    assert rc == 0 and not re.search(r'\barecord\s+-D', text), 'Pause concurrent capture first'
    text, rc = shell.run('ls -l /dev/bootloader')
    assert rc == 0 and re.search(r'\b16777216\b', text), 'Expected new 16 MiB AP partition'
    marker = out / 'owned-test-directory.txt'
    if marker.exists():
        assert marker.read_text().strip() == '/data/c20'
    else:
        _, rc = shell.run('mkdir /data/c20')
        assert rc == 0, 'Do not overwrite an unknown existing directory'
        marker.write_text('/data/c20\n')
    for i, row in enumerate(manifest['records']):
        assert sha(args.data / 'b' / f'{i:03}.wav') == row['sha256']
    adb(['push', str(args.data / 'b'), '/data/c20'], evidence / 'inputs-push.log', 120)
    assert not (out / 'input-roundtrip').exists(), 'Preserve previous roundtrip evidence'
    adb(['pull', '/data/c20/b', str(out / 'input-roundtrip')], evidence / 'inputs-pull.log', 120)
    for i, row in enumerate(manifest['records']):
        assert sha(out / 'input-roundtrip' / f'{i:03}.wav') == row['sha256']
    command = 'time "audiodetect /data/c20/b/000.wav --debug"'
    start = time.monotonic()
    text, rc = shell.run(command, 180)
    elapsed = time.monotonic() - start
    (evidence / 'first-timed-inference.log').write_text(text, encoding='utf-8')
    print(text, flush=True)
    expected = manifest['expected_cnn_v2_model_sha256']
    assert rc == 0 and f'Model SHA256: {expected}' in text
    (evidence / 'flash-success.json').write_text(json.dumps({
        'status': 'RUNNING_MODEL_AND_INPUTS_VERIFIED',
        'image_hash_scope': 'Operator-declared flashed image; this script does not read back NAND or flash hardware',
        'model_sha256': expected, 'ap_partition_bytes': 16777216,
        'inputs_roundtrip_sha256_pass': True, 'recordings': len(manifest['records']),
        'first_inference_host_wall_seconds': elapsed,
        'image_sha256': args.image_sha256
    }, ensure_ascii=False, indent=2), encoding='utf-8')
finally:
    shell.close()
