"""Evaluate the verified CNN v2 board without changing weights or input clips."""
import argparse, hashlib, json, math, re, time
from pathlib import Path
from board_io import Nsh

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--manifest',type=Path,required=True)
args=parser.parse_args()
out = args.output
manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
labels = manifest['labels']
expected = manifest['expected_cnn_v2_model_sha256']
assert not (out / 'results.json').exists(), 'Preserve completed evaluation'
assert json.loads((out / 'evidence/flash-success.json').read_text(encoding='utf-8'))['model_sha256'] == expected
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for i, row in enumerate(manifest['records']):
    assert sha(out / 'input-roundtrip' / f'{i:03}.wav') == row['sha256']

def stats(values):
    values = sorted(values)
    def percentile(p):
        at = (len(values)-1)*p
        lo, hi = math.floor(at), math.ceil(at)
        return values[lo] + (values[hi]-values[lo])*(at-lo)
    return {'count':len(values), 'mean':sum(values)/len(values),
            'p50':percentile(.5), 'p95':percentile(.95), 'min':values[0], 'max':values[-1]}

shell = Nsh(out / 'evidence/evaluation-shell.log')
records = []
try:
    text, rc = shell.run('ps')
    assert rc == 0 and not re.search(r'\barecord\s+-D', text), 'Capture must be paused for timing'
    identity, rc = shell.run('uname -a')
    assert rc == 0
    for i, row in enumerate(manifest['records']):
        command = f'time "audiodetect /data/c20/b/{i:03}.wav --debug"'
        start = time.monotonic()
        text, rc = shell.run(command, timeout=180)
        host_elapsed = time.monotonic()-start
        (out / (row['id']+'.log')).write_text(text, encoding='utf-8')
        assert rc == 0, (row['id'], text)
        assert f'Model SHA256: {expected}' in text, 'Wrong running model'
        pred = re.search(r'Result:\s*(background|cough|glass_break|baby_cry|dog_bark)\b',text)
        counts = re.search(r'Samples:\s*(\d+)\s+Frames:\s*(\d+)',text)
        timing = re.search(r'(?:^|[\r\n])\s*(\d+\.\d+)\s+sec\b',text)
        assert pred and counts and timing, (row['id'], text)
        assert int(counts[1]) == row['samples'], 'Full-file sample count mismatch'
        probs = {k:float(v) for k,v in re.findall(
            r'^\s*(background|cough|glass_break|baby_cry|dog_bark):\s*([\d.eE+-]+)',text,re.M)}
        assert set(probs) == set(labels)
        delta = max(abs(probs[label]-row['host_reference_probabilities'][j]) for j,label in enumerate(labels))
        duration = row['samples']/16000
        elapsed = float(timing[1])
        record = {'id':row['id'], 'reference_label':row['reference_label'],
                  'prediction':pred[1], 'correct':pred[1]==row['reference_label'],
                  'input_sha256':row['sha256'], 'model_sha256':expected,
                  'samples':int(counts[1]), 'frames':int(counts[2]),
                  'audio_seconds':duration, 'board_command_seconds':elapsed,
                  'real_time_factor':elapsed/duration, 'host_wall_seconds':host_elapsed,
                  'probabilities':probs, 'host_reference_prediction':row['host_reference_prediction'],
                  'host_prediction_match':pred[1]==row['host_reference_prediction'],
                  'max_probability_abs_delta_vs_host':delta, 'command':command}
        records.append(record)
        (out / 'results-progress.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print(f"BOARD CNN v2 {i+1}/{len(manifest['records'])} {row['id']} {pred[1]} correct={record['correct']} {elapsed:.4f}s delta={delta:.3g}",flush=True)
    per_class = {}
    matrix = [[0]*len(labels) for _ in labels]
    for record in records:
        matrix[labels.index(record['reference_label'])][labels.index(record['prediction'])] += 1
    for label in labels:
        rows = [r for r in records if r['reference_label']==label]
        per_class[label] = {'correct':sum(r['correct'] for r in rows), 'total':len(rows),
                            'recall':sum(r['correct'] for r in rows)/len(rows)}
    correct = sum(r['correct'] for r in records)
    result = {'status':'ACTUAL_CNN_V2_BOARD_WAV_EVALUATION_COMPLETE',
              'execution':'Physical R528S3 Gemini S1, NuttX/OpenVela, ADB NSH audiodetect',
              'identity':identity, 'model_sha256':expected, 'recordings':len(records),
              'correct':correct, 'accuracy':correct/len(records), 'per_class':per_class,
              'labels':labels, 'confusion_matrix_truth_rows_prediction_columns':matrix,
              'host_prediction_matches':sum(r['host_prediction_match'] for r in records),
              'max_probability_abs_delta_vs_host':max(r['max_probability_abs_delta_vs_host'] for r in records),
              'board_command_seconds':stats([r['board_command_seconds'] for r in records]),
              'real_time_factor':stats([r['real_time_factor'] for r in records]),
              'host_wall_seconds':stats([r['host_wall_seconds'] for r in records]),
              'total_audio_seconds':sum(r['audio_seconds'] for r in records),
              'total_board_command_seconds':sum(r['board_command_seconds'] for r in records),
              'timing_scope':'NSH time: full WAV file IO + DSP + CNN + stdout under normal task scheduling; UI alive, capture paused at test start. Excludes USB transfer and microphone capture; not LVGL alert latency. First setup inference excluded; clip lengths vary.',
              'inputs_roundtrip_sha256_pass':True, 'full_file_sample_counts_pass':True,
              'external_gradient_training':False, 'external_used_for_model_selection':True,
              'dataset_role':'Previously reviewed development-validation set, not blind testing',
              'microphone_accuracy_measured':False, 'ui_capture_latency_measured':False,
              'records':records}
    (out / 'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','identity')},ensure_ascii=False,indent=2),flush=True)
finally:
    shell.close()
