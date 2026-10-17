"""Check C/Torch parity, chunk invariance and prefix causality on Linux.

Prepare fixtures with prepare_parity.py first. This verifier needs NumPy and a
C compiler, but no PyTorch installation on the Linux machine.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import pathlib
import subprocess

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=pathlib.Path, default=REPO/'out/cnn-parity')
    parser.add_argument('--sanitizers', action='store_true', help='Also compile/run AddressSanitizer + UndefinedBehaviorSanitizer')
    args = parser.parse_args()
    root = args.fixtures.resolve()
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    header_lf = (REPO/'simulator/cnn/cnn_weights.h').read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(header_lf).hexdigest() == protocol['header_sha256_lf']
    for name, digest in protocol['inputs'].items():
        assert sha(root/name) == digest, name
    cc = os.environ.get('CC', 'cc')
    sources = [str(REPO/'simulator/cnn/cnn_stream.c'), str(REPO/'simulator/cnn/cnn_cli.c')]
    binary = root/'cnn_stream'
    command = [cc, '-std=c11', '-O2', '-Wall', '-Wextra', '-pedantic', *sources, '-lm', '-o', str(binary)]
    subprocess.run(command, check=True)
    checks, commands, full_trace = [], [command], None
    for name in ['input', 'prefix', 'short', 'silence']:
        with np.load(root/(name+'-reference.npz'), allow_pickle=False) as z:
            expected = z['LOGITS']
        outputs, traces = [], []
        for chunk in protocol['chunk_sizes']:
            trace = root/f'{name}-{chunk}.tsv'
            cmd = [str(binary), '--chunk', str(chunk), '--trace', str(trace), str(root/(name+'.wav'))]
            commands.append(cmd)
            output = subprocess.check_output(cmd, text=True)
            rows = list(csv.DictReader(io.StringIO(trace.read_text()), delimiter='\t'))
            actual = np.asarray([[float(r[f'l{i}']) for i in range(5)] for r in rows])
            assert actual.shape == expected.shape
            delta = float(np.max(np.abs(actual-expected)))
            assert delta < protocol['max_abs_logit_tolerance'], (name, chunk, delta)
            exp_probs = np.exp(expected-expected.max(axis=1, keepdims=True)); exp_probs /= exp_probs.sum(axis=1, keepdims=True)
            act_probs = np.exp(actual-actual.max(axis=1, keepdims=True)); act_probs /= act_probs.sum(axis=1, keepdims=True)
            pdelta = float(np.max(np.abs(exp_probs-act_probs)))
            assert pdelta < protocol['max_abs_probability_tolerance']
            avg = expected.mean(axis=0); probs = np.exp(avg-avg.max()); probs /= probs.sum()
            result = list(csv.DictReader(io.StringIO(output), delimiter='\t'))[0]
            assert max(abs(float(result[f'p{i}'])-probs[i]) for i in range(5)) < protocol['max_abs_probability_tolerance']
            emit_samples = [int(r['end_sample']) for r in rows if r['emit']=='1']
            assert len(emit_samples)<2 or np.array_equal(np.diff(emit_samples), np.full(len(emit_samples)-1, 1600))
            outputs.append(output); traces.append(trace.read_bytes())
            checks.append({'input': name, 'chunk': chunk, 'frames': len(rows), 'max_logit_delta': delta, 'max_probability_delta': pdelta})
        assert all(t==traces[0] for t in traces), 'Callback partition changed frame outputs'
        assert all(o==outputs[0] for o in outputs), 'Callback partition changed file result'
        current = list(csv.DictReader(io.StringIO(traces[0].decode()), delimiter='\t'))
        if name=='input': full_trace = current
        if name=='prefix': assert full_trace[:100] == current, 'Future audio changed prefix outputs'
    if args.sanitizers:
        san = root/'cnn_stream_sanitized'
        cmd = [cc, '-std=c11', '-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer', *sources, '-lm', '-o', str(san)]
        commands.append(cmd); subprocess.run(cmd, check=True)
        with (root/'sanitizer-run.log').open('w') as log:
            subprocess.run([str(san), '--chunk', '997', *[str(root/(n+'.wav')) for n in ['input', 'short', 'silence']]], check=True, stdout=log, stderr=subprocess.STDOUT)
    summary = {'status': 'PASS', 'model_sha256': protocol['model_sha256'], 'checks': checks,
               'callback_partition_invariant': True, 'future_audio_not_used': True,
               'sanitizers': 'PASS' if args.sanitizers else 'not run',
               'compiler': subprocess.check_output([cc, '--version'], text=True),
               'binary_sha256': sha(binary), 'commands': commands}
    (root/'verification.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print('PASS: 16 numerical/state checks; sanitizers:', summary['sanitizers'])

if __name__ == '__main__':
    main()
