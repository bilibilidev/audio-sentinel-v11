"""Evaluate a fixed model on a labelled manifest; no training or label changes.

The C backend needs NumPy and a previously built binary, not PyTorch. Audio is
verified by SHA256 then converted with the unchanged v11 WAV preprocessing.
"""
import argparse
import csv
import hashlib
import io
import json
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training'))
from audio_utils import read_wav_mono, write_wav_mono

LABELS = ['background', 'cough', 'glass_break', 'baby_cry', 'dog_bark']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=pathlib.Path, required=True)
    parser.add_argument('--audio-root', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--backend', choices=['c', 'python-float', 'python-int8'], default='c')
    parser.add_argument('--binary', type=pathlib.Path, default=REPO / 'out/cnn-simulator/cnn_stream')
    parser.add_argument('--model', type=pathlib.Path, default=REPO / 'model/cnn-v1/final_all_esc50.npz')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new empty output directory to preserve earlier results')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    records = manifest['records']
    if not records or len({r['id'] for r in records}) != len(records):
        parser.error('Manifest must contain nonempty records with unique IDs')
    args.output.mkdir(parents=True, exist_ok=True)
    paths, pcms = [], []
    for index, record in enumerate(records):
        path = (args.audio_root / record['audio_path']).resolve()
        if not path.is_relative_to(args.audio_root.resolve()):
            raise ValueError('Audio path escapes the supplied root')
        if record['reference_label'] not in LABELS or sha(path) != record['sha256']:
            raise ValueError(f"Invalid label or audio hash: {record['id']}")
        pcm = read_wav_mono(path, 16000)
        if not len(pcm):
            raise ValueError(f"Empty audio: {record['id']}")
        dest = (args.output / 'inputs' / f'{index:04d}.wav').resolve()
        write_wav_mono(dest, pcm, 16000)
        paths.append(dest)
        pcms.append(pcm)
    if args.backend == 'c':
        command = [str(args.binary.resolve()), *map(str, paths)]
        output = subprocess.check_output(command, text=True)
        (args.output / 'predictions.tsv').write_text(output, encoding='utf-8')
        raw = list(csv.DictReader(io.StringIO(output), delimiter='\t'))
        if len(raw) != len(records) or any(r['file'] != str(p) for r, p in zip(raw, paths)):
            raise ValueError('C output does not match input records')
        probs = np.asarray([[float(r[f'p{i}']) for i in range(5)] for r in raw])
        provenance = {'binary_sha256': sha(args.binary), 'command': command}
    else:
        import torch
        from causal_cnn import load_model, logmel
        torch.set_num_threads(4)
        model, mean, std = load_model(args.model, quantized=args.backend == 'python-int8')
        with torch.no_grad():
            probs = np.stack([torch.softmax(model(torch.from_numpy((logmel(p)[None] - mean) / std)), 1)[0].numpy() for p in pcms])
        provenance = {'model_sha256': sha(args.model)}
    if not np.isfinite(probs).all() or np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError('Invalid model probabilities')
    predictions = probs.argmax(axis=1)
    matrix = np.zeros((5, 5), dtype=int)
    results = []
    for record, pred, values in zip(records, predictions, probs):
        truth = LABELS.index(record['reference_label'])
        matrix[truth, pred] += 1
        results.append({'id': record['id'], 'reference_label': LABELS[truth],
                        'prediction': LABELS[pred], 'correct': bool(pred == truth),
                        'confidence': float(values[pred]), 'probabilities': values.tolist()})
    result = {'backend': args.backend, 'labels': LABELS, 'correct': int(np.trace(matrix)),
              'total': len(records), 'accuracy': float(np.trace(matrix) / len(records)),
              'confusion_matrix': matrix.tolist(), 'records': results,
              'manifest_sha256': sha(args.manifest), **provenance,
              'scope': 'Whole-recording mean-logit classification; not live-event accuracy'}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"{result['correct']}/{result['total']} = {result['accuracy']:.6%}")

if __name__ == '__main__':
    main()
