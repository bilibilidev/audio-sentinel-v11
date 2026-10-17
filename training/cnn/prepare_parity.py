"""Prepare independent Torch references for C numerical/state checks.

Generated tones/noise are boundary-test inputs, never accuracy evidence.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
import torch
from causal_cnn import load_model, logmel, read_wav_mono
from audio_utils import write_wav_mono

REPO = pathlib.Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, default=REPO / 'out/cnn-parity')
    parser.add_argument('--wav', type=pathlib.Path, help='Optional local WAV instead of generated boundary-test signal')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = REPO / 'model/cnn-v1/final_all_esc50.npz'
    model, mean, std = load_model(model_path, quantized=True)
    torch.set_num_threads(4)
    if args.wav:
        pcm = read_wav_mono(args.wav, 16000)
    else:
        rng = np.random.default_rng(20261017)
        t = np.arange(32000) / 16000
        pcm = np.clip(8000 * np.sin(2 * np.pi * (180*t + 130*t*t)) + rng.normal(0, 1500, len(t)), -32768, 32767).astype(np.int16)
    if len(pcm) < 400 + 100 * 160:
        parser.error('Input needs at least 1.025 seconds for the prefix-causality check')
    for name, values in [('input', pcm), ('prefix', pcm[:400+99*160]),
                         ('short', pcm[:320]), ('silence', np.zeros(16000, dtype=np.int16))]:
        write_wav_mono(args.output / (name+'.wav'), values, 16000)
        with torch.no_grad():
            logits = model.frame_logits(torch.from_numpy((logmel(values)[None]-mean)/std))[0].T.numpy()
        np.savez_compressed(args.output / (name+'-reference.npz'), LOGITS=logits)
    header_lf = (REPO/'simulator/cnn/cnn_weights.h').read_bytes().replace(b'\r\n', b'\n')
    protocol = {'model_sha256': sha(model_path), 'header_sha256_lf': hashlib.sha256(header_lf).hexdigest(),
                'chunk_sizes': [1, 160, 997, 4096], 'max_abs_logit_tolerance': .01,
                'max_abs_probability_tolerance': .002,
                'inputs': {p.name: sha(p) for p in args.output.iterdir() if p.suffix in ['.wav', '.npz']},
                'scope': 'Numerical and streaming-state tests; generated audio is not an accuracy test'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    print('Prepared final-model parity fixtures:', args.output)

if __name__ == '__main__':
    main()
