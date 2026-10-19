"""Prepare C/Python parity cases from a supplied WAV, without external tuning."""
import argparse,hashlib,json,pathlib,shutil,subprocess,sys,wave
import numpy as np
import torch
from cnn_v2 import all_features,select_features,load,predict,read_wav_mono,REPO
parser=argparse.ArgumentParser()
parser.add_argument('model',type=pathlib.Path);parser.add_argument('wav',type=pathlib.Path);parser.add_argument('output',type=pathlib.Path)
args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
assert not (out/'protocol.json').exists(),'Use a fresh validation directory'
subprocess.run([sys.executable,str(REPO/'training/cnn_v2/export_c.py'),str(args.model),str(out)],check=True)
for name in ['cnn_stream_v2.c','cnn_stream_v2.h']:shutil.copyfile(REPO/'app/audiodetect'/name,out/name)
shutil.copyfile(REPO/'simulator/cnn_v2/cnn_cli.c',out/'cnn_cli.c')
pcm=read_wav_mono(args.wav,16000);assert len(pcm)>=16240,'Supply at least 1.015 seconds of audio'
network,mean,std=load(args.model,quantized=True);torch.set_num_threads(2)
for name,audio in {'input':pcm,'prefix':pcm[:16240],'short':pcm[:123],'silence':np.zeros(16000,dtype=np.int16)}.items():
    with wave.open(str(out/(name+'.wav')),'wb') as w:
        w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(audio.astype('<i2').tobytes())
    x=select_features(all_features(audio),network.config['feature'])
    with torch.no_grad():frames=network.frame_logits(torch.from_numpy((x[None]-mean)/std))[0].T.numpy()
    np.savez_compressed(out/(name+'-reference.npz'),LOGITS=frames,PROB=predict(network,[x],mean,std)[0])
shutil.copyfile(REPO/'training/cnn_v2/verify_c_check.py',out/'verify.py')
inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.suffix in ['.c','.h','.npz','.wav','.py']}
(out/'protocol.json').write_text(json.dumps({'inputs':inputs,'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
    'chunk_sizes':[1,160,997,4096],'max_abs_logit_tolerance':.0003,'max_abs_probability_tolerance':.0001},indent=2),encoding='utf-8')
