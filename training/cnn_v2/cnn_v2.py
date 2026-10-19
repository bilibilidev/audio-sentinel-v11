"""Causal CNN v2 feature/architecture definitions shared by controlled experiments."""
import pathlib
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training/cnn'))
from causal_cnn import LABELS, logmel, read_wav_mono, metrics, HAMMING
from audio_utils import resample_linear
from keyword_detect import _MEL_FB

DCT = (np.cos(np.pi/26*(np.arange(26)[None,:]+.5)*np.arange(13)[:,None])*np.sqrt(2/26)).astype(np.float32)
DCT[0] /= np.sqrt(2)
CONFIGS = [
    dict(name='full', feature='logmel', channels=24, dilations=[1,2,4,8,16], augmentation='phone', pooling='mixed_lme'),
    dict(name='no_augmentation', augmentation='none'),
    dict(name='no_phone_augmentation', augmentation='legacy'),
    dict(name='time_only', feature='time'),
    dict(name='mfcc_only', feature='mfcc'),
    dict(name='hybrid_features', feature='hybrid'),
    dict(name='channels_16', channels=16),
    dict(name='channels_32', channels=32),
    dict(name='dilation_flat', dilations=[1,1,1,1,1]),
    dict(name='dilation_medium', dilations=[1,2,4,8,8]),
    dict(name='mean_pooling', pooling='mean'),
]
CONFIGS = [{**CONFIGS[0], **c} for c in CONFIGS]

def all_features(pcm):
    pcm = np.asarray(pcm, dtype=np.int16)
    mel = logmel(pcm)
    raw = pcm.astype(np.float32)/32768
    if len(raw)<400: raw=np.pad(raw,(0,400-len(raw)))
    frames=np.lib.stride_tricks.sliding_window_view(raw,400)[::160]
    energy=np.mean(frames*frames,axis=1)
    delta=np.r_[np.float32(0),np.abs(np.diff(energy))]
    time=np.stack([np.log(np.maximum(energy,1e-10)),
                   np.mean(np.signbit(frames[:,1:])!=np.signbit(frames[:,:-1]),axis=1),
                   np.log(np.maximum(np.mean(np.abs(frames),axis=1),1e-10)),
                   np.log(np.maximum(np.max(np.abs(frames),axis=1),1e-10)),
                   np.log(np.maximum(delta,1e-10))]).astype(np.float32)
    return np.concatenate([mel,time],axis=0)

def select_features(full, kind):
    if kind=='logmel':return full[...,:26,:]
    if kind=='time':return full[...,26:,:]
    if kind=='hybrid':return full
    if kind=='mfcc':return np.matmul(DCT,full[...,:26,:])
    raise ValueError(kind)

def gain_vector(kind):
    mel=np.ones(26,dtype=np.float32)
    time=np.asarray([1,0,.5,.5,1],dtype=np.float32)
    if kind=='logmel':return mel
    if kind=='time':return time
    if kind=='hybrid':return np.r_[mel,time]
    if kind=='mfcc':return DCT@mel
    raise ValueError(kind)

class CNNV2(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=dict(config)
        inputs={'logmel':26,'time':5,'mfcc':13,'hybrid':31}[config['feature']]
        channels=config['channels']
        self.convs=nn.ModuleList([nn.Conv1d(inputs if i==0 else channels,channels,5,dilation=d) for i,d in enumerate(config['dilations'])])
        self.dropout=nn.Dropout(.15);self.head=nn.Conv1d(channels,5,1)
    def frame_logits(self,x):
        for layer,d in zip(self.convs,self.config['dilations']):x=self.dropout(F.relu(layer(F.pad(x,(4*d,0)))))
        return self.head(x)
    def forward(self,x):
        logits=self.frame_logits(x);mean=logits.mean(dim=2)
        if self.config['pooling']=='mean':return mean
        return .5*mean+.5*(torch.logsumexp(logits,dim=2)-np.log(logits.shape[2]))

def export(path,model,mean,std,**metadata):
    import json
    values={'MEAN':mean,'STD':std,'LABELS':np.asarray(LABELS),'CONFIG':np.asarray(json.dumps(model.config,sort_keys=True)),
            'MEL_BANK':_MEL_FB,'HAMMING':HAMMING,'DCT':DCT,'DILATIONS':np.asarray(model.config['dilations']),**metadata}
    for i,layer in enumerate([*model.convs,model.head]):
        for prefix,tensor in [('W',layer.weight),('B',layer.bias)]:
            array=tensor.detach().cpu().numpy().astype(np.float32);scale=np.float32(max(float(abs(array).max()),1e-10)/127)
            values[f'{prefix}{i}']=array;values[f'Q{prefix}{i}']=np.clip(np.round(array/scale),-127,127).astype(np.int8);values[f'{prefix}S{i}']=scale
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**values)

def load(path,quantized=False):
    import json
    with np.load(path,allow_pickle=False) as z:
        model=CNNV2(json.loads(str(z['CONFIG'])))
        assert z['LABELS'].tolist()==LABELS
        with torch.no_grad():
            for i,layer in enumerate([*model.convs,model.head]):
                for prefix,tensor in [('W',layer.weight),('B',layer.bias)]:
                    value=z[f'Q{prefix}{i}'].astype(np.float32)*z[f'{prefix}S{i}'] if quantized else z[f'{prefix}{i}']
                    tensor.copy_(torch.from_numpy(value.copy()))
        return model.eval(),z['MEAN'].copy(),z['STD'].copy()

def predict(model,features,mean,std):
    model.eval()
    with torch.no_grad():
        if isinstance(features,list):return np.stack([torch.softmax(model(torch.from_numpy((x[None]-mean)/std)),dim=1)[0].numpy() for x in features])
        return np.concatenate([torch.softmax(model(torch.from_numpy((x-mean)/std)),dim=1).numpy() for x in np.array_split(features,max(1,(len(features)+31)//32))])
