"""Small causal temporal CNN: continuous 25 ms / 10 ms log-mel frames.

New experiment only. Uses the current v11 mel bank and WAV conversion, while
preserving temporal structure instead of summarizing independent one-second cuts.
"""
import pathlib,sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
SOURCE=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'training'))
from audio_utils import read_wav_mono
from keyword_detect import _MEL_FB
LABELS=['background','cough','glass_break','baby_cry','dog_bark']
CHANNELS=24;KERNEL=5;DILATIONS=[1,2,4,8,16]
HAMMING=(.54-.46*np.cos(2*np.pi*np.arange(400)/399)).astype(np.float32)
def logmel(pcm):
    x=np.asarray(pcm,dtype=np.float32)/32768.0
    if len(x)==0:raise ValueError('Empty audio')
    emphasized=np.empty_like(x);emphasized[0]=x[0];emphasized[1:]=x[1:]-np.float32(.97)*x[:-1]
    if len(emphasized)<400:emphasized=np.pad(emphasized,(0,400-len(emphasized)))
    frames=np.lib.stride_tricks.sliding_window_view(emphasized,400)[::160]
    power=(np.abs(np.fft.rfft(frames*HAMMING,n=512,axis=1))**2).astype(np.float32)
    return np.log(np.maximum(power@_MEL_FB.T,np.float32(1e-10))).T.astype(np.float32)
class CausalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs=nn.ModuleList([nn.Conv1d(26 if i==0 else CHANNELS,CHANNELS,KERNEL,dilation=d) for i,d in enumerate(DILATIONS)])
        self.dropout=nn.Dropout(.15)
        self.head=nn.Conv1d(CHANNELS,5,1)
    def frame_logits(self,x):
        for conv,d in zip(self.convs,DILATIONS):
            x=self.dropout(F.relu(conv(F.pad(x,(d*(KERNEL-1),0)))))
        return self.head(x)
    def forward(self,x):return self.frame_logits(x).mean(dim=2)
def layers(model):return [*model.convs,model.head]
def quantized_copy(model):
    copy=CausalCNN();copy.load_state_dict(model.state_dict())
    with torch.no_grad():
        for layer in layers(copy):
            for values in [layer.weight,layer.bias]:
                scale=max(float(values.abs().max()),1e-10)/127
                values.copy_(torch.clamp(torch.round(values/scale),-127,127)*scale)
    return copy.eval()
def export_npz(path,model,mean,std,**extra):
    values={'MEAN':mean,'STD':std,'LABELS':np.asarray(LABELS),'DILATIONS':np.asarray(DILATIONS),
            'FRAME_LEN':np.asarray(400),'HOP_LEN':np.asarray(160),'NFFT':np.asarray(512),
            'MEL_BANK':_MEL_FB,'HAMMING':HAMMING,**extra}
    for i,layer in enumerate(layers(model)):
        for prefix,tensor in [('W',layer.weight),('B',layer.bias)]:
            array=tensor.detach().cpu().numpy().astype(np.float32)
            scale=np.float32(max(float(np.abs(array).max()),1e-10)/127)
            values[f'{prefix}{i}']=array
            values[f'Q{prefix}{i}']=np.clip(np.round(array/scale),-127,127).astype(np.int8)
            values[f'{prefix}S{i}']=scale
    np.savez_compressed(path,**values)
def load_model(path,quantized=False):
    model=CausalCNN()
    with np.load(path,allow_pickle=False) as values:
        assert values['LABELS'].tolist()==LABELS
        mean=values['MEAN'].copy();std=values['STD'].copy()
        with torch.no_grad():
            for i,layer in enumerate(layers(model)):
                for prefix,tensor in [('W',layer.weight),('B',layer.bias)]:
                    array=values[f'Q{prefix}{i}'].astype(np.float32)*values[f'{prefix}S{i}'] if quantized else values[f'{prefix}{i}']
                    tensor.copy_(torch.from_numpy(array))
    return model.eval(),mean,std
def metrics(truth,pred):
    matrix=np.zeros((5,5),dtype=int)
    for t,p in zip(truth,pred):matrix[int(t),int(p)]+=1
    recall={k:float(matrix[i,i]/sum(matrix[i])) if sum(matrix[i]) else None for i,k in enumerate(LABELS)}
    return {'correct':int(np.trace(matrix)),'total':int(matrix.sum()),'accuracy':float(np.trace(matrix)/matrix.sum()),
            'balanced_accuracy_present_classes':float(np.mean([v for v in recall.values() if v is not None])),
            'per_class_recall':recall,'confusion_matrix':matrix.tolist()}
