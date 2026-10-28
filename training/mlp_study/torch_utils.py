"""Training-fold normalization and INT8 storage reference for Tiny MLP."""
import copy
import numpy as np
import torch

def transform(x,c,scale):
    if c.get('transform')=='asinh40':
        x=x.copy();x[...,:40]=np.arcsinh(x[...,:40]/scale)
    return x

def normalize_fit(views,indices,c):
    clean=views[0,indices]
    scale=np.median(np.abs(clean[...,:40]),axis=(0,1)).astype(np.float32)
    scale=np.maximum(scale,np.float32(1e-6));mapped=transform(views,c,scale)
    train=mapped[0,indices];mean=train.mean((0,1),dtype=np.float64).astype(np.float32);std=train.std((0,1),dtype=np.float64).astype(np.float32)
    std[std<1e-6]=1
    return mapped,mean,std,scale

def probabilities(model,x,c):
    model.eval();result=[]
    with torch.no_grad():
        for part in np.array_split(x,max(1,(len(x)+31)//32)):
            v=model(torch.from_numpy(part))
            result.append(v.numpy())
    return np.concatenate(result)

def qmodel(model,c):
    result=copy.deepcopy(model)
    with torch.no_grad():
        for p in result.parameters():
            scale=max(float(p.abs().max()),1e-10)/127;p.copy_(p.div(scale).round().clamp(-127,127).mul(scale))
    return result.eval()
