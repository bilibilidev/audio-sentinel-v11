"""Architecture counts only; no inference-time or power comparison is implied."""
import argparse,hashlib,json,pathlib,sys
import torch
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panns',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.panns.resolve()));from models import Cnn6
torch.set_num_threads(1);network=Cnn6(32000,1024,320,64,50,14000,527);network.eval()
learned=sum(v.numel() for v in network.parameters() if v.requires_grad);all_parameters=sum(v.numel() for v in network.parameters());mac=[0]
def count(layer,inputs,output):
    if isinstance(layer,torch.nn.Conv2d):mac[0]+=output.numel()*layer.weight.shape[1]*layer.weight.shape[2]*layer.weight.shape[3]
    if isinstance(layer,torch.nn.Linear):mac[0]+=output.numel()*layer.in_features
hooks=[m.register_forward_hook(count) for m in network.modules() if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear))]
with torch.no_grad():network(torch.zeros(1,160000))
for h in hooks:h.remove()
value=dict(input_seconds=5,mlp=dict(parameters=9413,network_MAC=5*(92*96+96*5),int8_weights_bias_bytes=9413,normalization_and_asinh_and_quantization_scale_bytes=2*92*4+40*4+4*4),
    panns=dict(learned_network_parameters=learned,all_parameter_elements_including_fixed_frontend=all_parameters,network_MAC_conv2d_linear=mac[0],linearSVM_extra_coefficients=5*512+5,
        checkpoint_bytes=(a.panns/'Cnn6_mAP=0.343.pth').stat().st_size,checkpoint_sha256=hashlib.sha256((a.panns/'Cnn6_mAP=0.343.pth').read_bytes()).hexdigest()),
    excludes='MLP MFCC/FFT/time/asinh and PANNs STFT/logmel, nonlinearities, pooling, softmax, normalization and SVM arithmetic are excluded from network MACs.',
    deployment='PANNs is not a Linux C/OpenVela/board deployment in this study; no same-board latency, power or energy measurement.')
a.output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print('RESOURCE COUNTS',value)
