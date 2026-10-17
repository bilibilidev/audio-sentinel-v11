"""Export candidate-only INT8 weight storage; C arithmetic remains float."""
import argparse,hashlib,json,pathlib
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('output');args=parser.parse_args()
out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
def floats(a):return ','.join(format(float(v),'.9e')+'f' for v in np.asarray(a).reshape(-1))
def ints(a):return ','.join(str(int(v)) for v in np.asarray(a).reshape(-1))
with np.load(args.model,allow_pickle=False) as model:
    pieces=['#ifndef CNN_WEIGHTS_H\n#define CNN_WEIGHTS_H\n#include <stdint.h>\n',
      'typedef struct { int in_channels,out_channels,kernel,dilation,history,offset; const int8_t *w,*b; float ws,bs; } cnn_layer_t;\n']
    for key,cname in [('MEAN','cnn_mean'),('STD','cnn_std'),('MEL_BANK','cnn_mel'),('HAMMING','cnn_hamming')]:
        pieces.append(f'static const float {cname}[]={{'+floats(model[key])+'};\n')
    layers=[];offset=0;parameter_count=0
    for i in range(6):
        w=model[f'QW{i}'];b=model[f'QB{i}'];oc,ic,k=w.shape
        d=int(model['DILATIONS'][i]) if i<5 else 1;history=d*(k-1)+1
        pieces.append(f'static const int8_t cnn_w{i}[]={{'+ints(w)+'};\n')
        pieces.append(f'static const int8_t cnn_b{i}[]={{'+ints(b)+'};\n')
        layers.append('{'+f'{ic},{oc},{k},{d},{history},{offset},cnn_w{i},cnn_b{i},'+floats([model[f'WS{i}'],model[f'BS{i}']])+'}')
        offset+=ic*history;parameter_count+=w.size+b.size
    pieces.append('static const cnn_layer_t cnn_layers[]={'+','.join(layers)+'};\n')
    pieces.append(f'#define CNN_HISTORY_SIZE {offset}\n#define CNN_PARAMETER_COUNT {parameter_count}\n#endif\n')
    (out/'cnn_weights.h').write_text(''.join(pieces),encoding='utf-8')
metadata={'model_sha256':hashlib.sha256(pathlib.Path(args.model).read_bytes()).hexdigest(),
          'header_sha256':hashlib.sha256((out/'cnn_weights.h').read_bytes()).hexdigest(),'parameters':parameter_count,
          'history_floats':offset,'history_bytes':offset*4,'quantization':'Per-tensor INT8 weights and biases; float normalization/accumulation/FFT/softmax; not a fully integer kernel'}
(out/'export.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata))
