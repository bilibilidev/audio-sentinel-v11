"""Export variable-feature/dilation CNN v2 to immutable C arrays and dimensions."""
import argparse,hashlib,json,pathlib
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('model',type=pathlib.Path);parser.add_argument('output',type=pathlib.Path);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
def floats(a):return ','.join(format(float(v),'.9e')+'f' for v in np.asarray(a).reshape(-1))
def ints(a):return ','.join(str(int(v)) for v in np.asarray(a).reshape(-1))
with np.load(args.model,allow_pickle=False) as z:
    config=json.loads(str(z['CONFIG']));pieces=['#ifndef AED_CNN_WEIGHTS_V2_H\n#define AED_CNN_WEIGHTS_V2_H\n#include <stdint.h>\n',
      'typedef struct { int in_channels,out_channels,kernel,dilation,history,offset; const int8_t *w,*b; float ws,bs; } cnnv2_layer_t;\n']
    for key,name in [('MEAN','mean'),('STD','std'),('MEL_BANK','mel'),('HAMMING','hamming'),('DCT','dct')]:pieces.append('static const float cnnv2_'+name+'[]={'+floats(z[key])+'};\n')
    rows=[];offset=0;count=0;n=len(config['dilations'])+1
    for i in range(n):
        w,b=z[f'QW{i}'],z[f'QB{i}'];oc,ic,k=w.shape;d=config['dilations'][i] if i<n-1 else 1;history=d*(k-1)+1
        assert max(oc,ic)<=32
        for prefix,array in [('w',w),('b',b)]:pieces.append(f'static const int8_t cnnv2_{prefix}{i}[]={{'+ints(array)+'};\n')
        rows.append('{'+f'{ic},{oc},{k},{d},{history},{offset},cnnv2_w{i},cnnv2_b{i},'+floats([z[f'WS{i}'],z[f'BS{i}']])+'}')
        offset+=ic*history;count+=w.size+b.size
    pieces.append('static const cnnv2_layer_t cnnv2_layers[]={'+','.join(rows)+'};\n#endif\n')
    (args.output/'cnn_weights_v2.h').write_text(''.join(pieces),encoding='utf-8')
    macros={'CNNV2_HISTORY_SIZE':offset,'CNNV2_LAYER_COUNT':n,'CNNV2_INPUT_DIM':int(z['QW0'].shape[1]),'CNNV2_FEATURE_KIND':{'logmel':0,'time':1,'mfcc':2,'hybrid':3}[config['feature']],
            'CNNV2_MIXED_LME':int(config['pooling']!='mean'),'CNNV2_PARAMETER_COUNT':count,'CNNV2_CONTEXT_FRAMES':1+4*sum(config['dilations'])}
    header='#ifndef AED_CNN_MODEL_CONFIG_V2_H\n#define AED_CNN_MODEL_CONFIG_V2_H\n'
    header+=''.join(f'#define {k} {v}\n' for k,v in macros.items())
    digest=hashlib.sha256(args.model.read_bytes()).hexdigest();header+=f'#define CNNV2_MODEL_SHA256 "{digest}"\n#endif\n'
    (args.output/'cnn_model_config_v2.h').write_text(header,encoding='utf-8')
receipt={'model_sha256':digest,'configuration':config,'macros':macros,'quantization':'INT8 weights/biases, float DSP and convolution, double stable aggregation'}
(args.output/'export.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
