"""Export a frozen optimized Tiny MLP, including its training-fitted transform."""
import argparse,hashlib,json,pathlib,sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from c_arrays import int8_header,c_float_array
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('model',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
with np.load(a.model,allow_pickle=False) as z:
    c=json.loads(str(z['CONFIG']));assert c.get('loss')=='window' and 'ATT_W' not in z.files
    hidden=z['W1'].shape[1];parameters=sum(z[n].size for n in ('W1','B1','W2','B2'))
    text=int8_header(z,z['LABEL_NAMES'].tolist(),'MLP_OPTIMIZED_WEIGHTS_H',True)
    extra='#define MLP_TIME_TRANSFORM_ASINH40 '+str(int(c['transform']=='asinh40'))+'\n'
    if c['transform']=='asinh40':extra+=c_float_array('TIME_ASINH_SCALE',z['TIME_ASINH_SCALE'],'[40]')
    text=text.replace('#include <stdint.h>','#include <stdint.h>\n'+extra)
(a.output/'mlp_weights.h').write_text(text,encoding='utf-8',newline='\n')
digest=hashlib.sha256(a.model.read_bytes()).hexdigest()
(a.output/'mlp_model_config.h').write_text('#ifndef MLP_MODEL_CONFIG_H\n#define MLP_MODEL_CONFIG_H\n#define MLP_MODEL_SHA256 "'+digest+'"\n#define MLP_PARAMETER_COUNT '+str(parameters)+'\n#endif\n',encoding='utf-8',newline='\n')
print(json.dumps(dict(model_sha256=digest,architecture=[92,hidden,5],parameters=parameters,config=c),indent=2))
