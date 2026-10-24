"""Separate C frontend error from neural-kernel error for an exported CNN."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,hashlib,json,pathlib,shutil,subprocess,sys
import numpy as np
import torch
REPO=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'training/cnn_v2'))
from cnn_v2 import load,all_features,select_features,predict,read_wav_mono
p=argparse.ArgumentParser(description=__doc__)
for name in ('source','model','wav','output'):p.add_argument('--'+name,type=pathlib.Path,required=True)
p.add_argument('--compiler',required=True);p.add_argument('--cflag',action='append',default=[]);a=p.parse_args()
out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
for name in ('cnn_stream_v2.c','cnn_stream_v2.h','cnn_weights_v2.h','cnn_model_config_v2.h'):shutil.copyfile(a.source/name,out/name)
h=out/'cnn_stream_v2.h';s=h.read_text(encoding='utf-8');anchor='  float previous_raw, previous_energy;';assert s.count(anchor)==1
h.write_text(s.replace(anchor,anchor+'\n  float raw_features[CNNV2_INPUT_DIM];'),encoding='utf-8',newline='\n')
c=out/'cnn_stream_v2.c';s=c.read_text(encoding='utf-8');anchor='  for(int i=0;i<CNNV2_INPUT_DIM;i++)a[i]=(a[i]-cnnv2_mean[i])/cnnv2_std[i];';assert s.count(anchor)==1
c.write_text(s.replace(anchor,'  memcpy(s->raw_features,a,CNNV2_INPUT_DIM*sizeof(float));\n'+anchor),encoding='utf-8',newline='\n')
main=out/'dump.c';main.write_text('''#include "cnn_stream_v2.h"
#include "wav_reader.h"
#include <stdio.h>
#include <stdlib.h>
static void trace(const cnn_v2_state_t *s,void *ctx){(void)ctx;
 for(int i=0;i<CNNV2_INPUT_DIM;i++)printf(i?" %.9g":"%.9g",s->raw_features[i]);
 for(int i=0;i<5;i++)printf(" %.9g",s->logits[i]);puts("");}
int main(int argc,char **argv){if(argc!=2)return 2;uint32_t bytes;FILE *f=open_wav_16k_mono(argv[1],&bytes);if(!f)return 1;
 cnn_v2_state_t *s=calloc(1,sizeof(*s));int16_t pcm[160];
 while(bytes){size_t n=bytes/2;if(n>160)n=160;if(fread(pcm,2,n,f)!=n)return 1;bytes-=(uint32_t)(n*2);cnn_v2_feed(s,pcm,n,trace,0);}fclose(f);
 cnn_v2_finish(s,trace,0);float p[5];cnn_v2_result(s,p);for(int i=0;i<5;i++)fprintf(stderr,i?" %.9g":"%.9g",p[i]);fputs("\\n",stderr);free(s);return 0;}
''',encoding='utf-8')
binary=out/'dump.exe';app=REPO/'app/audiodetect';cmd=[a.compiler,*a.cflag,'-std=c11','-O2','-I',str(out),'-I',str(app),str(main),str(c),str(app/'wav_reader.c'),'-lm','-o',str(binary)]
with (out/'build.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
r=subprocess.run([str(binary),str(a.wav.resolve())],capture_output=True,text=True,timeout=30,check=True)
(out/'trace.txt').write_text(r.stdout,encoding='utf-8');actual=np.asarray([[float(x) for x in row.split()] for row in r.stdout.splitlines()]);cprob=np.fromstring(r.stderr,sep=' ')
torch.set_num_threads(2);network,mean,std=load(a.model,quantized=True)
x=select_features(all_features(read_wav_mono(a.wav)),network.config['feature']);dims=x.shape[0]
native=actual[:,:dims].T.astype(np.float32);clogits=actual[:,dims:]
with torch.no_grad():expected=network.frame_logits(torch.from_numpy((native[None]-mean)/std))[0].T.numpy()
native_prob=predict(network,[native],mean,std)[0];raw_prob=predict(network,[x],mean,std)[0]
delta=np.abs(native-x);where=np.unravel_index(delta.argmax(),delta.shape)
summary=dict(model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),wav_sha256=hashlib.sha256(a.wav.read_bytes()).hexdigest(),frames=len(actual),
    max_feature_delta=float(delta.max()),feature_delta_max_location=[int(v) for v in where],per_feature_max_delta=delta.max(1).tolist(),
    max_neural_logit_delta_using_c_features=float(np.max(np.abs(expected-clogits))),
    probability_delta_c_vs_python_same_c_features=float(np.max(np.abs(cprob-native_prob))),
    probability_delta_c_vs_python_original_features=float(np.max(np.abs(cprob-raw_prob))),
    c_prediction=int(cprob.argmax()),python_prediction=int(raw_prob.argmax()),c_probabilities=cprob.tolist(),python_probabilities=raw_prob.tolist())
assert summary['max_neural_logit_delta_using_c_features']<.0003 and summary['probability_delta_c_vs_python_same_c_features']<.0001
(out/'diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['per_feature_max_delta','c_probabilities','python_probabilities']}))
