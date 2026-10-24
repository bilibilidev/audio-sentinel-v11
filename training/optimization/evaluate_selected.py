"""Execute optimized selected models in native C; ESC-only mode never opens external audio."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import argparse,csv,hashlib,io,json,pathlib,platform,shutil,subprocess,sys
import numpy as np
import torch
REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'training'));sys.path.insert(0,str(REPO/'training/cnn_v2'))
from export_to_c import int8_header,c_float_array,c_int8_array,c_scalar,quantize_symmetric
from train_edge_mfcc import quantize_dequantize
from sweep_edge_models import forward
from audio_utils import read_wav_mono
from extract_features_mfcc import extract_features_mfcc_from_pcm
from cnn_v2 import load,predict,all_features,LABELS
from adapters import pcen_full,adapt_pcen,centered_full,adapt_centered,energy_full,adapt_energy,pool_full,adapt_temporal
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--inputs',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--compiler',required=True);p.add_argument('--cflag',action='append',default=[])
p.add_argument('--kind',choices=['mlp','cnn','both'],default='both');p.add_argument('--only-esc',action='store_true')
p.add_argument('--mlp-selection',default='mlp-selected');p.add_argument('--cnn-selection',default='cnn-selected')
p.add_argument('--cnn-probability-tolerance',type=float,default=.0001)
a=p.parse_args();base=a.root.resolve();inputs=a.inputs.resolve();out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
assert not (out/'results.json').exists(),'Preserve completed C acceptance'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
kinds=['mlp','cnn'] if a.kind=='both' else [a.kind]
selections={'mlp':a.mlp_selection,'cnn':a.cnn_selection}
if not a.only_esc:assert all((base/selections[kind]/'frozen.json').exists() for kind in ('mlp','cnn')),'Freeze both selected families before external evaluation'
selected={kind:read(base/selections[kind]/'frozen.json') for kind in kinds}
for kind in kinds:
    for item in selected[kind]['models']:assert sha(base/item['model_path'])==item['model_sha256']
def save(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
save('freeze-before-inputs.json',dict(selected_hashes={k:sha(base/selections[k]/'frozen.json') for k in kinds},external_opened=False,only_esc=a.only_esc))
esc=read(inputs/'esc-manifest.json')['inputs'];external=[]
if not a.only_esc:external=read(inputs/'manifest.json')['inputs']
source_rows=[*external,*esc]
for row in source_rows:assert sha(inputs/row['path'])==row['sha256']
commands=[];suffix='.exe' if platform.system()=='Windows' else ''
def build(cmd,log):
    commands.append(cmd)
    with log.open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
def score(rows):
    m=[[0]*5 for _ in range(5)]
    for row in rows:m[row['truth']][row['prediction']]+=1
    correct=sum(m[i][i] for i in range(5));recall={name:m[i][i]/sum(m[i]) for i,name in enumerate(LABELS)}
    return dict(correct=correct,total=len(rows),accuracy=correct/len(rows),macro_recall=sum(recall.values())/5,confusion_matrix=m,per_class_recall=recall)
native={};python_raw={}
if 'mlp' in kinds:
    app=REPO/'app/audiodetect';binary=out/('feature_dump'+suffix)
    build([a.compiler,*a.cflag,'-std=c11','-O2','-I',str(app),str(REPO/'scripts/legacy_mlp_features.c'),*[str(app/n) for n in ['feature_extractor.c','wav_reader.c','keyword_detect.c']],'-lm','-o',str(binary)],out/'feature-build.log')
    cmd=[str(binary),*[r['path'] for r in source_rows]];commands.append(cmd)
    text=subprocess.check_output(cmd,cwd=inputs,text=True,timeout=180);(out/'native-features.txt').write_text(text,encoding='utf-8')
    per={i:[] for i in range(len(source_rows))}
    for row in text.splitlines():
        values=row.split();index=int(values[0]);assert int(values[1])==len(per[index]);per[index].append([float(x) for x in values[2:]])
    for i,s in enumerate(source_rows):
        native[s['path']]=np.asarray(per[i],dtype=np.float32)
        pcm=read_wav_mono(inputs/s['path']);x=[]
        for offset in range(0,len(pcm),16000):
            chunk=np.zeros(16000,dtype=np.int16);v=pcm[offset:offset+16000];chunk[:len(v)]=v
            x.append(extract_features_mfcc_from_pcm(chunk,True))
        python_raw[s['path']]=np.stack(x)
cnn_raw={}
if 'cnn' in kinds:
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    for s in source_rows:cnn_raw[s['path']]=all_features(read_wav_mono(inputs/s['path']))
HARNESSS='''#include "weights.h"
#include "feature_extractor.h"
#include "wav_reader.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
static void classify(float x[92],float p[5],float *attention) {
 float n[92],h[HIDDEN_DIM],l[5];
 for(int i=0;i<92;i++){
#ifdef OPT_ASINH
  if(i<40)x[i]=asinhf(x[i]/TIME_ASINH_SCALE[i]);
#endif
  n[i]=(x[i]-FEATURE_MEAN[i])/(FEATURE_STD[i]==0?1:FEATURE_STD[i]);
 }
 for(int j=0;j<HIDDEN_DIM;j++){float v=(float)B1_INT8[j]*B1_SCALE;for(int i=0;i<92;i++)v+=n[i]*(float)W1_INT8[i][j]*W1_SCALE;h[j]=v>0?v:0;}
 *attention=0;
#ifdef OPT_ATTENTION
 for(int i=0;i<HIDDEN_DIM;i++)*attention+=h[i]*(float)ATT_W_INT8[i]*ATT_W_SCALE;
#endif
 for(int j=0;j<5;j++){float v=(float)B2_INT8[j]*B2_SCALE;for(int i=0;i<HIDDEN_DIM;i++)v+=h[i]*(float)W2_INT8[i][j]*W2_SCALE;l[j]=v;}
 float m=l[0],sum=0;for(int i=1;i<5;i++)if(l[i]>m)m=l[i];for(int i=0;i<5;i++){p[i]=expf(l[i]-m);sum+=p[i];}for(int i=0;i<5;i++)p[i]/=sum;
}
int main(int argc,char **argv){
 puts("file\\tprediction\\tconfidence\\tp0\\tp1\\tp2\\tp3\\tp4");
 for(int a=1;a<argc;a++){uint32_t bytes;FILE *f=open_wav_16k_mono(argv[a],&bytes);if(!f)return 1;float total[5]={0};unsigned count=0;double weighted[5]={0},maximum=-INFINITY,weight_sum=0;
 while(bytes){int16_t pcm[16000];float x[92],p[5],attention;size_t n=bytes/2;if(n>16000)n=16000;memset(pcm,0,sizeof(pcm));if(fread(pcm,2,n,f)!=n)return 1;bytes-=(uint32_t)(n*2);extract_audio_features(pcm,16000,x);classify(x,p,&attention);
#ifdef OPT_ATTENTION
 if(attention>maximum){double rescale=isfinite(maximum)?exp(maximum-attention):0;weight_sum*=rescale;for(int i=0;i<5;i++)weighted[i]*=rescale;maximum=attention;}
 double w=exp(attention-maximum);weight_sum+=w;for(int i=0;i<5;i++)weighted[i]+=w*p[i];
#else
 for(int i=0;i<5;i++)total[i]+=p[i];
#endif
 count++;}fclose(f);
 if(!count)return 1;int pred=0;for(int i=0;i<5;i++){
#ifdef OPT_ATTENTION
 total[i]=(float)(weighted[i]/weight_sum);
#else
 total[i]/=count;
#endif
 if(total[i]>total[pred])pred=i;}
 printf("%s\\t%s\\t%.9g",argv[a],LABEL_NAMES[pred],total[pred]);for(int i=0;i<5;i++)printf("\\t%.9g",total[i]);puts("");}return 0;}
'''
results={}
for kind in kinds:
    outer=[];models=[]
    for item in selected[kind]['models']:
        name=item['model']
        if a.only_esc and name=='final_all_esc50':continue
        d=out/(kind+'-'+name);d.mkdir();model_path=base/item['model_path'];c=item['config'];binary=d/('inference'+suffix)
        if kind=='mlp':
            with np.load(model_path,allow_pickle=False) as z:
                mean=z['FEATURE_MEAN'].copy();std=z['FEATURE_STD'].copy();scale=z['TIME_ASINH_SCALE'].copy()
                weights=[dict(W=z['W1'].copy(),B=z['B1'].copy()),dict(W=z['W2'].copy(),B=z['B2'].copy())]
                header=int8_header(z,LABELS,'OPT_WEIGHTS_H',True)
                attention_weights=z['ATT_W'].copy() if 'ATT_W' in z.files else None
            if c['transform']=='asinh40':header='#define OPT_ASINH\n'+c_float_array('TIME_ASINH_SCALE',scale,'[40]')+header
            if attention_weights is not None:
                aq,ascale=quantize_symmetric(attention_weights)
                header+='\n#define OPT_ATTENTION\n'+c_int8_array('ATT_W_INT8',aq,'[HIDDEN_DIM]')+c_scalar('ATT_W_SCALE',ascale)
            (d/'weights.h').write_text(header,encoding='utf-8',newline='\n');(d/'harness.c').write_text(HARNESSS,encoding='utf-8',newline='\n')
            app=REPO/'app/audiodetect'
            cmd=[a.compiler,*a.cflag,'-std=c11','-O2','-Wall','-Wextra','-I',str(d),'-I',str(app),str(d/'harness.c'),*[str(app/n) for n in ['feature_extractor.c','wav_reader.c','keyword_detect.c']],'-lm','-o',str(binary)]
            build(cmd,d/'build.log')
            def predict_mlp(x,mode):
                x=x.copy()
                if c['transform']=='asinh40':x[:,:40]=np.arcsinh(x[:,:40]/scale)
                prob,activations,_=forward((x-mean)/std,quantize_dequantize(weights) if mode=='int8' else weights)
                if attention_weights is None:return prob.mean(0)
                aw=attention_weights
                if mode=='int8':
                    aq,ascale=quantize_symmetric(aw);aw=aq.astype(np.float32)*np.float32(ascale)
                scores=activations[1]@aw;pw=np.exp(scores-scores.max());pw/=pw.sum()
                return (prob*pw[:,None]).sum(0)
        else:
            cmd=[sys.executable,str(REPO/'training/cnn_v2/export_c.py'),str(model_path),str(d)];commands.append(cmd)
            with (d/'export.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log)
            for n in ('cnn_stream_v2.c','cnn_stream_v2.h'):shutil.copyfile(REPO/'app/audiodetect'/n,d/n)
            if c.get('frontend')=='pcen':adapt_pcen(d)
            if c.get('frontend')=='spectral_center':adapt_centered(d,c['center_bins'])
            if c.get('frontend')=='energy_norm':adapt_energy(d)
            shutil.copyfile(REPO/'simulator/cnn_v2/cnn_cli.c',d/'harness.c')
            if c.get('temporal_pool'):adapt_temporal(d,c['temporal_pool'])
            build([a.compiler,*a.cflag,'-D__USE_MINGW_ANSI_STDIO=1','-std=c11','-O2','-Wall','-Wextra','-I',str(d),str(d/'harness.c'),str(d/'cnn_stream_v2.c'),'-lm','-o',str(binary)],d/'build.log')
            network,mean,std=load(model_path,quantized=True);float_network,_,_=load(model_path,quantized=False)
            from cnn_v2 import select_features
        entry=dict(model=name,config=c,model_sha256=item['model_sha256'],binary_sha256=sha(binary))
        cohorts=[]
        if name.startswith('fold_'):cohorts.append(('official',[r for r in esc if r['fold']==int(name[-1])]))
        if external:cohorts.append(('external',external))
        for cohort,rows in cohorts:
            cmd=[str(binary),*[r['path'] for r in rows]];commands.append(cmd);text=subprocess.check_output(cmd,cwd=inputs,text=True,timeout=120)
            (d/(cohort+'.tsv')).write_text(text,encoding='utf-8',newline='\n');actual=list(csv.DictReader(io.StringIO(text),delimiter='\t'));assert len(actual)==len(rows)
            records=[];native_delta=0.;raw_delta=0.;raw_mismatch=0;fp=[]
            for s,r in zip(rows,actual):
                assert r['file']==s['path'];prob=np.asarray([float(r['p'+str(i)]) for i in range(5)])
                if kind=='mlp':
                    expected=predict_mlp(native[s['path']],'int8');raw=predict_mlp(python_raw[s['path']],'int8');floatp=predict_mlp(python_raw[s['path']],'float')
                else:
                    full=pcen_full(cnn_raw[s['path']]) if c.get('frontend')=='pcen' else cnn_raw[s['path']]
                    if c.get('frontend')=='spectral_center':full=centered_full(full,c['center_bins'])
                    if c.get('frontend')=='energy_norm':full=energy_full(full)
                    if c.get('temporal_pool'):full=pool_full(full,c['temporal_pool'])
                    x=select_features(full,c['feature']);expected=predict(network,[x],mean,std)[0];raw=expected;floatp=predict(float_network,[x],mean,std)[0]
                delta=float(np.max(np.abs(prob-expected)));native_delta=max(delta,native_delta);raw_delta=max(raw_delta,float(np.max(np.abs(prob-raw))))
                tolerance=.0001 if kind=='mlp' else a.cnn_probability_tolerance
                assert delta<tolerance,(kind,name,s['path'],delta)
                pred=LABELS.index(r['prediction']);assert pred==int(prob.argmax());raw_mismatch+=pred!=int(raw.argmax())
                truth=s['truth'] if cohort=='official' else LABELS.index(s['reference_label'])
                record=dict(id=s['id'],filename=s.get('filename'),truth=truth,prediction=pred,probabilities=prob.tolist(),python_float_prediction=int(floatp.argmax()),input_sha256=s['sha256'])
                records.append(record);fp.append(dict(truth=truth,prediction=int(floatp.argmax())))
            entry[cohort]=dict(**score(records),records=records,python_float=score(fp),max_reference_probability_delta=native_delta,max_python_frontend_probability_delta=raw_delta,python_frontend_prediction_mismatches=int(raw_mismatch),
                reference_frontend='Actual C feature dump' if kind=='mlp' else 'Python DSP frontend; energy-delta feature has float reduction differences')
            if cohort=='official':outer+=records
        models.append(entry);print('C VERIFIED',kind,name,{cohort:entry[cohort]['correct'] for cohort,_ in cohorts},flush=True)
    results[kind]=dict(models=models,official_nested_cv=score(outer),python_float_nested_cv=score([dict(truth=r['truth'],prediction=r['python_float_prediction']) for r in outer]))
    if external:results[kind]['external_final']=models[-1]['external']
save('results.json',dict(status='PASS',runtime=platform.platform(),compiler=subprocess.check_output([a.compiler,'--version'],text=True),only_esc=a.only_esc,
    models=results,external_role='Previously used development65, not new blind data',default_app_unchanged=True,board_tested=False,full_openvela_tested=False,
    cnn_probability_tolerance=a.cnn_probability_tolerance,cnn_precision_diagnostic_sha256=sha(base/'cnn-precision/diagnostic.json') if (base/'cnn-precision/diagnostic.json').exists() else None))
save('commands.json',commands);print('SELECTED C ACCEPTANCE COMPLETE',flush=True)
