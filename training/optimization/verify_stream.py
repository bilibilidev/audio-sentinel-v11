"""Verify private experimental CNN adapters at PCM chunk, prefix and EOF boundaries."""
import argparse,csv,hashlib,io,json,pathlib,shutil,subprocess,sys,wave
import numpy as np
import torch
REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'training/cnn_v2'))
from cnn_v2 import load,all_features,select_features,predict,read_wav_mono
from adapters import *
p=argparse.ArgumentParser(description=__doc__)
for n in ('model','output','wav'):p.add_argument('--'+n,type=pathlib.Path,required=True)
p.add_argument('--compiler',required=True);p.add_argument('--cflag',action='append',default=[]);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True);torch.set_num_threads(2)
network,mean,std=load(a.model,True);c=network.config
commands=[]
def call(cmd,**kw):
    commands.append([str(v) for v in cmd]);return subprocess.check_output(cmd,**kw)
call([sys.executable,str(REPO/'training/cnn_v2/export_c.py'),str(a.model.resolve()),str(out)])
for n in ('cnn_stream_v2.c','cnn_stream_v2.h'):shutil.copyfile(REPO/'app/audiodetect'/n,out/n)
shutil.copyfile(REPO/'simulator/cnn_v2/cnn_cli.c',out/'harness.c')
if c.get('frontend')=='energy_norm':adapt_energy(out)
if c.get('frontend')=='pcen':adapt_pcen(out)
if c.get('frontend')=='spectral_center':adapt_centered(out,c['center_bins'])
factor=c.get('temporal_pool',1)
if factor!=1:adapt_temporal(out,factor)
binary=out/'stream.exe'
cmd=[a.compiler,*a.cflag,'-D__USE_MINGW_ANSI_STDIO=1','-std=c11','-O2','-Wall','-Wextra','-I',str(out),str(out/'harness.c'),str(out/'cnn_stream_v2.c'),'-lm','-o',str(binary)]
commands.append(cmd)
with (out/'build.log').open('w') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
pcm=read_wav_mono(a.wav);rng=np.random.default_rng(20261024)
cases={'recording':pcm,'prefix':pcm[:16240],'one':pcm[:1],'399':pcm[:399],'400':pcm[:400],
       'partial_pool':pcm[:560],'partial_audio':pcm[:16037],'silence':np.zeros(17003,dtype=np.int16),
       'noise':rng.integers(-32768,32768,17003,dtype=np.int16)}
records=[];traces={}
for name,data in cases.items():
    path=out/(name+'.wav')
    with wave.open(str(path),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(data.tobytes())
    full=all_features(data)
    if c.get('frontend')=='energy_norm':full=energy_full(full)
    if c.get('frontend')=='pcen':full=pcen_full(full)
    if c.get('frontend')=='spectral_center':full=centered_full(full,c['center_bins'])
    if factor!=1:full=pool_full(full,factor)
    features=select_features(full,c['feature']);reference=predict(network,[features],mean,std)[0]
    with torch.no_grad():logits=network.frame_logits(torch.from_numpy((features[None]-mean)/std))[0].T.numpy()
    alpha=1-.98**factor;ema=np.empty_like(logits)
    for i,l in enumerate(logits):ema[i]=l if not i else (1-alpha)*ema[i-1]+alpha*l
    baseline=None;maxp=maxl=maxe=0
    for chunk in (1,7,160,511,4096,65536):
        trace=out/(name+'-'+str(chunk)+'.tsv')
        text=call([str(binary),'--chunk',str(chunk),'--trace',str(trace),str(path)],text=True,timeout=60)
        rows=list(csv.DictReader(io.StringIO(text),delimiter='\t'));assert len(rows)==1
        prob=np.asarray([float(rows[0]['p'+str(i)]) for i in range(5)])
        debug=np.genfromtxt(trace,delimiter='\t',names=True)
        matrix=np.atleast_1d(debug)
        l=np.stack([matrix['l'+str(i)] for i in range(5)],axis=1);e=np.stack([matrix['e'+str(i)] for i in range(5)],axis=1)
        assert l.shape==logits.shape,(name,l.shape,logits.shape)
        if baseline is None:baseline=(text,trace.read_bytes());traces[name]=l
        else:assert text==baseline[0] and trace.read_bytes()==baseline[1],(name,'chunk mismatch')
        maxp=max(maxp,float(abs(prob-reference).max()));maxl=max(maxl,float(abs(l-logits).max()));maxe=max(maxe,float(abs(e-ema).max()))
        assert maxp<.001 and maxl<.005 and maxe<.005,(name,maxp,maxl,maxe)
        assert int(prob.argmax())==int(reference.argmax())
        assert int(rows[0]['samples'])==len(data)
    records.append(dict(case=name,samples=len(data),neural_frames=len(logits),chunks=[1,7,160,511,4096,65536],
        max_probability_delta=maxp,max_frame_logit_delta=maxl,max_ema_delta=maxe,chunk_bit_identical=True))
assert np.array_equal(traces['recording'][:len(traces['prefix'])],traces['prefix']),'Causality failure'
result=dict(status='PASS',config=c,model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),cases=records,
    prefix_causality='100 observed DSP frames, exactly divisible by pool factor; byte-identical C logits',
    eof_note='Partial observed pool emitted only at EOF; no synthetic future frames except padding clips shorter than25ms',
    ema_alpha=alpha,output_interval_ms=10*factor*((10+factor-1)//factor),board_tested=False,sanitizers_run=False)
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n',encoding='utf-8')
print('STREAM ADAPTER PASS',result,flush=True)
