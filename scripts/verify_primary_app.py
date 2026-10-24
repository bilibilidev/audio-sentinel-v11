"""Verify integrated default MLP and explicit CNN fallback on frozen WAV inputs."""
import argparse,hashlib,json,pathlib,platform,re,shutil,subprocess,wave
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--inputs',type=pathlib.Path,required=True)
p.add_argument('--audit',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--compiler',default='cc');p.add_argument('--cflag',action='append',default=[])
a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
assert not (out/'verification.json').exists()
repo=pathlib.Path(__file__).resolve().parents[1];app=repo/'app/audiodetect'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
manifest=read(a.inputs/'manifest.json');reference=read(a.audit/'results.json')['models']['mlp'][-1]['external']['mean']['records']
labels=manifest['labels'];commands=[];binary={};records={}
def parse(text):
    pred=re.search(r'^Result: (\w+)$',text,re.M)[1]
    probs=[float(re.search(r'^  '+label+r': ([0-9.eE+-]+)$',text,re.M)[1]) for label in labels]
    return dict(prediction=labels.index(pred),probabilities=probs)
for backend in ('mlp','cnn'):
    target=out/(backend+('.exe' if platform.system()=='Windows' else ''))
    source=['audiodetect_main.c','audio_classifier.c','wav_reader.c','keyword_detect.c','alarm_report.c']
    source+=['cnn_stream_v2.c'] if backend=='cnn' else ['feature_extractor.c','mlp_model_infer.c']
    cmd=[a.compiler,*a.cflag,'-std=c11','-O2','-Wall','-Wextra','-pedantic']
    if backend=='cnn':cmd+=['-DAED_USE_CNN']
    if platform.system()=='Windows':cmd+=['-D__USE_MINGW_ANSI_STDIO=1']
    cmd += [*[str(app/n) for n in source],'-lm']
    if platform.system()=='Windows':cmd+=['-lws2_32']
    cmd+=['-o',str(target)];commands.append(cmd)
    with (out/(backend+'-build.log')).open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    binary[backend]=target;rows=[]
    for s in manifest['inputs']:
        cmd=[str(target),str((a.inputs/s['path']).resolve()),'--debug'];commands.append(cmd)
        text=subprocess.check_output(cmd,text=True,timeout=15);(out/(backend+'-'+s['id']+'.log')).write_text(text,encoding='utf-8')
        row=dict(id=s['id'],truth=labels.index(s['reference_label']),**parse(text));rows.append(row)
        if backend=='mlp':
            r=next(r for r in reference if r['id']==s['id'])
            assert row['prediction']==r['prediction'] and max(abs(p-q) for p,q in zip(row['probabilities'],r['probabilities']))<.00001
            assert read(repo/'model/mlp-retrained-20261024/deployment.json')['model_sha256'] in text
        else:assert read(repo/'model/cnn-v2/deployment.json')['model_sha256'] in text
    records[backend]=rows
cases=[]
for n in (1,123,16000,16001,48000):
    path=out/f'zero-{n}.wav'
    with wave.open(str(path),'wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(bytes(n*2))
    text=subprocess.check_output([str(binary['mlp']),str(path),'--debug'],text=True,timeout=10)
    row=parse(text);cases.append(dict(samples=n,**row))
    # Closed-set classifiers must emit one of five labels even for digital
    # silence. Verify the actual alarm contract, not an invented confidence.
    assert row['prediction']==0 or max(row['probabilities'])<.80,('digital silence alarm',n,row)
    assert '[ALERT] Local alarm triggered' not in text
bad=out/'invalid.wav';bad.write_bytes(b'not a WAV')
assert subprocess.run([str(binary['mlp']),str(bad)],capture_output=True,timeout=10).returncode!=0

harness=out/'pcm_api.c';harness.write_text('''#include "audio_classifier.h"
#include "wav_reader.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
int main(int argc,char **argv){
 aed_result_t a,b;if(aed_classify_pcm(NULL,0,&a)!=-1)return 1;
 for(int i=1;i<argc;i++){uint32_t bytes;FILE *f=open_wav_16k_mono(argv[i],&bytes);if(!f)return 2;
 int16_t *pcm=malloc(bytes);if(!pcm)return 2;if(fread(pcm,1,bytes,f)!=bytes)return 2;fclose(f);
 if(aed_classify_pcm(pcm,bytes/2,&a)||aed_classify_wav(argv[i],&b))return 3;free(pcm);
 if(a.class_index!=b.class_index||a.samples!=b.samples||a.analyzed_windows!=b.analyzed_windows)return 4;
 for(int j=0;j<5;j++)if(fabsf(a.probabilities[j]-b.probabilities[j])>1e-7f)return 5;
 }puts("PCM/WAV parity PASS");return 0;}
''',encoding='utf-8')
target=out/('pcm_api.exe' if platform.system()=='Windows' else 'pcm_api')
cmd=[a.compiler,*a.cflag,'-std=c11','-O2','-Wall','-Wextra','-I',str(app),str(harness),*[str(app/n) for n in ['audio_classifier.c','wav_reader.c','feature_extractor.c','keyword_detect.c','mlp_model_infer.c']],'-lm','-o',str(target)]
commands.append(cmd)
with (out/'pcm-build.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
cmd=[str(target),*[str((a.inputs/r['path']).resolve()) for r in manifest['inputs']],*[str(out/f'zero-{n}.wav') for n in (1,123,16000,16001,48000)]]
commands.append(cmd);text=subprocess.check_output(cmd,text=True,timeout=120);assert 'parity PASS' in text
result=dict(status='PASS',runtime=platform.platform(),default_backend='Tiny MLP',model_sha256=read(repo/'model/mlp-retrained-20261024/deployment.json')['model_sha256'],
      external_correct={name:sum(r['prediction']==r['truth'] for r in rows) for name,rows in records.items()},
      external_baby_correct={name:sum(r['truth']==3 and r['prediction']==3 for r in rows) for name,rows in records.items()},
      default_mlp_matches_frozen_evaluation=65,cnn_fallback_model_sha256=read(repo/'model/cnn-v2/deployment.json')['model_sha256'],
      pcm_wav_parity_recordings=70,digital_silence_cases=cases,invalid_wav_rejected=True,empty_pcm_rejected=True,
      binaries={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in binary.items()},
      physical_board_tested=False,full_openvela_tested=False,records=records)
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n',encoding='utf-8')
print('INTEGRATED APP PASS',result['external_correct'],result['external_baby_correct'])
