"""Native C equal-weight ensemble of two already frozen Tiny MLP pipelines."""
import argparse,csv,hashlib,io,json,pathlib,re,shutil,subprocess,platform
p=argparse.ArgumentParser(description=__doc__)
for name in ('member-a','member-b','inputs','output'):p.add_argument('--'+name,type=pathlib.Path,required=True)
p.add_argument('--compiler',required=True);p.add_argument('--cflag',action='append',default=[]);a=p.parse_args()
REPO=pathlib.Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
assert not (out/'results.json').exists()
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
member_paths=[a.member_a.resolve(),a.member_b.resolve()];members=[read(p/'results.json')['models']['mlp']['models'] for p in member_paths]
assert [r['model'] for r in members[0]]==[r['model'] for r in members[1]]
inputs=a.inputs.resolve();esc=read(inputs/'esc-manifest.json')['inputs'];external=[]
if members[0][-1]['model']=='final_all_esc50':external=read(inputs/'manifest.json')['inputs']
def scores(rows):
    m=[[0]*5 for _ in range(5)]
    for r in rows:m[r['truth']][r['prediction']]+=1
    n=len(rows);correct=sum(m[i][i] for i in range(5))
    return dict(correct=correct,total=n,accuracy=correct/n,confusion_matrix=m,macro_recall=sum(m[i][i]/sum(m[i]) for i in range(5))/5)
MAIN='''#include "feature_extractor.h"
#include "wav_reader.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
void predict_window_1(float x[92],float p[5],float *attention);
void predict_window_2(float x[92],float p[5],float *attention);
typedef struct{float plain[5];double weighted[5],maximum,z;unsigned n;} acc_t;
static void add(acc_t *a,float p[5],float score,int attention){
 if(!attention){for(int i=0;i<5;i++)a->plain[i]+=p[i];a->n++;return;}
 if(a->n==0||score>a->maximum){double scale=a->n?exp(a->maximum-score):0;a->z*=scale;for(int i=0;i<5;i++)a->weighted[i]*=scale;a->maximum=score;}
 double w=exp(score-a->maximum);a->z+=w;for(int i=0;i<5;i++)a->weighted[i]+=w*p[i];a->n++;
}
static float value(acc_t *a,int i,int attention){return attention?(float)(a->weighted[i]/a->z):a->plain[i]/a->n;}
int main(int argc,char **argv){
 static const char *labels[]={"background","cough","glass_break","baby_cry","dog_bark"};
 const int attention_1=ATTENTION_ONE,attention_2=ATTENTION_TWO;
 puts("file\\tprediction\\tconfidence\\tp0\\tp1\\tp2\\tp3\\tp4");
 for(int arg=1;arg<argc;arg++){uint32_t bytes;FILE *f=open_wav_16k_mono(argv[arg],&bytes);if(!f)return 1;acc_t a={0},b={0};
 while(bytes){int16_t pcm[16000];float x[92],y[92],p[5],q[5],sa,sb;size_t n=bytes/2;if(n>16000)n=16000;memset(pcm,0,sizeof(pcm));if(fread(pcm,2,n,f)!=n)return 1;bytes-=(uint32_t)(n*2);
 extract_audio_features(pcm,16000,x);memcpy(y,x,sizeof(x));predict_window_1(x,p,&sa);predict_window_2(y,q,&sb);add(&a,p,sa,attention_1);add(&b,q,sb,attention_2);}fclose(f);
 if(!a.n||!b.n)return 1;float prob[5];int pred=0;for(int i=0;i<5;i++){prob[i]=.5f*(value(&a,i,attention_1)+value(&b,i,attention_2));if(prob[i]>prob[pred])pred=i;}
 printf("%s\\t%s\\t%.9g",argv[arg],labels[pred],prob[pred]);for(int i=0;i<5;i++)printf("\\t%.9g",prob[i]);puts("");}return 0;}
'''
labels=['background','cough','glass_break','baby_cry','dog_bark'];pooled=[];results=[];commands=[]
for i,entry in enumerate(members[0]):
    name=entry['model'];d=out/name;d.mkdir();modes=[];sources=[];parameters=[]
    for k,path in enumerate(member_paths):
        src=path/('mlp-'+name);target=d/('member'+str(k+1));target.mkdir()
        header=(src/'weights.h').read_text(encoding='utf-8');shutil.copyfile(src/'weights.h',target/'weights.h')
        code=(src/'harness.c').read_text(encoding='utf-8').split('int main(')[0]
        assert 'static void classify(' in code
        code=code.replace('static void classify(',f'void predict_window_{k+1}(')
        (target/'model.c').write_text(code,encoding='utf-8',newline='\n');sources.append(str(target/'model.c'))
        mode=int('#define OPT_ATTENTION' in header);modes.append(mode)
        hidden=int(re.search(r'#define HIDDEN_DIM (\d+)',header)[1]);parameters.append((92+1)*hidden+(hidden+1)*5+mode*hidden)
    (d/'main.c').write_text(MAIN.replace('ATTENTION_ONE',str(modes[0])).replace('ATTENTION_TWO',str(modes[1])),encoding='utf-8',newline='\n')
    binary=d/('ensemble.exe' if platform.system()=='Windows' else 'ensemble');app=REPO/'app/audiodetect'
    cmd=[a.compiler,*a.cflag,'-std=c11','-O2','-Wall','-Wextra','-I',str(app),str(d/'main.c'),*sources,*[str(app/n) for n in ['feature_extractor.c','wav_reader.c','keyword_detect.c']],'-lm','-o',str(binary)]
    commands.append(cmd)
    with (d/'build.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    result=dict(model=name,member_parameters=parameters,total_parameters=sum(parameters),binary_sha256=sha(binary))
    cohorts=[]
    if name.startswith('fold_'):cohorts.append(('official',[r for r in esc if r['fold']==int(name[-1])]))
    if external:cohorts.append(('external',external))
    for cohort,rows in cohorts:
        cmd=[str(binary),*[r['path'] for r in rows]];commands.append(cmd)
        text=subprocess.check_output(cmd,cwd=inputs,text=True,timeout=120);(d/(cohort+'.tsv')).write_text(text,encoding='utf-8')
        actual=list(csv.DictReader(io.StringIO(text),delimiter='\t'));assert len(actual)==len(rows)
        refs=[{r['id']:r for r in member[i][cohort]['records']} for member in members]
        records=[];delta=0.
        for s,r in zip(rows,actual):
            p=[float(r['p'+str(k)]) for k in range(5)];expected=[.5*(refs[0][s['id']]['probabilities'][k]+refs[1][s['id']]['probabilities'][k]) for k in range(5)]
            delta=max(delta,max(abs(x-y) for x,y in zip(p,expected)));assert delta<.00001
            pred=labels.index(r['prediction']);assert pred==max(range(5),key=p.__getitem__)
            truth=s['truth'] if cohort=='official' else labels.index(s['reference_label'])
            records.append(dict(id=s['id'],filename=s.get('filename'),truth=truth,prediction=pred,probabilities=p,input_sha256=s['sha256']))
        result[cohort]=dict(**scores(records),records=records,max_probability_delta_vs_member_average=delta)
        if cohort=='official':pooled+=records
    results.append(result);print('C ENSEMBLE',name,{cohort:result[cohort]['correct'] for cohort,_ in cohorts},flush=True)
summary=dict(status='PASS',type='Two Tiny MLPs, fixed equal probability average; not a single model',runtime=platform.platform(),official_nested_cv=scores(pooled),models=results,
    member_results_sha256=[sha(path/'results.json') for path in member_paths],blend_weights=[.5,.5],old_default_unchanged=True,board_tested=False)
if external:summary['external_final']=results[-1]['external']
(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n',encoding='utf-8')
print('TWO-MLP C ACCEPTANCE',summary['official_nested_cv'],flush=True)
