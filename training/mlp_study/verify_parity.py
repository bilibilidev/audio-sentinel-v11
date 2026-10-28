"""Compare actual Linux C predictions against NumPy using dumped Linux C features."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,pathlib,sys
import numpy as np
REPO=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'training'))
from numerics import quantize_dequantize,forward
p=argparse.ArgumentParser(description=__doc__)
for n in ('linux','inputs','study','output'):p.add_argument('--'+n,type=pathlib.Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text(encoding='utf-8'))
rows=read(a.inputs/'esc-manifest.json')['inputs']+read(a.inputs/'manifest.json')['inputs'];dump={i:[] for i in range(len(rows))}
for line in (a.linux/'features.log').read_text().splitlines():
    if not line or line.startswith('STDERR:'):continue
    values=line.split();idx=int(values[0]);assert int(values[1])==len(dump[idx]);dump[idx].append([float(x) for x in values[2:]])
features={r['id']:np.asarray(dump[i],dtype=np.float32) for i,r in enumerate(rows)}
study=read(a.linux/'ablations.json');reports=[]
cohorts=[(v['variant'],v['models'],a.study/v['variant']) for v in study]
for name,models,base in cohorts:
    for m in models:
        path=base/m['model']/'model.npz'
        with np.load(path,allow_pickle=False) as z:
            c=json.loads(str(z['CONFIG']));indices=z['FEATURE_INDICES'] if 'FEATURE_INDICES' in z.files else np.arange(92)
            weights=quantize_dequantize([dict(W=z['W1'].copy(),B=z['B1'].copy()),dict(W=z['W2'].copy(),B=z['B2'].copy())]);mean=z['FEATURE_MEAN'].copy();std=z['FEATURE_STD'].copy();scale=z['TIME_ASINH_SCALE'].copy()
        maximum=0.;mismatch=0;n=0
        for group in ('official','external'):
            if group not in m:continue
            for record in m[group]['records']:
                x=features[record['id']].copy()
                if c['transform']=='asinh40':x[:,:40]=np.arcsinh(x[:,:40]/scale)
                x=x[:,indices];prob=forward((x-mean)/std,weights)[0].mean(0)
                maximum=max(maximum,float(abs(prob-record['probabilities']).max()));mismatch+=int(prob.argmax())!=record['prediction'];n+=1
        assert maximum<.0001 and mismatch==0,(name,m['model'],maximum,mismatch)
        reports.append(dict(variant=name,model=m['model'],recordings=n,max_probability_delta=maximum,prediction_mismatches=mismatch))
result=dict(status='PASS',linux_frontend='Fresh actual Linux C feature dump on390+65 inputs',reference='NumPy2 on Windows CPU, same Linux-dumped features; NOT Python running in Linux',
    cases=reports,max_probability_delta=max(v['max_probability_delta'] for v in reports),total_model_recording_pairs=sum(v['recordings'] for v in reports),
    feature_dump_sha256=hashlib.sha256((a.linux/'features.log').read_bytes()).hexdigest(),tolerance=.0001)
a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('LINUX C/NumPy SAME-FEATURE PARITY',result['total_model_recording_pairs'],result['max_probability_delta'])
