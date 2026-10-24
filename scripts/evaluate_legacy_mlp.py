"""Run the retained tiny MLP on the same hash-frozen external65 as CNN.

Preparation copies original sources without modification; execution works with
an existing GCC on Linux or Windows. No training and no weight export writes.
"""
import argparse,csv,hashlib,io,json,pathlib,platform,shutil,subprocess,tarfile,time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--prepare',type=pathlib.Path,help='Existing external evaluation directory')
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--compiler',default='cc')
p.add_argument('--cflag',action='append',default=[])
p.add_argument('--run',action='store_true')
a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def save(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
if a.prepare:
    repo=pathlib.Path(__file__).resolve().parents[1]
    names=['app/audiodetect/'+n for n in ['audio_classifier.h','audio_classifier_mlp.c','model_infer.h','model_weights.h','feature_extractor.c','feature_extractor.h','wav_reader.c','wav_reader.h','keyword_detect.c','keyword_detect.h','keyword_templates.h']]
    names+=['simulator/model_infer_int8.c','simulator/model_infer_int8.h','simulator/generated/model_weights_int8.h','scripts/board_simulator_harness.c','scripts/legacy_mlp_evaluation.c','scripts/evaluate_legacy_mlp.py','model/tiny_mlp_mfcc92_robust_simulator.npz']
    for name in names:
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,dest)
    assert sha(out/'model/tiny_mlp_mfcc92_robust_simulator.npz')=='c96333aaae66d324b8b8cec48ffb0ec981ac4e975d3cf9a7d4a2fdba4307c97e'
    manifest=json.loads((a.prepare/'manifest.json').read_text(encoding='utf-8'))
    for row in manifest['inputs']:
        src=a.prepare/row['path'];assert sha(src)==row['sha256']
        dest=out/row['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    save('protocol.json',dict(labels=manifest['labels'],inputs=manifest['inputs'],model_sha256=sha(out/'model/tiny_mlp_mfcc92_robust_simulator.npz'),
         sources={name:sha(out/name) for name in names},external_role=manifest['external_role'],
         modes=dict(legacy='Unmodified original aed_classify_three_second_wav; reads FIRST THREE seconds only',
                    full='Evaluation adapter across whole WAV; original one-second features, quiet gate2e-5, event-max>=.8 else mean probabilities; zero-pad last partial second'),
         trained=False,weights_changed=False))
    with tarfile.open(out.with_suffix('.tar.gz'),'w:gz') as tar:
        for file in out.rglob('*'):
            if file.is_file():tar.add(file,arcname=file.relative_to(out).as_posix())
    print('Prepared retained MLP and65 identical WAV inputs',flush=True)
if a.run:
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'));labels=protocol['labels']
    for name,h in protocol['sources'].items():assert sha(out/name)==h
    for row in protocol['inputs']:assert sha(out/row['path'])==row['sha256']
    common=['app/audiodetect/audio_classifier_mlp.c','app/audiodetect/wav_reader.c','app/audiodetect/feature_extractor.c','app/audiodetect/keyword_detect.c','simulator/model_infer_int8.c']
    binary='mlp_evaluation.exe' if platform.system()=='Windows' else 'mlp_evaluation'
    commands=[]
    for harness,target in [('scripts/legacy_mlp_evaluation.c',binary),('scripts/board_simulator_harness.c','original_harness')]:
        cmd=[a.compiler,*a.cflag,'-std=c11','-O2','-Wall','-Wextra','-I','simulator','-I','app/audiodetect',harness,*common,'-lm','-o',target]
        commands.append(cmd)
        with (out/(target+'-build.log')).open('w') as log:subprocess.run(cmd,cwd=out,check=True,stdout=log,stderr=subprocess.STDOUT)
    results={};inputs=protocol['inputs']
    for mode in ('legacy','full'):
        cmd=[str(out/binary),mode,*[r['path'] for r in inputs]];commands.append(cmd)
        start=time.perf_counter();text=subprocess.check_output(cmd,cwd=out,text=True,timeout=120);elapsed=time.perf_counter()-start
        (out/(mode+'.tsv')).write_text(text,encoding='utf-8',newline='\n')
        parsed=list(csv.DictReader(io.StringIO(text),delimiter='\t'));assert len(parsed)==65
        matrix=[[0]*5 for _ in range(5)];records=[]
        for source,row in zip(inputs,parsed):
            assert source['path']==row['file'];truth=labels.index(source['reference_label']);pred=labels.index(row['prediction']);matrix[truth][pred]+=1
            records.append(dict(id=source['id'],truth=source['reference_label'],prediction=row['prediction'],correct=truth==pred,confidence=float(row['confidence']),
                 probabilities=[float(row['p'+str(i)]) for i in range(5)],windows=int(row['windows']),quiet_windows=int(row['quiet_windows']),input_sha256=source['sha256']))
        results[mode]=dict(correct=sum(matrix[i][i] for i in range(5)),total=65,accuracy=sum(matrix[i][i] for i in range(5))/65,
             confusion_matrix=matrix,per_class={name:dict(correct=matrix[i][i],total=sum(matrix[i]),recall=matrix[i][i]/sum(matrix[i])) for i,name in enumerate(labels)},records=records,host_process_seconds=elapsed)
        print(mode,results[mode]['correct'],'/65; baby',results[mode]['per_class']['baby_cry'],flush=True)
    original=out/('original_harness.exe' if platform.system()=='Windows' else 'original_harness')
    cmd=[str(original),*[r['path'] for r in inputs]];commands.append(cmd)
    text=subprocess.check_output(cmd,cwd=out,text=True,timeout=120);(out/'original-harness.tsv').write_text(text,encoding='utf-8')
    rows=list(csv.DictReader(io.StringIO(text),delimiter='\t'));assert len(rows)==65
    assert all(r['prediction']==o['predicted'] and abs(r['confidence']-float(o['confidence']))<1e-6 for r,o in zip(results['legacy']['records'],rows))
    save('results.json',dict(status='PASS',runtime=platform.platform(),model_sha256=protocol['model_sha256'],labels=labels,modes=results,
         compiler=subprocess.check_output([a.compiler,'--version'],text=True),protocol_sha256=sha(out/'protocol.json'),
         original_harness_parity=65,scope='Actual native C; not a new board test or full OpenVela boot',external_role=protocol['external_role'],
         cnn_comparison=dict(finetuned=dict(correct=52,total=65,baby_correct=13,baby_total=20),from_scratch=dict(correct=52,total=65,baby_correct=17,baby_total=20))))
    save('commands.json',commands)
