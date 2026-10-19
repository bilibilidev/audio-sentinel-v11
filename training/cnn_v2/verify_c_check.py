"""Compile and verify streaming C on Linux; no external model selection here."""
import csv,hashlib,io,json,pathlib,subprocess,sys,time
import numpy as np
root=pathlib.Path(sys.argv[1]).resolve() if len(sys.argv)>1 else pathlib.Path(__file__).resolve().parent
protocol=json.loads((root/'protocol.json').read_text())
for name,h in protocol['inputs'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
command=['cc','-std=c11','-O2','-Wall','-Wextra','-pedantic','cnn_stream_v2.c','cnn_cli.c','-lm','-o','cnn_stream']
with (root/'build.log').open('w') as log:subprocess.run(command,cwd=root,check=True,stdout=log,stderr=subprocess.STDOUT)
commands=[command];checks=[];full_trace=None;start=time.perf_counter()
for name in ['input','prefix','short','silence']:
    expected=np.load(root/(name+'-reference.npz'),allow_pickle=False)['LOGITS']
    outputs=[];traces=[]
    for chunk in protocol['chunk_sizes']:
        trace=root/f'{name}-{chunk}.tsv'
        cmd=[str(root/'cnn_stream'),'--chunk',str(chunk),'--trace',str(trace),str(root/(name+'.wav'))]
        commands.append(cmd);output=subprocess.check_output(cmd,text=True);outputs.append(output)
        (root/f'{name}-{chunk}-result.tsv').write_text(output)
        rows=list(csv.DictReader(io.StringIO(trace.read_text()),delimiter='\t'))
        actual=np.asarray([[float(r[f'l{i}']) for i in range(5)] for r in rows])
        assert actual.shape==expected.shape
        delta=float(np.max(np.abs(actual-expected)))
        assert delta<protocol['max_abs_logit_tolerance'],(name,chunk,delta)
        exp_probs=np.exp(expected-expected.max(axis=1,keepdims=True));exp_probs/=exp_probs.sum(axis=1,keepdims=True)
        act_probs=np.exp(actual-actual.max(axis=1,keepdims=True));act_probs/=act_probs.sum(axis=1,keepdims=True)
        pdelta=float(np.max(np.abs(exp_probs-act_probs)))
        assert pdelta<protocol['max_abs_probability_tolerance'],(name,chunk,pdelta)
        expected_ema=[];state=None
        for logits in expected:
            state=logits.copy() if state is None else np.float32(.98)*state+np.float32(.02)*logits
            expected_ema.append(state.copy())
        actual_ema=np.asarray([[float(r[f'e{i}']) for i in range(5)] for r in rows])
        edelta=float(np.max(np.abs(actual_ema-np.asarray(expected_ema))))
        assert edelta<protocol['max_abs_logit_tolerance'],(name,chunk,'EMA',edelta)
        probs=np.load(root/(name+'-reference.npz'),allow_pickle=False)['PROB']
        result=list(csv.DictReader(io.StringIO(output),delimiter='\t'))[0]
        assert result['prediction']==['background','cough','glass_break','baby_cry','dog_bark'][int(probs.argmax())]
        assert max(abs(float(result[f'p{i}'])-probs[i]) for i in range(5))<protocol['max_abs_probability_tolerance']
        emit_samples=[int(r['end_sample']) for r in rows if r['emit']=='1']
        assert len(emit_samples)<2 or np.array_equal(np.diff(emit_samples),np.full(len(emit_samples)-1,1600))
        checks.append({'input':name,'chunk':chunk,'frames':len(rows),'max_logit_delta':delta,'max_probability_delta':pdelta,'max_ema_delta':edelta})
        traces.append(trace.read_bytes())
    assert all(t==traces[0] for t in traces),'Input-callback boundaries changed outputs'
    assert all(o==outputs[0] for o in outputs),'Input-callback boundaries changed final result'
    if name=='input':full_trace=list(csv.DictReader(io.StringIO(traces[0].decode()),delimiter='\t'))
    if name=='prefix':assert full_trace[:100]==list(csv.DictReader(io.StringIO(traces[0].decode()),delimiter='\t'))
san_command=['cc','-std=c11','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer','cnn_stream_v2.c','cnn_cli.c','-lm','-o','cnn_stream_sanitized']
commands.append(san_command)
with (root/'sanitizer-build.log').open('w') as log:subprocess.run(san_command,cwd=root,check=True,stdout=log,stderr=subprocess.STDOUT)
with (root/'sanitizer-run.log').open('w') as log:subprocess.run([str(root/'cnn_stream_sanitized'),'--chunk','997',str(root/'input.wav'),str(root/'short.wav'),str(root/'silence.wav')],check=True,stdout=log,stderr=subprocess.STDOUT)
elapsed=time.perf_counter()-start
summary={'status':'PASS','executed_linux_c':True,'checks':checks,'callback_partition_invariant':True,'future_audio_not_used':True,
         'emission_interval_seconds':.1,'sanitizers':'AddressSanitizer + UndefinedBehaviorSanitizer pass',
         'model_sha256':protocol['model_sha256'],'binary_sha256':hashlib.sha256((root/'cnn_stream').read_bytes()).hexdigest(),
         'compiler':subprocess.check_output(['cc','--version'],text=True),'test_elapsed_seconds':elapsed,
         'note':'Numerical, state and safety tests on ESC audio and generated boundary cases, not external-set accuracy'}
(root/'commands.json').write_text(json.dumps(commands,indent=2));(root/'verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
