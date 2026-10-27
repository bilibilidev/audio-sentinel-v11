"""Physical Tiny MLP acceptance. Does not flash/reboot/read NAND or alter models."""
import csv,hashlib,json,math,pathlib,re,statistics,subprocess,sys,time,wave
sys.path.insert(0,str(pathlib.Path('board-validation-20261020/scripts').resolve()))
from board_io import Nsh,adb,ADB,SERIAL
root=pathlib.Path('board-validation-20261027')
manifest=json.loads((root/'input-manifest.json').read_text(encoding='utf-8'))
labels=manifest['labels'];model=manifest['expected_model_sha256'];remote='/data/m27'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
    (root/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def stats(v):
    if not v:return None
    s=sorted(v)
    def pct(p):
        at=(len(s)-1)*p;lo=math.floor(at);hi=math.ceil(at);return s[lo]+(s[hi]-s[lo])*(at-lo)
    return dict(count=len(s),mean=statistics.mean(s),p50=pct(.5),p95=pct(.95),minimum=s[0],maximum=s[-1],raw_values=v)
def parse(text,row):
    clean=text.replace('\r','');assert model in clean and 'Audio Sentinel Tiny MLP' in clean
    pred=re.search(r'Result:\s*(background|cough|glass_break|baby_cry|dog_bark)',clean)
    counts=re.search(r'Samples:\s*(\d+)\s+Frames:\s*(\d+)',clean)
    timing=re.search(r'(?:^|\n)\s*(\d+\.\d+)\s+sec\b',clean)
    probs={k:float(v) for k,v in re.findall(r'^\s*(background|cough|glass_break|baby_cry|dog_bark):\s*([\d.eE+-]+)',clean,re.M)}
    assert pred and counts and timing and set(probs)==set(labels),clean
    assert int(counts[1])==row['samples'],(row['id'],counts[1],row['samples'])
    assert all(math.isfinite(x) and 0<=x<=1 for x in probs.values()) and abs(sum(probs.values())-1)<1e-5
    result=dict(id=row['id'],truth=row['reference_label'],prediction=pred[1],correct=pred[1]==row['reference_label'],
                probabilities=probs,confidence=probs[pred[1]],alerted='[ALERT] Local alarm triggered:' in clean,
                samples=int(counts[1]),frames=int(counts[2]),input_sha256=row['sha256'],model_sha256=model,
                board_command_seconds=float(timing[1]),audio_seconds=int(counts[1])/16000)
    if 'host_reference_prediction' in row:
        result['linux_prediction']=row['host_reference_prediction']
        result['linux_prediction_match']=pred[1]==row['host_reference_prediction']
        result['max_probability_abs_delta_vs_linux']=max(abs(probs[k]-row['host_reference_probabilities'][i]) for i,k in enumerate(labels))
    return result
def summarize(records):
    matrix=[[0]*5 for _ in labels];per={}
    for r in records:matrix[labels.index(r['truth'])][labels.index(r['prediction'])]+=1
    for label in labels:
        same=[r for r in records if r['truth']==label];other=[r for r in records if r['truth']!=label]
        correct=sum(r['correct'] for r in same);fp=sum(r['prediction']==label for r in other)
        per[label]=dict(correct=correct,total=len(same),recall=correct/len(same),false_positives=fp,
                        negatives=len(other),one_vs_rest_fpr=fp/len(other),
                        correct_alarm_count=sum(r['correct'] and r['alerted'] for r in same),
                        alarm_false_positives=sum(r['alerted'] and r['prediction']==label for r in other))
    return dict(status='ACTUAL_BOARD_WAV_COMPLETE',model_sha256=model,total=len(records),correct=sum(r['correct'] for r in records),
                accuracy=sum(r['correct'] for r in records)/len(records),macro_recall=statistics.mean(x['recall'] for x in per.values()),
                labels=labels,confusion_matrix=matrix,per_class=per,board_command_seconds=stats([r['board_command_seconds'] for r in records]),
                total_audio_seconds=sum(r['audio_seconds'] for r in records),total_command_seconds=sum(r['board_command_seconds'] for r in records),
                linux_prediction_matches=sum(r.get('linux_prediction_match',False) for r in records) if 'linux_prediction_match' in records[0] else None,
                max_probability_abs_delta_vs_linux=max(r.get('max_probability_abs_delta_vs_linux',0) for r in records) if 'linux_prediction_match' in records[0] else None,
                timing_scope='NSH time: whole WAV IO + feature DSP + asinh + MLP + aggregation + stdout, under normal scheduling, UI alive and capture stopped. Excludes recording/USB and is not event-onset-to-alert latency.',
                dataset_role=manifest['data_role'],microphone_accuracy_measured=False,
                board_input_directory=remote,filesystem='tmpfs RAM' if remote.startswith('/tmp/') else 'YAFFS NAND',records=records)

shell=None
def run(cmd,timeout=30,expected=0):
    text,rc=shell.run(cmd,timeout)
    if expected is not None:assert rc==expected,(cmd,rc,text)
    return text,rc
def idle():
    text,_=run('ps');assert not re.search(r'\barecord\s+-D',text),'UI/microphone recording is active; timing aborted'
    return text
def upload():
    idle();text,_=run('ls -l /dev/bootloader');assert '16777216' in text
    text,rc=run('ls '+remote,expected=None)
    prefix='ram-' if remote.startswith('/tmp/') else ''
    marker=root/(prefix+'owned-test-directory.txt')
    if rc==0:assert marker.exists() and marker.read_text().strip()==remote,'Unknown existing remote test directory'
    else:
        run('mkdir '+remote);marker.write_text(remote+'\n')
    roundtrip=root/(prefix+'roundtrip');roundtrip.mkdir(exist_ok=True)
    for dirname in ['b','n20','n10','n5','k','i']:
        print('UPLOAD',dirname,flush=True)
        adb(['push',str(root/'inputs'/dirname),remote],root/f'{prefix}push-{dirname}.log',240)
        assert not (roundtrip/dirname).exists()
        adb(['pull',remote+'/'+dirname,str(roundtrip/dirname)],root/f'{prefix}pull-{dirname}.log',240)
    adb(['push',str(root/'inputs/f.wav'),remote+'/f.wav'],root/(prefix+'push-fixed.log'))
    adb(['pull',remote+'/f.wav',str(roundtrip/'f.wav')],root/(prefix+'pull-fixed.log'))
    expected=json.loads((root/'input-sha256.json').read_text())
    for rel,digest in expected.items():assert sha(roundtrip/rel)==digest,rel
    text,_=run('time "audiodetect '+remote+'/b/000.wav --debug"')
    row=parse(text,manifest['records'][0]);assert row['model_sha256']==model
    write(prefix+'setup.json',dict(status='RUNNING_MODEL_AND_280_INPUTS_VERIFIED',model_sha256=model,
                           input_roundtrip_hashes_pass=True,files=len(expected),
                           image_sha256=manifest['operator_declared_image_sha256'],image_hash_scope='User-declared flashed image, not NAND readback',
                           initial_inference=row,free=run('free')[0],df=run('df')[0]))
def evaluate(name,rows):
    assert not (root/f'{name}.json').exists();idle();records=[]
    logdir=root/'logs'/name;logdir.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows):
        command=f'time "audiodetect {remote}/{row["board_file"]} --debug"'
        start=time.monotonic();text,_=run(command,120);elapsed=time.monotonic()-start
        (logdir/f'{row["id"]}.log').write_text(text,encoding='utf-8')
        r=parse(text,row);r['host_wall_seconds_including_prompt_and_status']=elapsed;records.append(r)
        write('progress.json',dict(stage=name,completed=i+1,total=len(rows),correct=sum(x['correct'] for x in records)))
        print(name,i+1,len(rows),row['id'],r['prediction'],r['correct'],r['board_command_seconds'],flush=True)
    idle();result=summarize(records);write(name+'.json',result)
    return result
def latency():
    idle();logdir=root/'logs/latency';logdir.mkdir(parents=True,exist_ok=True)
    values=[];row=dict(id='fixed3s',reference_label='dog_bark',samples=48000,sha256=manifest['fixed_wav_sha256'])
    before,_=run('free')
    for i in range(33):
        text,_=run('time "audiodetect '+remote+'/f.wav --debug"');r=parse(text,row)
        (logdir/f'{i:02}.log').write_text(text,encoding='utf-8')
        if i>=3:values.append(r['board_command_seconds'])
    after,_=run('free');idle()
    write('latency.json',dict(model_sha256=model,warmups=3,fixed_3s_wav_sha256=row['sha256'],
                             board_command_seconds=stats(values),free_before=before,free_after=after,
                             scope='NSH time: 3-second WAV read + DSP + asinh + MLP + mean aggregation + stdout; excludes microphone capture, USB transfer, process observed wall time and UI alert latency',
                             board_input_directory=remote,filesystem='tmpfs RAM' if remote.startswith('/tmp/') else 'YAFFS NAND',
                             memory_scope='NuttX free/maxused are system heap including OS/UI, test input tmpfs allocation and lifetime high-water, not inference-exclusive peak RAM',power_watts=None,energy_joules=None))
def auxiliary(clean):
    idle();logdir=root/'logs/auxiliary';logdir.mkdir(parents=True,exist_ok=True)
    keywords=[];names=['救命','帮助','报警']
    for i,row in enumerate(manifest['keywords']):
        text,rc=run('audiodetect keyword '+remote+'/'+row['board_file'],60,expected=None)
        (logdir/f'keyword-{i:02}.log').write_text(text,encoding='utf-8')
        expected=names[row['expected']] if row['expected']>=0 else 'none'
        match=re.search(r'Keyword:\s*(\S+)',text)
        keywords.append(dict(board_file=row['board_file'],expected=expected,actual=match[1] if match else None,
                             passed=rc==0 and match is not None and match[1]==expected,synthetic=True,exit_code=rc))
    cases=[]
    for p in sorted((root/'inputs/i').glob('*.wav')):
        text,rc=run('audiodetect '+remote+'/i/'+p.name+' --debug',30,expected=None)
        (logdir/p.with_suffix('.log').name).write_text(text,encoding='utf-8')
        reject=p.name in ['empty.wav','bad.wav','trunc.wav']
        cases.append(dict(file=p.name,exit_code=rc,expected_reject=reject,passed=(rc!=0 if reject else rc==0 and '[ALERT] Local alarm triggered:' not in text)))
    aliases=[]
    for row in [clean['records'][i] for i in [0,12,30,45,64]]:
        idx=next(i for i,r in enumerate(manifest['records']) if r['id']==row['id'])
        text,rc=run(f'audiodetect {remote}/b/{idx:03}.wav --int8')
        pred=re.search(r'Result:\s*(\w+)',text)[1]
        aliases.append(dict(id=row['id'],prediction=pred,passed=pred==row['prediction']))
        (logdir/f'alias-{idx:03}.log').write_text(text,encoding='utf-8')
    selected=next(r for r in clean['records'] if r['prediction']!='background' and r['confidence']>.9)
    idx=next(i for i,r in enumerate(manifest['records']) if r['id']==selected['id'])
    text,rc=run(f'audiodetect {remote}/b/{idx:03}.wav --report',30,expected=None)
    (logdir/'network-report.log').write_text(text,encoding='utf-8')
    reverse=subprocess.run([ADB,'-s',SERIAL,'reverse','--list'],capture_output=True,timeout=10)
    network=dict(fixture=selected['id'],exit_code=rc,http_success='HTTP report: SUCCESS' in text,
                 mqtt_success='MQTT report: SUCCESS' in text,configured_host='127.0.0.1',
                 reverse_supported=reverse.returncode==0,reverse_output=(reverse.stdout+reverse.stderr).decode(errors='replace'),
                 scope='Actual board application invocation. localhost refers to the board; no HTTP/MQTT service provisioned in this firmware. Connection failure does not constitute delivered network alarm.',
                 ui_network_integration=False)
    write('auxiliary.json',dict(keywords=keywords,keyword_real_voice_accuracy=False,input_validation=cases,int8_alias=aliases,network=network))
def capture():
    idle();records=[];logdir=root/'logs/capture';logdir.mkdir(parents=True,exist_ok=True)
    before,_=run('free')
    for i in range(5):
        script=root/f'mic-{i}.nsh'
        script.write_text(f'arecord -D default -r 16000 -f 16 -c 1 -d 3 {remote}/m{i}.wav\naudiodetect {remote}/m{i}.wav --debug\n',encoding='ascii')
        adb(['push',str(script),f'{remote}/m{i}.nsh'],root/f'mic-script-push-{i}.log')
        start=time.monotonic()
        try:
            text,rc=run(f'time "sh {remote}/m{i}.nsh"',20,expected=None)
            (logdir/f'{i}.log').write_text(text,encoding='utf-8')
            dest=root/f'microphone-{i}.wav';adb(['pull',f'{remote}/m{i}.wav',str(dest)],root/f'mic-pull-{i}.log')
            with wave.open(str(dest)) as w:
                info=dict(rate=w.getframerate(),channels=w.getnchannels(),sample_width=w.getsampwidth(),frames=w.getnframes());pcm=w.readframes(w.getnframes())
            assert (info['rate'],info['channels'],info['sample_width'],info['frames'])==(16000,1,2,48000)
            timing=re.search(r'(?:^|[\r\n])\s*(\d+\.\d+)\s+sec\b',text);assert timing and rc==0 and model in text
            import array
            samples=array.array('h',pcm)
            records.append(dict(index=i,passed=True,**info,sha256=sha(dest),
                                rms_pcm=math.sqrt(sum(x*x for x in samples)/len(samples)),peak_pcm=max(abs(x) for x in samples),
                                nonzero_samples=sum(x!=0 for x in samples),clipped_samples=sum(abs(x)>=32767 for x in samples),
                                capture_plus_inference_seconds=float(timing[1]),host_wall_seconds=time.monotonic()-start))
        except Exception as exc:
            records.append(dict(index=i,passed=False,error=str(exc)));write('capture.json',dict(records=records,passed=False))
            raise
        idle()
    after,_=run('free')
    write('capture.json',dict(passed=all(r['passed'] for r in records),records=records,free_before=before,free_after=after,
                             capture_plus_inference_seconds=stats([r['capture_plus_inference_seconds'] for r in records]),
                             scope='Actual room microphone capture, 3s arecord then full WAV inference; no known event labels supplied. This validates acquisition, not acoustic classification accuracy or event-onset alert latency.',
                             raw_microphone_audio_for_local_inspection_only=True,microphone_accuracy_measured=False,
                             board_input_directory=remote,filesystem='tmpfs RAM' if remote.startswith('/tmp/') else 'YAFFS NAND'))
def soak():
    idle();start=time.monotonic();rows=[];free=[]
    for i in range(200):
        text,rc=run('audiodetect '+remote+'/f.wav',30,expected=None)
        pred=re.search(r'Result:\s*(\w+)',text)
        rows.append(dict(iteration=i+1,passed=rc==0 and model in text and pred is not None,prediction=pred[1] if pred else None))
        assert rows[-1]['passed'],text
        if i%20==0:
            free.append(dict(iteration=i+1,text=run('free')[0]));idle()
            write('soak-progress.json',dict(completed=i+1,target=200,elapsed_seconds=time.monotonic()-start))
        time.sleep(.25)
    write('soak.json',dict(iterations=len(rows),elapsed_wall_seconds=time.monotonic()-start,all_passed=all(r['passed'] for r in rows),
                           unique_predictions=sorted(set(r['prediction'] for r in rows)),memory_observations=free,
                           rows=rows,scope='Short repeated-WAV stress with UI idle; not 1h/24h continuous microphone stability',identity=run('uname -a')[0],uptime=run('uptime')[0]))

def main():
    global shell
    assert not (root/'completed.json').exists()
    shell=Nsh(root/'suite-shell.log');stages=[]
    try:
        upload();stages.append('input_roundtrip280')
        clean=evaluate('clean65',manifest['records']);stages.append('clean65')
        latency();stages.append('fixed3s_latency')
        for snr in [20,10,5]:evaluate('noise'+str(snr),manifest['noise'][str(snr)]);stages.append('noise'+str(snr))
        auxiliary(clean);stages.append('auxiliary')
        capture();stages.append('microphone_capture')
        soak();stages.append('short_soak200')
        write('completed.json',dict(status='BOARD_SUITE_COMPLETE',stages=stages,model_sha256=model,final_free=run('free')[0],final_processes=idle(),uptime=run('uptime')[0]))
        print('BOARD_SUITE_COMPLETE',flush=True)
    except Exception as exc:
        write('suite-error.json',dict(error=str(exc),completed_stages=stages));raise
    finally:shell.close()
if __name__=='__main__':main()
