"""Measure capture and inference separately; avoid the NSH time/sh limitation."""
import array,json,math,pathlib,re,time,wave
import board_suite as s
from board_io import Nsh,adb
root=s.root;s.remote='/tmp/m27';s.shell=Nsh(root/'capture-soak-shell.log')
records=[]
def seconds(text):
    m=re.search(r'(?:^|[\r\n])\s*(\d+\.\d+)\s+sec\b',text);assert m,text
    return float(m[1])
try:
    s.idle();s.run('cd /tmp/m27');before=s.run('free')[0]
    for i in range(5):
        mic_path=f'/tmp/q{i}.wav'
        _,existing=s.run('ls '+mic_path,expected=None)
        assert existing!=0,'Preserve an existing unknown microphone file'
        t=time.monotonic()
        command=f'time "arecord -Ddefault -r16000 -f16 -c1 -d3 {mic_path}"'
        assert len(command)<=60,(len(command),command)
        cap,rc=s.run(command,20)
        infer,rc=s.run(f'time "audiodetect {mic_path} --debug"',20)
        host_seconds=time.monotonic()-t
        (root/f'capture-retry-{i}.log').write_text(cap+'\n'+infer,encoding='utf-8')
        target=root/f'microphone-retry-{i}.wav'
        for attempt in range(3):
            try:
                adb(['pull',mic_path,str(target)],root/f'capture-retry-pull-{i}-{attempt}.log',30);break
            except RuntimeError:
                if attempt==2:raise
                time.sleep(2)
        with wave.open(str(target)) as w:
            info=dict(rate=w.getframerate(),channels=w.getnchannels(),sample_width=w.getsampwidth(),frames=w.getnframes());raw=w.readframes(w.getnframes())
        assert (info['rate'],info['channels'],info['sample_width'],info['frames'])==(16000,1,2,48000)
        assert s.model in infer and re.search(r'Samples:\s*48000\b',infer)
        samples=array.array('h',raw)
        records.append(dict(index=i,passed=True,board_file=mic_path,**info,sha256=s.sha(target),capture_seconds=seconds(cap),
                            inference_seconds=seconds(infer),sum_of_separate_command_seconds=seconds(cap)+seconds(infer),
                            host_sequential_wall_seconds=host_seconds,rms_pcm=math.sqrt(sum(x*x for x in samples)/len(samples)),
                            peak_pcm=max(abs(x) for x in samples),nonzero_samples=sum(x!=0 for x in samples),
                            clipped_samples=sum(abs(x)>=32767 for x in samples),
                            prediction=re.search(r'Result:\s*(\w+)',infer)[1]))
        s.idle();print('MIC',i+1,records[-1],flush=True)
    after=s.run('free')[0]
    s.write('capture.json',dict(passed=True,records=records,capture_seconds=s.stats([r['capture_seconds'] for r in records]),
                               inference_seconds=s.stats([r['inference_seconds'] for r in records]),
                               sum_of_separate_command_seconds=s.stats([r['sum_of_separate_command_seconds'] for r in records]),
                               host_sequential_wall_seconds=s.stats([r['host_sequential_wall_seconds'] for r in records]),
                               board_input_directory=s.remote,filesystem='tmpfs RAM',free_before=before,free_after=after,
                               scope='Actual local microphone, 3s arecord and WAV classification measured as two sequential NSH time commands. Sum is not a single timed chain or event-onset alert latency. Host sequential wall includes ADB command/status exchange.',
                               microphone_accuracy_measured=False,room_event_labels_supplied=False,raw_microphone_audio_local_only=True,
                               previous_script_timing_issue='time sh script returned no elapsed line and nonzero status despite producing a 48000-frame WAV; original failed-attempt logs retained'))
    s.soak()
    s.write('completed.json',dict(status='CORE_EXPERIMENTS_COMPLETE_WITH_NAND_WRITE_FAILURE',
                                  model_sha256=s.model,completed_stages=['NAND clean65','RAM input280 roundtrip','RAM clean65','RAM fixed3s latency33','RAM noise20/10/5 each65','keyword11','input validation8','INT8 alias5','network failure path','RAM microphone5','short soak200'],
                                  final_free=s.run('free')[0],final_processes=s.idle(),uptime=s.run('uptime')[0],
                                  storage_issue='SPI0 timeout and YAFFS retries on bulk NAND upload; all later uploads/captures used tmpfs',
                                  system_acceptance_pass=False,one_hour_stability_tested=False,power_measured=False))
    print('CORE_EXPERIMENTS_COMPLETE_WITH_NAND_WRITE_FAILURE',flush=True)
except Exception as exc:
    s.write('capture-soak-error.json',dict(error=str(exc),successful_microphone_records=records));raise
finally:s.shell.close()
