"""Measure a fixed 3-second WAV; never flashes, reboots or records audio."""
import argparse,hashlib,json,math,re,wave
from pathlib import Path
from board_io import Nsh,adb

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--wav',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--repeats',type=int,default=20)
a=p.parse_args()
assert a.repeats>=2 and not a.output.exists()
a.output.mkdir(parents=True)
with wave.open(str(a.wav)) as w:
    assert (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes())==(1,2,16000,48000)
expected=json.loads((Path(__file__).resolve().parents[2]/'model/cnn-v2/deployment.json').read_text())['model_sha256']
s=Nsh(a.output/'shell.log')
try:
    text,rc=s.run('ps');assert rc==0 and not re.search(r'\barecord\s+-D',text),'Pause capture first'
    # /data/c20 must already be the operator-owned directory from prepare.py.
    text,rc=s.run('ls /data/c20');assert rc==0
    adb(['push',str(a.wav),'/data/c20/f.wav'],a.output/'push.log')
    adb(['pull','/data/c20/f.wav',str(a.output/'roundtrip.wav')],a.output/'pull.log')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    assert sha(a.wav)==sha(a.output/'roundtrip.wav')
    values=[]
    for i in range(3+a.repeats):
        text,rc=s.run('time "audiodetect /data/c20/f.wav --debug"')
        assert rc==0 and expected in text and re.search(r'Samples:\s*48000\b',text)
        elapsed=re.search(r'(?:^|[\r\n])\s*(\d+\.\d+)\s+sec\b',text);assert elapsed
        if i>=3:values.append(float(elapsed[1]))
    text,rc=s.run('ps');assert rc==0 and not re.search(r'\barecord\s+-D',text)
    v=sorted(values)
    def pct(p):
        x=(len(v)-1)*p;lo=math.floor(x);hi=math.ceil(x)
        return v[lo]+(v[hi]-v[lo])*(x-lo)
    result=dict(model_sha256=expected,wav_sha256=sha(a.wav),count=len(v),warmups=3,
        mean=sum(v)/len(v),p50=pct(.5),p95=pct(.95),min=v[0],max=v[-1],raw_seconds=values,
        scope='NSH time in seconds: WAV IO, DSP, CNN and stdout under normal task scheduling; UI alive and capture paused. Excludes USB, capture and UI alert latency.',power_measured=False)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
finally:s.close()
