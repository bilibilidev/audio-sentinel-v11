"""Smoke-test unchanged keyword and HTTP/MQTT code using local fixtures only."""
import hashlib, json, pathlib, re, socket, subprocess, threading, time, wave
root=pathlib.Path(__file__).resolve().parent
out=root/'feature-results';out.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
def save(name, obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
commands=[]
def invoke(cmd):
    commands.append(cmd);return subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=12)
src=root/'app-source'
harness=out/'feature_harness.c'
harness.write_text('''#include "alarm_report.h"
#include "keyword_detect.h"
#include "wav_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv) {
  if(argc==3 && !strcmp(argv[1],"keyword")) {
    static int16_t pcm[32000]; float confidence=0;
    int n=read_wav_16k_mono(argv[2],pcm,32000); if(n<=0)return 2;
    int kw=keyword_detect(pcm,n,&confidence);
    printf("KW %d %.9g %d\\n",kw,confidence,n);return 0;
  }
  if(argc!=4)return 2;
  if(alarm_report_init("127.0.0.1",atoi(argv[2]),atoi(argv[3]),"scratch_probe"))return 2;
  int rc=!strcmp(argv[1],"http")?alarm_send_http("baby_cry",.95f,1,1792800000):alarm_send_mqtt("baby_cry",.95f,1,1792800000);
  printf("RETURN %d\\n",rc);return rc?1:0;
}
''')
binary=out/'feature_harness'
cmd=['cc','-std=c11','-O2','-Wall','-Wextra','-pedantic','-I',str(src),str(harness),str(src/'keyword_detect.c'),str(src/'wav_reader.c'),str(src/'alarm_report.c'),'-lm','-o',str(binary)]
r=invoke(cmd);(out/'build.log').write_text(r.stdout+r.stderr);assert r.returncode==0
app=root/'audiodetect'; keyword=[]
for case in read(root/'keyword-manifest.json')['cases']:
    p=root/case['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==case['sha256']
    api=invoke([str(binary),'keyword',str(p)]);cli=invoke([str(app),'keyword',str(p)])
    assert api.returncode==cli.returncode==0
    match=re.search(r'^KW (-?\d+) ([0-9.eE+-]+) (\d+)$',api.stdout,re.M);assert match
    keyword.append(dict(**case,api_prediction=int(match[1]),api_confidence=float(match[2]),api_correct=int(match[1])==case['expected'],cli_stdout=cli.stdout))

with wave.open(str(root/'keyword-inputs/kw-0-0.wav'),'rb') as w: pcm=w.readframes(w.getnframes())
delayed=out/'delayed-keyword.wav'
with wave.open(str(delayed),'wb') as w:
    w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(bytes(32000)+pcm)
delay_api=invoke([str(binary),'keyword',str(delayed)]);delay_cli=invoke([str(app),'keyword',str(delayed)])
save('keywords.json',dict(scope='Synthetic generator fixtures; no real spoken-word accuracy claim',cases=keyword,
     api_matches=sum(r['api_correct'] for r in keyword),total=len(keyword),
     delayed_keyword=dict(leading_silence_seconds=1,api=delay_api.stdout,cli=delay_cli.stdout),
     current_cli_limit_samples=16000,ui_integration=False))

class Fixture:
    def __init__(self, handler, port=0):
        self.sock=socket.socket();self.sock.bind(('127.0.0.1',port));self.sock.listen(1);self.sock.settimeout(8)
        self.port=self.sock.getsockname()[1];self.result={};self.handler=handler
    def __enter__(self):
        def worker():
            try:
                with self.sock.accept()[0] as c:
                    c.settimeout(5);self.handler(c,self.result)
            except Exception as e:self.result['error']=repr(e)
            finally:self.sock.close()
        self.thread=threading.Thread(target=worker,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.thread.join(9)
def recv_exact(c,n):
    b=b''
    while len(b)<n:
        p=c.recv(n-len(b))
        if not p:raise EOFError()
        b+=p
    return b
def http_handler(response, fragment=False):
    def handle(c,result):
        data=b''
        while b'\r\n\r\n' not in data:data+=c.recv(4096)
        header,body=data.split(b'\r\n\r\n',1)
        length=int(re.search(br'Content-Length: (\d+)',header,re.I)[1])
        if len(body)<length:body+=recv_exact(c,length-len(body))
        result.update(request_line=header.split(b'\r\n')[0].decode(),json=json.loads(body[:length]))
        if fragment:
            c.sendall(response[:8]);time.sleep(.1)
            try:c.sendall(response[8:])
            except OSError:pass
        else:c.sendall(response)
    return handle
def mqtt_packet(c):
    kind=recv_exact(c,1)[0];remaining=0;scale=1
    for _ in range(4):
        value=recv_exact(c,1)[0];remaining+=(value&127)*scale
        if value<128:return kind,recv_exact(c,remaining)
        scale*=128
    raise ValueError('bad remaining length')
def mqtt_handler(ack, fragment=False):
    def handle(c,result):
        kind,data=mqtt_packet(c);assert kind==0x10
        result['connect_hex']=data.hex();assert data[:7]==b'\x00\x04MQTT\x04'
        if fragment:
            for b in ack:c.sendall(bytes([b]));time.sleep(.025)
        else:c.sendall(ack)
        if ack[3]!=0:return
        kind,data=mqtt_packet(c);assert kind==0x30
        size=int.from_bytes(data[:2],'big');result.update(topic=data[2:2+size].decode(),json=json.loads(data[2+size:]),qos=0)
    return handle
network=[]
for name,status,body,fragment,expected in [('http200',200,b'OK',False,True),('http201',201,b'Created',False,True),
    ('http500',500,b'FAIL',False,False),('http500_body_200',500,b'error code 200',False,False),
    ('http200_fragmented_status',200,b'OK',True,True)]:
    response=f'HTTP/1.1 {status} Test\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n'.encode()+body
    with Fixture(http_handler(response,fragment)) as f:r=invoke([str(binary),'http',str(f.port),'1'])
    accepted=r.returncode==0
    network.append(dict(test=name,expected_accepted=expected,actual_accepted=accepted,pass_check=accepted==expected,stdout=r.stdout,fixture=f.result))
for name,ack,fragment,expected in [('mqtt_valid',b'\x20\x02\x00\x00',False,True),
    ('mqtt_fragmented_ack',b'\x20\x02\x00\x00',True,True),('mqtt_refused',b'\x20\x02\x00\x05',False,False),
    ('mqtt_invalid_remaining_length',b'\x20\x01\x00\x00',False,False)]:
    with Fixture(mqtt_handler(ack,fragment)) as f:r=invoke([str(binary),'mqtt','1',str(f.port)])
    accepted=r.returncode==0
    network.append(dict(test=name,expected_accepted=expected,actual_accepted=accepted,pass_check=accepted==expected,stdout=r.stdout,fixture=f.result))

summary=read(root/'c-results/summary.json')
record=next(r for r in summary['external'][-1]['records'] if r['prediction']!=0 and max(r['probabilities'])>.80)
source=next(r for r in read(root/'manifest.json')['inputs'] if r['id']==record['id'])
try:
    with Fixture(http_handler(b'HTTP/1.1 201 Created\r\nContent-Length: 0\r\nConnection: close\r\n\r\n'),8080) as http, Fixture(mqtt_handler(b'\x20\x02\x00\x00'),1883) as mqtt:
        result=invoke([str(app),source['path'],'--report'])
    assert result.returncode==0 and 'HTTP report: SUCCESS' in result.stdout and 'MQTT report: SUCCESS' in result.stdout
    assert http.result['json']['event']==mqtt.result['json']['event']==read(root/'manifest.json')['labels'][record['prediction']]
    endtoend=dict(status='PASS',input=source['id'],stdout=result.stdout,http=http.result,mqtt=mqtt.result)
except OSError as e:endtoend=dict(status='BLOCKED existing port binding',reason=str(e))
save('network.json',dict(scope='Linux loopback fixtures; MQTT protocol mock, not a deployed broker or physical board',
     cases=network,known_failures=[r['test'] for r in network if not r['pass_check']],end_to_end_app=endtoend,
     ui_integration=False,keyword_triggers_network=False))
save('commands.json',commands)
print('FEATURE CHECKS DONE; synthetic keyword API matches',sum(r['api_correct'] for r in keyword),'/',len(keyword),
      '; network known failures',[r['test'] for r in network if not r['pass_check']],'; CLI loopback',endtoend['status'],flush=True)
