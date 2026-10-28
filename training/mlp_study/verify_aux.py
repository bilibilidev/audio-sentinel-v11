import argparse,http.server,json,pathlib,re,socket,subprocess,threading,platform,hashlib
p=argparse.ArgumentParser();p.add_argument('--binary',type=pathlib.Path,required=True);p.add_argument('--inputs',type=pathlib.Path,required=True);p.add_argument('--reference',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[2]/'results/mlp-study-20261026/linux/app-verification.json');p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
binary=a.binary.resolve();captured={}
class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        captured['http_path']=self.path;captured['http']=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(201);self.send_header('Content-Length','0');self.end_headers()
    def log_message(self,*args):pass
def exact(c,n):
    data=b''
    while len(data)<n:
        b=c.recv(n-len(data))
        if not b:raise EOFError()
        data+=b
    return data
def packet(c):
    kind=exact(c,1)[0];size=0;factor=1
    for _ in range(4):
        value=exact(c,1)[0];size+=(value&127)*factor
        if value<128:return kind,exact(c,size)
        factor*=128
    raise ValueError('length')
http=http.server.HTTPServer(('127.0.0.1',8080),Handler)
mqtt=socket.socket();mqtt.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);mqtt.bind(('127.0.0.1',1883));mqtt.listen(1);mqtt.settimeout(15)
def mqtt_worker():
    try:
        with mqtt.accept()[0] as c:
            c.settimeout(8);kind,body=packet(c);assert kind==16 and body[:7]==b'\x00\x04MQTT\x04'
            c.sendall(b'\x20\x02\x00\x00');kind,body=packet(c);assert kind==48
            size=int.from_bytes(body[:2],'big');captured['mqtt_topic']=body[2:2+size].decode();captured['mqtt']=json.loads(body[2+size:])
    except Exception as e:captured['mqtt_error']=repr(e)
    finally:mqtt.close()
th=threading.Thread(target=http.serve_forever,daemon=True);tm=threading.Thread(target=mqtt_worker,daemon=True);th.start();tm.start()
evaluation=json.loads((a.reference).read_text(encoding='utf-8'))
row=next(r for r in evaluation['records']['mlp'] if r['prediction']!=0 and max(r['probabilities'])>.80)
inputs=a.inputs.resolve()
manifest=json.loads((inputs/'manifest.json').read_text(encoding='utf-8'))
source=next(r for r in manifest['inputs'] if r['id']==row['id'])
try:
    command=[str(binary),str(inputs/source['path']),'--report']
    response=subprocess.check_output(command,timeout=20).decode('utf-8',errors='replace')
    tm.join(10)
finally:http.shutdown();http.server_close()
assert 'HTTP report: SUCCESS' in response and 'MQTT report: SUCCESS' in response
assert captured['http_path']=='/api/alarm' and captured['mqtt_topic']=='audiodetect/openvela_aed_01/alarm'
assert captured['http']==captured['mqtt'];assert isinstance(captured['http']['timestamp'],int)
keywords=[];kw_inputs=inputs/'keyword-inputs';names=['救命','帮助','报警']
for k in range(3):
    for seed in range(3):
        text=subprocess.check_output([str(binary),'keyword',str(kw_inputs/f'kw-{k}-{seed}.wav')],timeout=10).decode('utf-8',errors='replace')
        matched=re.search(r'Keyword: (\S+)',text)[1];assert matched==names[k],(k,matched)
        keywords.append(dict(keyword=k,fixture=seed,matched=True,synthetic=True))
for name in ('silence','low-noise'):
    text=subprocess.check_output([str(binary),'keyword',str(kw_inputs/(name+'.wav'))],timeout=10).decode('utf-8',errors='replace')
    assert 'Keyword: none' in text
result=dict(status='PASS',runtime=platform.platform(),http=captured['http'],mqtt=captured['mqtt'],mqtt_topic=captured['mqtt_topic'],
    keyword_synthetic_cases=keywords,keyword_negative_cases=2,real_voice_accuracy_tested=False,
    loopback_only=True,mqtt_fixture='Minimal protocol test service, not deployed broker',ui_alarm_integration=False,
    known_network_issues='Previously reproduced HTTP status matching/fragmentation and CONNACK validation defects remain; this run tests positive integration only')
a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Integrated default MLP keyword and HTTP/MQTT loopback PASS')
