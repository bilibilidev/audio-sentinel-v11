import http.server as http_server
import json,socket,threading
from common import ROOT,read,title,table,entry
def demo(ctx):
    title('亮点 4/5 · 远程告警接口：本机 HTTP / MQTT 实际收发')
    print('启动本机 HTTP 接收端和 MQTT QoS 0 接收端，运行声音事件上报。')
    row,wav=ctx.example('cough',alarm=True);captured={};errors=[]
    class Handler(http_server.BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                assert self.path=='/api/alarm'
                size=int(self.headers.get('Content-Length','0'));assert 0<size<2048
                captured['http_path']=self.path;captured['http']=json.loads(self.rfile.read(size))
                self.send_response(201);self.send_header('Content-Length','0');self.end_headers()
            except Exception as exc:errors.append(str(exc));self.send_error(400)
        def log_message(self,*args):pass
    http=None;mqtt=None;worker=None
    def exact(c,n):
        data=b''
        while len(data)<n:
            b=c.recv(n-len(data))
            if not b:raise EOFError('MQTT closed early')
            data+=b
        return data
    def packet(c):
        kind=exact(c,1)[0];size=0;factor=1
        for _ in range(4):
            v=exact(c,1)[0];size+=(v&127)*factor
            if v<128:return kind,exact(c,size)
            factor*=128
        raise ValueError('Invalid MQTT length')
    def receive():
        try:
            with mqtt.accept()[0] as c:
                c.settimeout(15);kind,body=packet(c)
                assert kind==16 and body[:7]==b'\x00\x04MQTT\x04'
                c.sendall(b'\x20\x02\x00\x00');kind,body=packet(c);assert kind==48
                size=int.from_bytes(body[:2],'big');assert 0<size<len(body)-2
                captured['mqtt_topic']=body[2:2+size].decode('utf-8')
                captured['mqtt']=json.loads(body[2+size:])
        except Exception as exc:errors.append('MQTT: '+str(exc))
    try:
        http=http_server.HTTPServer(('127.0.0.1',8080),Handler)
        mqtt=socket.socket();mqtt.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        mqtt.bind(('127.0.0.1',1883));mqtt.listen(1);mqtt.settimeout(15)
        worker=threading.Thread(target=receive,daemon=True);worker.start()
        threading.Thread(target=http.serve_forever,daemon=True).start()
        print('[接收端] HTTP POST /api/alarm 与 MQTT 测试服务已启动。')
        negative,background=ctx.example('background')
        print('\n背景输入：'+negative['id'])
        text,_=ctx.run([background,'--report'],background,echo=False)
        if '[ALARM]' in text or '[ALERT] Local alarm triggered:' in text or captured:
            raise RuntimeError('背景负例意外触发告警。')
        print('背景识别完成，上报事件：0。')
        print('\n运行：audiodetect '+wav.name+' --report')
        text,_=ctx.run([wav,'--report'],wav,echo=False);worker.join(16)
        if errors:raise RuntimeError('; '.join(errors))
        if 'HTTP report: SUCCESS' not in text or 'MQTT report: SUCCESS' not in text:raise RuntimeError('发送端未报告两个协议均成功。')
        if captured.get('http')!=captured.get('mqtt') or not captured.get('http'):raise RuntimeError('两个接收端载荷不一致。')
        payload=captured['http'];expected='audiodetect/'+payload['device']+'/alarm'
        if captured.get('mqtt_topic')!=expected:raise RuntimeError('MQTT topic不匹配。')
        print('\n[HTTP实际收到] POST '+captured['http_path']);print(json.dumps(payload,ensure_ascii=False,indent=2))
        print('\n[MQTT实际收到] topic='+captured['mqtt_topic']);print(json.dumps(captured['mqtt'],ensure_ascii=False,indent=2))
        print('\n咳嗽事件已送达 HTTP 和 MQTT 接收端。')
        ctx.save_result('alarm',{'input':row['id'],'background_reports':0,'http':payload,'mqtt':captured['mqtt'],'mqtt_topic':captured['mqtt_topic'],'loopback':True})
    except OSError as exc:raise RuntimeError('测试端口8080/1883不可用；请检查已有服务，脚本不会结束其他程序。 '+str(exc)) from exc
    finally:
        if http:
            # serve_forever might not have started when a later bind failed.
            if worker is not None:http.shutdown()
            http.server_close()
        if mqtt:mqtt.close()
    def records():
        title('实验记录 · Linux 网络回环（2026-10-26）')
        reference=read(ROOT/'results/mlp-study-20261026/linux/auxiliary.json')
        state='成功' if reference['status']=='PASS' else reference['status']
        table(['接口','目标','结果'],[['HTTP POST','127.0.0.1:8080/api/alarm',state],['MQTT PUBLISH',reference['mqtt_topic'],state]])
        print('来源：results/mlp-study-20261026/linux/auxiliary.json')
    ctx.show_records(records)
if __name__=='__main__':raise SystemExit(entry(demo,'HTTP/MQTT本机回环告警独立演示'))
