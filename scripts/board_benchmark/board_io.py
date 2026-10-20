"""Prompt-synchronized NuttX ADB access; no flashing functionality."""
import os,pathlib,queue,re,subprocess,threading,time
ADB=os.environ.get('ADB','adb')
SERIAL=os.environ.get('ANDROID_SERIAL','1234')
class Nsh:
    def __init__(self,log):
        self.log=pathlib.Path(log).open('ab');self.q=queue.Queue()
        self.p=subprocess.Popen([ADB,'-s',SERIAL,'shell'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        def read():
            import os
            while True:
                b=os.read(self.p.stdout.fileno(),4096)
                if not b:self.q.put(None);return
                self.q.put(b)
        threading.Thread(target=read,daemon=True).start()
        try:self.prompt(15)
        except Exception:
            self.p.kill();self.p.wait();self.log.close();raise
    def prompt(self,seconds):
        deadline=time.monotonic()+seconds;data=b''
        while time.monotonic()<deadline:
            try:b=self.q.get(timeout=min(1,max(.001,deadline-time.monotonic())))
            except queue.Empty:continue
            if b is None:raise RuntimeError('ADB shell closed before prompt: '+repr(data[-500:]))
            data+=b;self.log.write(b);self.log.flush()
            if re.search(rb'(?:^|[\r\n])nsh>',data):return data.decode('utf-8',errors='replace')
        raise TimeoutError('NuttX prompt timeout: '+repr(data[-1000:]))
    def run(self,command,timeout=30):
        assert '\n' not in command and '\r' not in command
        assert len(command.encode('utf-8'))<=60,'Old board ADB shell fragments long commands; use short paths'
        self.p.stdin.write((command+'\n').encode());self.p.stdin.flush();text=self.prompt(timeout)
        self.p.stdin.write(b'echo AED_RC $?\n');self.p.stdin.flush();status=self.prompt(10)
        found=re.findall(r'AED_RC\s+(-?\d+)',status)
        if not found:raise RuntimeError('Missing NuttX exit status: '+status)
        return text,int(found[-1])
    def close(self):
        try:
            self.p.stdin.write(b'exit\n');self.p.stdin.flush();self.p.wait(timeout=3)
        except (OSError,subprocess.TimeoutExpired):self.p.kill();self.p.wait()
        self.log.close()

def adb(args,log,timeout=120):
    # The bundled 2018 Windows ADB has inconsistent Unicode absolute-path handling.
    # All host files are inside the current workspace; pass ASCII relative paths.
    import os
    args=list(args)
    if args[0]=='pull':args[-1]=os.path.relpath(args[-1])
    if args[0]=='push':args[1]=os.path.relpath(args[1])
    p=subprocess.run([ADB,'-s',SERIAL,*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    pathlib.Path(log).write_bytes(p.stdout)
    if p.returncode:raise RuntimeError(p.stdout.decode('utf-8',errors='replace')[-2000:])
    return p.stdout.decode('utf-8',errors='replace')
