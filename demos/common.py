"""Linux-only demo orchestration; inference uses the shared native C app."""
import argparse,hashlib,json,os,pathlib,platform,re,shutil,subprocess,sys,time,unicodedata
ROOT=pathlib.Path(__file__).resolve().parents[1]
LABELS=['background','cough','glass_break','baby_cry','dog_bark']
NAMES=dict(zip(LABELS,['日常背景','咳嗽','玻璃破碎','婴儿哭声','狗叫']))
SOURCES=['audiodetect_main.c','audio_classifier.c','wav_reader.c','keyword_detect.c','alarm_report.c','feature_extractor.c','mlp_model_infer.c']
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def title(text):print('\n'+'='*70+'\n'+text+'\n'+'='*70,flush=True)

def table(headers,rows,right=()):
    """Align CJK text in a monospace terminal without external packages."""
    def width(s):
        return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)
    cells=[[str(c) for c in row] for row in [headers,*rows]]
    sizes=[max(width(row[i]) for row in cells) for i in range(len(headers))]
    border='+'+'+'.join('-'*(n+2) for n in sizes)+'+'
    print(border)
    for n,row in enumerate(cells):
        output=[]
        for i,c in enumerate(row):
            padding=' '*(sizes[i]-width(c))
            output.append(' '+(padding+c if i in right and n else c+padding)+' ')
        print('|'+'|'.join(output)+'|')
        if n==0:print(border)
    print(border,flush=True)
def pause(ctx,text='按 Enter 继续；Ctrl+C 退出'):
    if not ctx.args.auto and sys.stdin.isatty():input('\n'+text+' ')
def arguments(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--audio-dir',type=pathlib.Path)
    p.add_argument('--keyword-dir',type=pathlib.Path)
    p.add_argument('--cc');p.add_argument('--binary',type=pathlib.Path)
    p.add_argument('--auto',action='store_true',help='Run without Enter pauses')
    p.add_argument('--play',action='store_true',help='Play WAVs on the Linux guest using an existing aplay')
    p.add_argument('--probe',action='store_true',help='Also run the first-second single-WAV noise prototype')
    p.add_argument('--build-only',action='store_true',help='Build and verify the current Linux C ELF, then exit')
    p.add_argument('--section',choices=['menu','all','classify','robust','model','alarm','keyword'],default='menu',help='Select a highlight; default shows the five-entry menu')
    return p.parse_args()
class Context:
    def __init__(self,args):
        if sys.platform!='linux':
            raise RuntimeError('本演示要求 Linux。请用 VS Code Remote SSH 连接 Ubuntu，在远程终端运行 bash demos/run_demo.sh。')
        self.args=args;local=ROOT/'demos/local_paths.json';cfg=read(local) if local.exists() else {}
        def path(value,key,env):
            raw=value or os.environ.get(env) or cfg.get(key)
            if not raw:return None
            p=pathlib.Path(raw).expanduser();return (ROOT/p).resolve() if not p.is_absolute() else p.resolve()
        self.audio_dir=path(args.audio_dir,'audio_dir','AUDIO_SENTINEL_AUDIO_DIR')
        self.keyword_dir=path(args.keyword_dir,'keyword_dir','AUDIO_SENTINEL_KEYWORD_DIR')
        self.model=read(ROOT/'model/mlp-optimized-20261024/deployment.json')
        assert sha(ROOT/'model/mlp-optimized-20261024/final_all_esc50.npz')==self.model['model_sha256']
        for name,digest in self.model['header_sha256'].items():assert sha(ROOT/'app/audiodetect'/name)==digest
        self.manifest=read(ROOT/'results/board-mlp-20261027/input-manifest.json')
        self.predictions=read(ROOT/'results/board-mlp-20261027/clean65.json')['records']
        self.binary=path(args.binary,'binary','AUDIO_SENTINEL_BINARY')
        self.compiler=args.cc or cfg.get('cc') or os.environ.get('CC') or 'gcc'
        if self.binary:
            if not self.binary.is_file():raise RuntimeError('指定的 C 程序不存在。')
    def ensure_binary(self):
        if self.binary is None:self.binary=self.build(self.compiler)
        data=self.binary.read_bytes()
        if not data.startswith(b'\x7fELF'):raise RuntimeError('需要 Linux ELF 程序，拒绝执行 Windows EXE 或其他文件。')
        if self.model['model_sha256'].encode() not in data:
            raise RuntimeError('C 程序没有当前正式模型标识；拒绝使用旧模型。')
    def build(self,cc):
        compiler=shutil.which(cc) or (cc if pathlib.Path(cc).is_file() else None)
        if not compiler:raise RuntimeError('找不到现有 C 编译器。用 --cc 指定 GCC，或用 --binary 指定当前 Tiny MLP 程序。脚本不安装软件。')
        app=ROOT/'app/audiodetect';out=ROOT/'out/linux-demo';out.mkdir(parents=True,exist_ok=True)
        target=out/'audiodetect';stamp=out/'build.json'
        script=ROOT/'scripts/build_audio_sentinel_app.sh'
        files=sorted([app/s for s in SOURCES]+list(app.glob('*.h'))+[script])
        fingerprint=hashlib.sha256((''.join(sha(p) for p in files)+str(compiler)).encode()).hexdigest()
        if target.is_file() and stamp.is_file() and read(stamp).get('source_fingerprint')==fingerprint and read(stamp).get('binary_sha256')==sha(target):return target
        print('[编译] Linux GCC / C11；调用 scripts/build_audio_sentinel_app.sh',flush=True)
        cmd=['sh',str(script)]
        env=os.environ.copy();env['PATH']=str(pathlib.Path(compiler).resolve().parent)+os.pathsep+env.get('PATH','')
        env.update(CC=str(pathlib.Path(compiler).resolve()),OUT_DIR=str(out),DEFAULT_MODEL='mlp')
        r=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,timeout=120)
        (out/'build.log').write_bytes(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('编译失败，详见 out/linux-demo/build.log。')
        version=subprocess.check_output([str(compiler),'--version'],text=True).splitlines()[0]
        stamp.write_text(json.dumps(dict(source_fingerprint=fingerprint,binary_sha256=sha(target),model_sha256=self.model['model_sha256'],compiler=version,system=platform.platform(),command='CC='+str(compiler)+' OUT_DIR=out/linux-demo DEFAULT_MODEL=mlp sh scripts/build_audio_sentinel_app.sh'),indent=2),encoding='utf-8')
        return target
    def wav(self,row):
        if not self.audio_dir:raise RuntimeError('缺少音频目录。请配置 demos/local_paths.json，或指定 --audio-dir。音频不随 GitHub 分发。')
        i=next(i for i,r in enumerate(self.manifest['records']) if r['id']==row['id'])
        for p in [self.audio_dir/(row['id']+'.wav'),self.audio_dir/row['path'],self.audio_dir/row['board_file'],self.audio_dir/f'{i:03}.wav']:
            p=p.resolve()
            if p.is_relative_to(self.audio_dir) and p.is_file() and sha(p)==row['sha256']:return p
        raise RuntimeError('找不到与冻结 SHA256 一致的录音：'+row['id'])
    def example(self,label,alarm=False):
        for result in self.predictions:
            if result['truth']==label and result['correct'] and (not alarm or result['confidence']>.90):
                row=next(r for r in self.manifest['records'] if r['id']==result['id'])
                try:return row,self.wav(row)
                except RuntimeError:continue
        raise RuntimeError('缺少该类别所需的已确认示例：'+label)
    def play(self,p):
        if self.args.play:
            player=shutil.which('aplay')
            if not player:raise RuntimeError('来宾中没有现成 aplay；请去掉 --play，脚本不安装软件。')
            print('[播放] 使用 Ubuntu 来宾音频设备；SSH 不会将音频转发到 Windows。')
            if subprocess.run([player,'-q',str(p)],timeout=90).returncode:
                raise RuntimeError('Ubuntu 来宾音频播放失败。')
    def run(self,args,wav=None,require_model=True):
        self.ensure_binary()
        start=time.perf_counter();r=subprocess.run([str(self.binary),*map(str,args)],capture_output=True,timeout=45)
        elapsed=(time.perf_counter()-start)*1000
        text=r.stdout.decode('utf-8',errors='replace');err=r.stderr.decode('utf-8',errors='replace')
        if r.returncode:raise RuntimeError((text+err).replace(str(ROOT),'${PROJECT}')[-1500:])
        if require_model and self.model['model_sha256'] not in text:raise RuntimeError('运行模型不匹配。')
        # Hide machine-local input paths from the recording.
        text=text.replace('\r\r\n','\n').replace('\r\n','\n')
        if wav:
            text=re.sub(r'^Input:.*$',lambda _: 'Input: '+wav.name,text,flags=re.M)
            text=text.replace(str(wav),wav.name)
        print(text.strip(),flush=True)
        return text,elapsed
def entry(fn,description):
    for s in [sys.stdout,sys.stderr]:
        if hasattr(s,'reconfigure'):s.reconfigure(encoding='utf-8')
    try:
        ctx=Context(arguments(description))
        print('[运行端] Linux '+platform.machine()+'；Python仅编排/制表，音频推理由原生C ELF执行。')
        if ctx.args.build_only:
            ctx.ensure_binary()
            print('Linux C ELF: '+str(ctx.binary.relative_to(ROOT) if ctx.binary.is_relative_to(ROOT) else ctx.binary.name))
            print('Model SHA256: '+ctx.model['model_sha256'])
        else:fn(ctx)
    except KeyboardInterrupt:print('\n已退出演示。');return 130
    except (RuntimeError,AssertionError,OSError,subprocess.TimeoutExpired) as exc:
        print('\n[未完成] '+str(exc));return 1
    return 0
