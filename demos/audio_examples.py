"""Small PCM16 example utilities; Python standard library only."""
import array,math,random,sys,wave

def read_pcm(path):
    with wave.open(str(path),'rb') as f:
        if (f.getnchannels(),f.getsampwidth(),f.getframerate())!=(1,2,16000):
            raise RuntimeError('示例音频需要16 kHz、PCM16、单声道格式。')
        pcm=array.array('h',f.readframes(f.getnframes()))
    if sys.byteorder!='little':pcm.byteswap()
    if not pcm:raise RuntimeError('示例录音为空。')
    return pcm

def write_pcm(path,pcm):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=array.array('h',pcm)
    if sys.byteorder!='little':data.byteswap()
    with wave.open(str(path),'wb') as f:
        f.setparams((1,2,16000,0,'NONE','not compressed'));f.writeframes(data.tobytes())

def add_gaussian(pcm,snr_db,seed):
    rng=random.Random(seed);signal_energy=sum(v*v for v in pcm)
    noise=[rng.gauss(0.,1.) for _ in pcm]
    noise_energy=sum(v*v for v in noise)
    scale=math.sqrt(signal_energy/(10**(snr_db/10)*noise_energy)) if signal_energy else 0.
    result=array.array('h',(max(-32768,min(32767,round(v+n*scale))) for v,n in zip(pcm,noise)))
    error=sum((int(a)-int(b))**2 for a,b in zip(result,pcm))
    actual=10*math.log10(signal_energy/error) if signal_energy and error else None
    return result,actual
