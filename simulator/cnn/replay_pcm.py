"""Replay an existing PCM16 mono 16k WAV at its sample rate; never record a mic."""
import argparse,sys,time,wave
parser=argparse.ArgumentParser();parser.add_argument('wav');args=parser.parse_args()
with wave.open(args.wav,'rb') as source:
    if (source.getnchannels(),source.getframerate(),source.getsampwidth())!=(1,16000,2):
        raise SystemExit('Expected PCM16 mono 16 kHz WAV')
    start=time.monotonic();sent=0
    try:
        while True:
            data=source.readframes(160)
            if not data:break
            sent+=len(data)//2;delay=start+sent/16000-time.monotonic()
            if delay>0:time.sleep(delay)
            sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
    except BrokenPipeError:
        pass
