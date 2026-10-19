"""Fetch the approved official evaluation reference, pinned and hash checked."""
import os,hashlib,json,pathlib,subprocess,urllib.request

REPO=pathlib.Path(__file__).resolve().parents[2]
ROOT=pathlib.Path(os.environ.get('CNN_V2_RUN_DIR',str(REPO/'out/cnn-v2-training'))).resolve()
ROOT.mkdir(parents=True,exist_ok=True)
OUT=ROOT/'artifacts/panns';OUT.mkdir(parents=True,exist_ok=True)
ref='d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4' # exact source used for this experiment
sources={}
for source,name in [('pytorch/models.py','models.py'),('pytorch/pytorch_utils.py','pytorch_utils.py'),('LICENSE.MIT','LICENSE.MIT'),('requirements.txt','upstream-requirements.txt')]:
    url=f'https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/{ref}/{source}'
    data=urllib.request.urlopen(url,timeout=30).read();(OUT/name).write_bytes(data);sources[name]={'url':url,'sha256':hashlib.sha256(data).hexdigest()}
name='Cnn6_mAP=0.343.pth';path=OUT/name;url='https://zenodo.org/records/3987831/files/Cnn6_mAP%3D0.343.pth?download=1'
if not path.exists():
    with urllib.request.urlopen(url,timeout=60) as response,path.with_suffix('.pth.part').open('wb') as dest:
        size=0
        while True:
            chunk=response.read(1024*1024)
            if not chunk:break
            dest.write(chunk);size+=len(chunk)
            if size%(4*1024*1024)==0:print('PANNs bytes',size,flush=True)
    path.with_suffix('.pth.part').replace(path)
data=path.read_bytes();assert hashlib.md5(data).hexdigest()=='e25e26b84585b14c7754c91e48efc9be'
receipt={'repository':'https://github.com/qiuqiangkong/audioset_tagging_cnn','source_commit':ref,'files':sources,
         'checkpoint':{'filename':name,'url':url,'bytes':len(data),'md5':hashlib.md5(data).hexdigest(),'sha256':hashlib.sha256(data).hexdigest()},
         'scope':'Frozen AudioSet-pretrained reference, extra pretraining data; not the deployed lightweight model; not redistributed with the application'}
(OUT/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt['checkpoint']),flush=True)
