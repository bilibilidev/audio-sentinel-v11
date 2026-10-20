"""Single-thread CPU resource comparison, without training CNN weights.

Run once per model in a fresh process. This is a host benchmark, not a
PANNs deployment or power measurement on NuttX. Inputs are in-memory PCM;
file IO and classifier fitting are excluded from timing.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse,ctypes,hashlib,json,pathlib,platform,sys,time
import numpy as np
import torch

REPO=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'training'),str(REPO/'training/cnn_v2')]
from cnn_v2 import load,all_features,select_features

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def memory():
    if os.name!='nt':
        import resource
        return dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),
                  *[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage',
                    'PagefileUsage','PeakPagefileUsage','PrivateUsage')]]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=ctypes.c_void_p
    fn=ctypes.WinDLL('psapi').GetProcessMemoryInfo
    fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    if not fn(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):raise ctypes.WinError()
    return dict(rss_bytes=counters.WorkingSetSize,peak_rss_bytes=counters.PeakWorkingSetSize,private_bytes=counters.PrivateUsage)

def stats(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(count=len(values),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),min=float(a.min()),max=float(a.max()),raw_seconds=values)

def timing(fn,repeats):
    for _ in range(3):fn()
    values=[]
    for _ in range(repeats):
        start=time.perf_counter();fn();values.append(time.perf_counter()-start)
    return stats(values)

def mac_profile(model,x):
    counts=dict(learned_network_conv_linear_macs=0,fixed_frontend_conv_macs=0,largest_single_module_output_bytes=0)
    hooks=[]
    def hook(name,module,inputs,output):
        if not isinstance(output,torch.Tensor):return
        counts['largest_single_module_output_bytes']=max(counts['largest_single_module_output_bytes'],output.numel()*output.element_size())
        if isinstance(module,(torch.nn.Conv1d,torch.nn.Conv2d)):
            macs=output.numel()*module.weight[0].numel()
        elif isinstance(module,torch.nn.Linear):macs=output.numel()*module.in_features
        else:return
        key='fixed_frontend_conv_macs' if name.startswith(('spectrogram_extractor','logmel_extractor')) else 'learned_network_conv_linear_macs'
        counts[key]+=int(macs)
    for name,module in model.named_modules():
        if isinstance(module,(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Linear)):
            hooks.append(module.register_forward_hook(lambda m,i,o,n=name:hook(n,m,i,o)))
    model(x)
    for h in hooks:h.remove()
    counts['scope']='Conv/Linear multiply-accumulates only; excludes bias, normalization, nonlinearities, pooling, FFT/mel matrix operations outside Conv/Linear, and data movement. One MAC is one multiply-accumulate.'
    return counts

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['cnn','panns'],required=True)
    p.add_argument('--audio',type=pathlib.Path,required=True)
    p.add_argument('--panns-dir',type=pathlib.Path)
    p.add_argument('--embedding-cache',type=pathlib.Path)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--repeats',type=int,default=10)
    args=p.parse_args();assert not args.output.exists(),'Preserve earlier results'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from audio_utils import read_wav_mono
    pcm=read_wav_mono(args.audio,16000)
    assert len(pcm)>=160000,'Use an actual recording at least 10 seconds long'
    before=memory();classifier=None
    if args.model=='cnn':
        model_path=REPO/'model/cnn-v2/final_all_esc50.npz'
        deployment=json.loads((REPO/'model/cnn-v2/deployment.json').read_text())
        assert sha(model_path)==deployment['model_sha256']
        model,mean,std=load(model_path,quantized=True)
        with np.load(model_path,allow_pickle=False) as z:
            qbytes=sum(z[k].nbytes for k in z.files if k.startswith(('QW','QB')))
            scales=sum(z[k].nbytes for k in z.files if k.startswith(('WS','BS')))
        profile=dict(model_sha256=sha(model_path),learned_parameters=sum(t.numel() for t in model.parameters()),
                     fixed_frontend_parameters=0,int8_learned_parameter_bytes=qbytes,quantization_scale_bytes=scales,
                     npz_archive_bytes=model_path.stat().st_size,torch_parameter_bytes=sum(t.numel()*t.element_size() for t in model.parameters()))
        def prepare(a):
            f=select_features(all_features(a),'logmel')
            return torch.from_numpy(((f[None]-mean)/std).astype(np.float32))
        def infer(x):return torch.softmax(model(x),dim=1).numpy()
        def pipeline(a):return infer(prepare(a))
    else:
        assert args.panns_dir and args.embedding_cache
        provenance=json.loads((args.panns_dir/'provenance.json').read_text())
        for name in ('models.py','pytorch_utils.py'):
            assert sha(args.panns_dir/name)==provenance['files'][name]['sha256']
        checkpoint=args.panns_dir/provenance['checkpoint']['filename']
        assert sha(checkpoint)==provenance['checkpoint']['sha256']
        sys.path.insert(0,str(args.panns_dir.resolve()))
        from models import Cnn6
        import librosa
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        model=Cnn6(32000,1024,320,64,50,14000,527)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
        model.eval()
        learned=sum(t.numel() for n,t in model.named_parameters() if not n.startswith(('spectrogram_extractor','logmel_extractor')))
        fixed=sum(t.numel() for n,t in model.named_parameters() if n.startswith(('spectrogram_extractor','logmel_extractor')))
        baselines=json.loads((REPO/'results/cnn-v2/baselines.json').read_text())
        ref=next(r for r in baselines['baselines'] if r['baseline']=='PANNs_Cnn6_linearSVM')
        selection=json.loads((REPO/'training/cnn/esc50-selection.json').read_text())['records']
        with np.load(args.embedding_cache,allow_pickle=False) as z:features=z['X'].copy()
        assert features.shape==(455,512)
        y=np.asarray([r['label_id'] for r in selection]);assert len(y)==390
        classifier=make_pipeline(StandardScaler(),LinearSVC(**ref['final_parameters'],class_weight='balanced',random_state=42,max_iter=10000))
        classifier.fit(features[:390],y)
        labels=['background','cough','glass_break','baby_cry','dog_bark']
        predictions=[labels[int(v)] for v in classifier.predict(features[390:])]
        assert predictions==[r['prediction'] for r in ref['external_final']['predictions']]
        linear=classifier[-1]
        profile=dict(checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
                     learned_parameters=learned,fixed_frontend_parameters=fixed,total_parameters=learned+fixed,
                     learned_parameter_fp32_bytes=learned*4,state_dict_tensor_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values()),
                     linear_svm_parameters=int(linear.coef_.size+linear.intercept_.size),
                     svm_and_scaler_array_bytes=int(sum(a.nbytes for a in (linear.coef_,linear.intercept_,classifier[0].mean_,classifier[0].scale_,classifier[0].var_))),
                     classifier_reconstructed_from_cached_esc50_features=True,classifier_cached_external_predictions_match=True,
                     cnn_training_performed=False,embedding_cache_sha256=sha(args.embedding_cache),extra_pretraining='AudioSet')
        def prepare(a):
            f=librosa.resample(a.astype(np.float32)/32768,orig_sr=16000,target_sr=32000)
            return torch.from_numpy(f[None])
        def infer(x):
            embedding=model(x)['embedding'].numpy()
            return classifier.predict(embedding)
        def pipeline(a):return infer(prepare(a))
    model.eval()
    loaded=memory();rows=[]
    with torch.inference_mode():
        for seconds in (3,5,10):
            segment=pcm[:seconds*16000].copy();x=prepare(segment)
            profile_macs=mac_profile(model,x)
            if classifier is not None:profile_macs['linear_svm_macs']=int(classifier[-1].coef_.size)
            core=timing(lambda:infer(x),args.repeats)
            full=timing(lambda:pipeline(segment),args.repeats)
            row=dict(audio_seconds=seconds,pcm16_sha256=hashlib.sha256(segment.tobytes()).hexdigest(),
                     model_input_shape=list(x.shape),prepared_input_inference=core,pcm_preprocess_and_inference=full,
                     full_pipeline_rtf=full['mean']/seconds,operation_profile=profile_macs)
            rows.append(row)
            print(args.model,seconds,'seconds: core=%.6fs full=%.6fs'%(core['mean'],full['mean']),flush=True)
    result=dict(benchmark='SINGLE_THREAD_HOST_CPU_NOT_BOARD_POWER',model=args.model,
                platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),logical_cpus=os.cpu_count(),
                torch_version=torch.__version__,numpy_version=np.__version__,torch_threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(),mkldnn_enabled=torch.backends.mkldnn.enabled,
                input_recording_sha256=sha(args.audio),profile=profile,
                process_memory_before_model=before,process_memory_loaded=loaded,process_memory_after_benchmark=memory(),
                process_memory_scope='OS process working set/peak; includes Python, PyTorch, libraries, caches and model. Fresh process per model; not board RAM and not activation-only memory.',
                timing_scope='Batch=1, one CPU thread, three warmups then repeated calls; inputs in RAM; full pipeline includes per-model PCM frontend/resampling and inference; excludes file IO, model load, classifier fitting and transfers.',
                power_watts_measured=False,energy_joules_measured=False,board_panns_executed=False,measurements=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
