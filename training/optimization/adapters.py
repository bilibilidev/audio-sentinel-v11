"""Experimental preprocessing adapters, never modifying the deployed App."""
import difflib
import numpy as np

def energy_full(features):
    out=features.copy();energy=out[26:27].copy()
    out[:26]-=energy;out[26]=0;out[28:30]-=.5*energy;out[30:31]-=energy
    return out

def pool_full(features,n):
    return np.stack([features[...,t:t+n].mean(axis=-1) for t in range(0,features.shape[-1],n)],axis=-1)

def adapt_energy(directory):
    source=directory/'cnn_stream_v2.c';old=source.read_text(encoding='utf-8')
    anchor='#if CNNV2_FEATURE_KIND == 0 || CNNV2_FEATURE_KIND == 4'
    assert old.count(anchor)==1
    prefix='''  /* Instantaneous energy normalization, no file-wide statistics. */
  float log_energy=tim[0];
  for(int m=0;m<26;m++)mel[m]-=log_energy;
  tim[0]=0;tim[2]-=.5f*log_energy;tim[3]-=.5f*log_energy;tim[4]-=log_energy;
'''
    new=old.replace(anchor,prefix+anchor);source.write_text(new,encoding='utf-8',newline='\n')
    (directory/'energy-adapter.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original/cnn_stream_v2.c',tofile='energy/cnn_stream_v2.c')),encoding='utf-8',newline='\n')

def adapt_temporal(directory,n):
    """Pool observed DSP frames before the shared causal CNN; flush only at EOF."""
    assert n in (2,4)
    paths=[directory/'cnn_stream_v2.h',directory/'cnn_stream_v2.c',directory/'harness.c']
    old=[p.read_text(encoding='utf-8') for p in paths];h,c,cli=old
    h=h.replace('uint64_t samples, frames;','uint64_t samples, frames, dsp_frames;\n  unsigned pool_pending;\n  float feature_sum[CNNV2_INPUT_DIM];')
    start=c.index('  for(int i=0;i<CNNV2_INPUT_DIM;i++)a[i]=')
    end=c.index('\nvoid cnn_v2_feed',start)
    neural=c[start:end]
    neural=neural.replace('.98f*s->ema[k]+.02f*a[k]',f'{.98**n:.12g}f*s->ema[k]+{1-.98**n:.12g}f*a[k]')
    helper='''static void neural_frame(cnn_v2_state_t *s,float a[CNNV2_SCRATCH_CHANNELS],cnn_v2_callback callback,void *context)
{
  float b[CNNV2_SCRATCH_CHANNELS];
'''+neural
    replacement=f'''  s->dsp_frames++;
  for(int i=0;i<CNNV2_INPUT_DIM;i++)s->feature_sum[i]+=a[i];
  s->pool_pending++;
  if(s->pool_pending=={n}) {{
    for(int i=0;i<CNNV2_INPUT_DIM;i++){{a[i]=s->feature_sum[i]/s->pool_pending;s->feature_sum[i]=0;}}
    s->pool_pending=0;neural_frame(s,a,callback,context);
  }}
}}
'''
    c=c[:start]+replacement+c[end:]
    pos=c.index('static void process_frame(');c=c[:pos]+helper+c[pos:]
    c=c.replace('a[CNNV2_SCRATCH_CHANNELS]={0},b[CNNV2_SCRATCH_CHANNELS],mel[26]','a[CNNV2_SCRATCH_CHANNELS]={0},mel[26]')
    c=c.replace('s->frames?fabsf(energy-s->previous_energy)','s->dsp_frames?fabsf(energy-s->previous_energy)')
    c=c.replace('if(s->frames==0&&s->samples>0)','if(s->dsp_frames==0&&s->samples>0)')
    finish='cnn_v2_feed(s,zero,400-(size_t)s->samples,callback,context);}'
    assert c.count(finish)==1
    c=c.replace(finish,finish+'''
  if(s->pool_pending){
    float a[CNNV2_SCRATCH_CHANNELS]={0};
    for(int i=0;i<CNNV2_INPUT_DIM;i++){a[i]=s->feature_sum[i]/s->pool_pending;s->feature_sum[i]=0;}
    s->pool_pending=0;neural_frame(s,a,callback,context);
  }''')
    cli=cli.replace('s->frames%10==0',f's->frames%{(10+n-1)//n}==0')
    new=[h,c,cli];patch=''
    for p,before,after in zip(paths,old,new):
        p.write_text(after,encoding='utf-8',newline='\n')
        patch+=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='unpooled/'+p.name,tofile='temporal/'+p.name))
    (directory/'temporal-adapter.diff').write_text(patch,encoding='utf-8',newline='\n')

def pcen_full(features):
    out=features.copy();power=np.exp(out[:26]).astype(np.float32);smooth=power[:,0].copy()
    for t in range(power.shape[-1]):
        if t:smooth=np.float32(.975)*smooth+np.float32(.025)*power[:,t]
        value=power[:,t]/np.power(np.float32(1e-6)+smooth,np.float32(.98))
        out[:26,t]=np.sqrt(value+np.float32(2))-np.float32(np.sqrt(2))
    return out

def adapt_pcen(directory):
    """Retain PCEN state in a private copy of the existing causal DSP core."""
    header=directory/'cnn_stream_v2.h';source=directory/'cnn_stream_v2.c'
    old_h=header.read_text(encoding='utf-8');old_c=source.read_text(encoding='utf-8')
    anchor='  float previous_raw, previous_energy;'
    assert old_h.count(anchor)==1
    new_h=old_h.replace(anchor,anchor+'\n  float pcen_smooth[26]; /* Causal PCEN M0=S0; reset with stream. */')
    anchor='    mel[m]=logf(fmaxf(value,1e-10f));'
    assert old_c.count(anchor)==1
    replacement='''    /* Training cache uses exp(log(mel power)); retain the same convention. */
    value=expf(logf(fmaxf(value,1e-10f)));
    s->pcen_smooth[m]=s->frames ? .975f*s->pcen_smooth[m]+.025f*value : value;
    mel[m]=sqrtf(value/powf(1e-6f+s->pcen_smooth[m],.98f)+2.0f)-sqrtf(2.0f);'''
    new_c=old_c.replace(anchor,replacement)
    header.write_text(new_h,encoding='utf-8',newline='\n');source.write_text(new_c,encoding='utf-8',newline='\n')
    patch=''.join(difflib.unified_diff(old_h.splitlines(True),new_h.splitlines(True),fromfile='original/cnn_stream_v2.h',tofile='pcen/cnn_stream_v2.h'))
    patch+=''.join(difflib.unified_diff(old_c.splitlines(True),new_c.splitlines(True),fromfile='original/cnn_stream_v2.c',tofile='pcen/cnn_stream_v2.c'))
    (directory/'pcen-adapter.diff').write_text(patch,encoding='utf-8',newline='\n')

def centered_full(features,bins):
    out=features.copy();out[:bins]-=out[:bins].mean(axis=0,keepdims=True);return out

def adapt_centered(directory,bins):
    assert bins in (19,26)
    source=directory/'cnn_stream_v2.c';old=source.read_text(encoding='utf-8')
    anchor='#if CNNV2_FEATURE_KIND == 0 || CNNV2_FEATURE_KIND == 4'
    assert old.count(anchor)==1
    prefix=f'''  /* Per-frame centering uses no future audio and no file statistics. */
  float spectral_mean=0;
  for(int m=0;m<{bins};m++)spectral_mean+=mel[m];
  spectral_mean/={bins};
  for(int m=0;m<{bins};m++)mel[m]-=spectral_mean;
'''
    new=old.replace(anchor,prefix+anchor);source.write_text(new,encoding='utf-8',newline='\n')
    patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original/cnn_stream_v2.c',tofile='centered/cnn_stream_v2.c'))
    (directory/'spectral-center-adapter.diff').write_text(patch,encoding='utf-8',newline='\n')
