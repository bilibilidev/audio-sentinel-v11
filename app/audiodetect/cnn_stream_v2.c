/* Audio Sentinel CNN v2. FFT structure derives from this repository's
 * keyword_detect.c and CNN v1. INT8 weight storage, floating-point arithmetic.
 * Causal DSP/state and final pooling match the Python training implementation.
 */
#include "cnn_stream_v2.h"
#include "cnn_weights_v2.h"
#include <math.h>
#include <string.h>

#define CNNV2_PI 3.14159265358979323846
static void fft(float *real, float *imag)
{
  int j=0;
  for(int i=1;i<512;i++) {
    int bit=256;while(j&bit){j^=bit;bit>>=1;}j^=bit;
    if(i<j){float t=real[i];real[i]=real[j];real[j]=t;t=imag[i];imag[i]=imag[j];imag[j]=t;}
  }
  for(int len=2;len<=512;len<<=1) {
    float angle=(float)(-2.0*CNNV2_PI/len),wr=cosf(angle),wi=sinf(angle);
    for(int i=0;i<512;i+=len) {
      float ar=1,ai=0;
      for(int k=0;k<len/2;k++) {
        int b=i+k+len/2,a=i+k;float tr=ar*real[b]-ai*imag[b],ti=ar*imag[b]+ai*real[b];
        real[b]=real[a]-tr;imag[b]=imag[a]-ti;real[a]+=tr;imag[a]+=ti;
        float nr=ar*wr-ai*wi;ai=ar*wi+ai*wr;ar=nr;
      }
    }
  }
}
const char *cnn_v2_label(int i)
{
  static const char *const labels[5]={"background","cough","glass_break","baby_cry","dog_bark"};
  return i>=0&&i<5?labels[i]:"unknown";
}
void cnn_v2_reset(cnn_v2_state_t *s){memset(s,0,sizeof(*s));}
void cnn_v2_softmax(const float logits[5],float probabilities[5])
{
  float mx=logits[0],sum=0;
  for(int i=1;i<5;i++)if(logits[i]>mx)mx=logits[i];
  for(int i=0;i<5;i++){probabilities[i]=expf(logits[i]-mx);sum+=probabilities[i];}
  for(int i=0;i<5;i++)probabilities[i]/=sum;
}
static void process_frame(cnn_v2_state_t *s,cnn_v2_callback callback,void *context)
{
  float a[CNNV2_SCRATCH_CHANNELS]={0},b[CNNV2_SCRATCH_CHANNELS],mel[26],tim[5];
  (void)mel; (void)tim; /* Feature subsets compile without unused warnings. */
  float energy=0,absolute=0,peak=0;int crossings=0;float previous=0;
  memset(s->fft_real,0,sizeof(s->fft_real));memset(s->fft_imag,0,sizeof(s->fft_imag));
  for(int i=0;i<400;i++) {
    int position=(s->audio_pos+i)%400;float raw=s->raw[position];
    s->fft_real[i]=s->audio[position]*cnnv2_hamming[i];
    energy+=raw*raw;absolute+=fabsf(raw);if(fabsf(raw)>peak)peak=fabsf(raw);
    if(i>0&&(raw<0)!=(previous<0))crossings++;
    previous=raw;
  }
  energy/=400;absolute/=400;
  tim[0]=logf(fmaxf(energy,1e-10f));tim[1]=crossings/399.0f;
  tim[2]=logf(fmaxf(absolute,1e-10f));tim[3]=logf(fmaxf(peak,1e-10f));
  tim[4]=logf(fmaxf(s->frames?fabsf(energy-s->previous_energy):0,1e-10f));s->previous_energy=energy;
  fft(s->fft_real,s->fft_imag);
  for(int i=0;i<257;i++)s->power[i]=s->fft_real[i]*s->fft_real[i]+s->fft_imag[i]*s->fft_imag[i];
  for(int m=0;m<26;m++) {
    float value=0;for(int i=0;i<257;i++)value+=s->power[i]*cnnv2_mel[m*257+i];
    mel[m]=logf(fmaxf(value,1e-10f));
  }
#if CNNV2_FEATURE_KIND == 0 || CNNV2_FEATURE_KIND == 4
  memcpy(a,mel,CNNV2_INPUT_DIM*sizeof(float));
#elif CNNV2_FEATURE_KIND == 1
  memcpy(a,tim,sizeof(tim));
#elif CNNV2_FEATURE_KIND == 2
  for(int i=0;i<13;i++)for(int j=0;j<26;j++)a[i]+=cnnv2_dct[i*26+j]*mel[j];
#else
  memcpy(a,mel,sizeof(mel));memcpy(a+26,tim,sizeof(tim));
#endif
  for(int i=0;i<CNNV2_INPUT_DIM;i++)a[i]=(a[i]-cnnv2_mean[i])/cnnv2_std[i];
  for(int layer=0;layer<CNNV2_LAYER_COUNT;layer++) {
    const cnnv2_layer_t *p=&cnnv2_layers[layer];int pos=s->positions[layer];float *history=s->history+p->offset;
    for(int ic=0;ic<p->in_channels;ic++)history[ic*p->history+pos]=a[ic];
    for(int oc=0;oc<p->out_channels;oc++) {
      float value=(float)p->b[oc]*p->bs;
      for(int ic=0;ic<p->in_channels;ic++)for(int k=0;k<p->kernel;k++) {
        int t=pos-p->dilation*(p->kernel-1-k);if(t<0)t+=p->history;
        value+=history[ic*p->history+t]*(float)p->w[(oc*p->in_channels+ic)*p->kernel+k]*p->ws;
      }
      b[oc]=layer<CNNV2_LAYER_COUNT-1?fmaxf(value,0):value;
    }
    s->positions[layer]=(pos+1)%p->history;memcpy(a,b,(size_t)p->out_channels*sizeof(float));
  }
  for(int k=0;k<5;k++) {
    s->logits[k]=a[k];s->sums[k]+=a[k];s->ema[k]=s->frames?.98f*s->ema[k]+.02f*a[k]:a[k];
    if(s->frames==0){s->logsumexp_max[k]=a[k];s->sum_exp[k]=1;}
    else if(a[k]<=s->logsumexp_max[k])s->sum_exp[k]+=exp((double)a[k]-s->logsumexp_max[k]);
    else {s->sum_exp[k]=s->sum_exp[k]*exp((double)s->logsumexp_max[k]-a[k])+1;s->logsumexp_max[k]=a[k];}
  }
  s->frames++;if(callback)callback(s,context);
}
void cnn_v2_feed(cnn_v2_state_t *s,const int16_t *pcm,size_t count,cnn_v2_callback callback,void *context)
{
  for(size_t i=0;i<count;i++) {
    float raw=(float)pcm[i]/32768.0f,emphasized=s->samples?raw-.97f*s->previous_raw:raw;
    s->previous_raw=raw;s->energy_sum+=(double)raw*raw;
    s->audio[s->audio_pos]=emphasized;s->raw[s->audio_pos]=raw;s->audio_pos=(s->audio_pos+1)%400;s->samples++;
    if(s->samples>=400&&(s->samples-400)%160==0)process_frame(s,callback,context);
  }
}
void cnn_v2_finish(cnn_v2_state_t *s,cnn_v2_callback callback,void *context)
{
  if(s->frames==0&&s->samples>0){int16_t zero[400]={0};s->previous_raw=0;cnn_v2_feed(s,zero,400-(size_t)s->samples,callback,context);}
}
void cnn_v2_result(const cnn_v2_state_t *s,float probabilities[5])
{
  float logits[5]={0};
  for(int k=0;k<5;k++)if(s->frames) {
    double mean=s->sums[k]/s->frames;
#if CNNV2_MIXED_LME
    logits[k]=(float)(.5*mean+.5*(s->logsumexp_max[k]+log(s->sum_exp[k]/s->frames)));
#else
    logits[k]=(float)mean;
#endif
  }
  cnn_v2_softmax(logits,probabilities);
}
