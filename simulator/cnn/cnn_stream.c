/* Candidate-only causal CNN. FFT structure follows current v11 keyword_detect.c.
 * Original deployed feature extraction and models are not modified.
 */
#include "cnn_stream.h"
#include <math.h>
#include <string.h>
#define PI 3.14159265358979323846
static void fft(float *real,float *imag) {
  int j=0;
  for(int i=1;i<512;i++) {
    int bit=256;while(j&bit){j^=bit;bit>>=1;}j^=bit;
    if(i<j){float t=real[i];real[i]=real[j];real[j]=t;t=imag[i];imag[i]=imag[j];imag[j]=t;}
  }
  for(int len=2;len<=512;len<<=1) {
    float angle=(float)(-2.0*PI/len),wr=cosf(angle),wi=sinf(angle);
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
void cnn_softmax(const float logits[5],float probabilities[5]) {
  float mx=logits[0],sum=0;
  for(int i=1;i<5;i++)if(logits[i]>mx)mx=logits[i];
  for(int i=0;i<5;i++){probabilities[i]=expf(logits[i]-mx);sum+=probabilities[i];}
  for(int i=0;i<5;i++)probabilities[i]/=sum;
}
void cnn_reset(cnn_state_t *s){memset(s,0,sizeof(*s));}
static void process_frame(cnn_state_t *s,cnn_frame_callback callback,void *context) {
  float a[26],b[26];
  memset(s->fft_real,0,sizeof(s->fft_real));memset(s->fft_imag,0,sizeof(s->fft_imag));
  for(int i=0;i<400;i++)s->fft_real[i]=s->audio[(s->audio_pos+i)%400]*cnn_hamming[i];
  fft(s->fft_real,s->fft_imag);
  for(int i=0;i<257;i++)s->power[i]=s->fft_real[i]*s->fft_real[i]+s->fft_imag[i]*s->fft_imag[i];
  for(int m=0;m<26;m++) {
    float energy=0;for(int i=0;i<257;i++)energy+=s->power[i]*cnn_mel[m*257+i];
    a[m]=(logf(fmaxf(energy,1e-10f))-cnn_mean[m])/cnn_std[m];
  }
  for(int layer=0;layer<6;layer++) {
    const cnn_layer_t *p=&cnn_layers[layer];int position=s->positions[layer];
    float *history=s->history+p->offset;
    for(int ic=0;ic<p->in_channels;ic++)history[ic*p->history+position]=a[ic];
    for(int oc=0;oc<p->out_channels;oc++) {
      float value=(float)p->b[oc]*p->bs;
      for(int ic=0;ic<p->in_channels;ic++)for(int k=0;k<p->kernel;k++) {
        int t=position-p->dilation*(p->kernel-1-k);if(t<0)t+=p->history;
        value+=history[ic*p->history+t]*(float)p->w[(oc*p->in_channels+ic)*p->kernel+k]*p->ws;
      }
      b[oc]=layer<5?fmaxf(value,0):value;
    }
    s->positions[layer]=(position+1)%p->history;
    memcpy(a,b,(size_t)p->out_channels*sizeof(float));
  }
  for(int k=0;k<5;k++) {
    s->logits[k]=a[k];s->sums[k]+=a[k];
    s->ema[k]=s->frames==0?a[k]:.98f*s->ema[k]+.02f*a[k];
  }
  s->frames++;
  /* Debug consumers may inspect every frame; live UIs should emit at frames%10=0. */
  if(callback)callback(s,context);
}
void cnn_feed(cnn_state_t *s,const int16_t *samples,size_t count,cnn_frame_callback callback,void *context) {
  for(size_t i=0;i<count;i++) {
    float raw=(float)samples[i]/32768.0f;
    float emphasized=s->samples==0?raw:raw-.97f*s->previous_raw;s->previous_raw=raw;
    s->audio[s->audio_pos]=emphasized;s->audio_pos=(s->audio_pos+1)%400;s->samples++;
    if(s->samples>=400&&(s->samples-400)%160==0)process_frame(s,callback,context);
  }
}
void cnn_finish(cnn_state_t *s,cnn_frame_callback callback,void *context) {
  if(s->frames==0&&s->samples>0){int16_t zero[400]={0};s->previous_raw=0;cnn_feed(s,zero,400-(size_t)s->samples,callback,context);}
}
void cnn_result(const cnn_state_t *s,float probabilities[5]) {
  float logits[5]={0};for(int k=0;k<5;k++)logits[k]=s->frames?(float)(s->sums[k]/s->frames):0;
  cnn_softmax(logits,probabilities);
}
