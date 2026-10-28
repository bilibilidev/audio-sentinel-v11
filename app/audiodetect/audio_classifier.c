/* Shared CLI/UI classifier for the formal Tiny MLP model. */
#include "audio_classifier.h"
#include "wav_reader.h"
#include <stdlib.h>
#include <string.h>

const char *aed_label_name(int index)
{
  static const char *names[]={"background","cough","glass_break","baby_cry","dog_bark"};
  return index>=0 && index<NUM_CLASSES ? names[index] : "unknown";
}

#include "feature_extractor.h"
#include "mlp_model_infer.h"
#include "mlp_model_config.h"
const char *aed_model_name(void) { return "Tiny MLP"; }
const char *aed_model_banner(void) { return "Audio Sentinel Tiny MLP"; }
const char *aed_model_sha256(void) { return MLP_MODEL_SHA256; }

typedef struct {
  int16_t pcm[AUDIO_NUM_SAMPLES];
  size_t used;
  uint64_t samples;
  unsigned windows;
  float sum[NUM_CLASSES];
  double energy_sum;
} mlp_state_t;

static void mlp_window(mlp_state_t *s)
{
  float features[AUDIO_FEATURE_DIM],probabilities[NUM_CLASSES];
  extract_audio_features(s->pcm,AUDIO_NUM_SAMPLES,features);
  mlp_model_predict(features,probabilities);
  for(int i=0;i<NUM_CLASSES;i++)s->sum[i]+=probabilities[i];
  s->windows++;s->used=0;
}
static void mlp_feed(mlp_state_t *s,const int16_t *pcm,size_t count)
{
  while(count) {
    size_t n=AUDIO_NUM_SAMPLES-s->used;if(n>count)n=count;
    memcpy(s->pcm+s->used,pcm,n*sizeof(*pcm));
    for(size_t i=0;i<n;i++){double v=pcm[i]/32768.0;s->energy_sum+=v*v;}
    s->used+=n;s->samples+=n;pcm+=n;count-=n;
    if(s->used==AUDIO_NUM_SAMPLES)mlp_window(s);
  }
}
static void mlp_finish(mlp_state_t *s,aed_result_t *r)
{
  if(s->used){memset(s->pcm+s->used,0,(AUDIO_NUM_SAMPLES-s->used)*sizeof(*s->pcm));mlp_window(s);}
  memset(r,0,sizeof(*r));
  for(int i=0;i<NUM_CLASSES;i++){
    r->probabilities[i]=s->sum[i]/s->windows;
    if(r->probabilities[i]>r->probabilities[r->class_index])r->class_index=i;
  }
  r->confidence=r->probabilities[r->class_index];r->energy=(float)(s->energy_sum/s->samples);
  r->samples=s->samples;r->analyzed_windows=(int)s->windows;
  r->analyzed_frames=(uint64_t)s->windows*98; /* Includes zero-padded tail. */
}
int aed_classify_pcm(const int16_t *pcm,size_t count,aed_result_t *r)
{
  if(!pcm || !count || !r)return -1;
  mlp_state_t *s=calloc(1,sizeof(*s));if(!s)return -1;
  mlp_feed(s,pcm,count);mlp_finish(s,r);free(s);return 0;
}
int aed_classify_wav(const char *path,aed_result_t *r)
{
  uint32_t bytes;int16_t buffer[160];if(!r)return -1;
  FILE *f=open_wav_16k_mono(path,&bytes);if(!f)return -1;
  mlp_state_t *s=calloc(1,sizeof(*s));if(!s){fclose(f);return -1;}
  while(bytes){size_t n=bytes/2;if(n>160)n=160;
    if(fread(buffer,sizeof(*buffer),n,f)!=n){free(s);fclose(f);return -1;}
    mlp_feed(s,buffer,n);bytes-=(uint32_t)(n*2);
  }
  fclose(f);mlp_finish(s,r);free(s);return 0;
}
int aed_classify_three_second_wav(const char *path,aed_result_t *r)
{
  return aed_classify_wav(path,r);
}
