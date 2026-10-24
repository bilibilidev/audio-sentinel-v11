/* Shared CLI/UI classifier: train-aligned MLP by default, optional CNN v2. */
#include "audio_classifier.h"
#include "wav_reader.h"
#include <stdlib.h>
#include <string.h>

const char *aed_label_name(int index)
{
  static const char *names[]={"background","cough","glass_break","baby_cry","dog_bark"};
  return index>=0 && index<NUM_CLASSES ? names[index] : "unknown";
}

#if defined(AED_USE_CNN) || defined(CONFIG_AE_AUDIODETECT_CNN)
#include "cnn_stream_v2.h"
const char *aed_model_name(void) { return "CNN v2"; }
const char *aed_model_banner(void) { return "Audio Sentinel CNN v2"; }
const char *aed_model_sha256(void) { return CNNV2_MODEL_SHA256; }

static void finish_result(cnn_v2_state_t *state, aed_result_t *result)
{
  uint64_t count = state->samples;
  memset(result, 0, sizeof(*result));
  result->energy = count ? (float)(state->energy_sum / count) : 0;
  cnn_v2_finish(state, NULL, NULL);
  cnn_v2_result(state, result->probabilities);
  for (int i = 1; i < NUM_CLASSES; i++)
    if (result->probabilities[i] > result->probabilities[result->class_index])
      result->class_index = i;
  result->confidence = result->probabilities[result->class_index];
  result->samples = count;
  result->analyzed_frames = state->frames;
  result->analyzed_windows = 0; /* Compatibility: CNN frames are not MLP windows. */
}
int aed_classify_pcm(const int16_t *pcm, size_t count, aed_result_t *result)
{
  cnn_v2_state_t *state;
  if (!pcm || !count || !result) return -1;
  state = calloc(1, sizeof(*state));
  if (!state) return -1;
  cnn_v2_feed(state, pcm, count, NULL, NULL);
  finish_result(state, result);
  free(state);
  return 0;
}
int aed_classify_wav(const char *path, aed_result_t *result)
{
  uint32_t remaining;
  int16_t buffer[160];
  cnn_v2_state_t *state;
  FILE *fp;
  if (!result) return -1;
  fp = open_wav_16k_mono(path, &remaining);
  if (!fp) return -1;
  state = calloc(1, sizeof(*state));
  if (!state) { fclose(fp); return -1; }
  while (remaining)
    {
      size_t count = remaining / 2;
      if (count > 160) count = 160;
      if (fread(buffer, sizeof(*buffer), count, fp) != count)
        { free(state); fclose(fp); return -1; }
      cnn_v2_feed(state, buffer, count, NULL, NULL);
      remaining -= (uint32_t)(count * 2);
    }
  fclose(fp);
  finish_result(state, result);
  free(state);
  return 0;
}
int aed_classify_three_second_wav(const char *path, aed_result_t *result)
{
  return aed_classify_wav(path, result);
}
#else
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
#endif
