/* Evaluation-only full-recording adapter. Legacy first-three-second API stays
 * unchanged. Extended mode applies its exact one-second feature, quiet gate,
 * and event-max/mean aggregation rules across the whole WAV. */
#include "audio_classifier.h"
#include "feature_extractor.h"
#include "model_infer.h"
#include "wav_reader.h"
#include <stdio.h>
#include <string.h>

int model_predict_int8(const float features[FEATURE_DIM], float probabilities[NUM_CLASSES]);
const char *model_label_name_int8(int index);

static int full_recording(const char *path, aed_result_t *r, int *quiet_windows, int mean_only)
{
  int16_t pcm[AUDIO_NUM_SAMPLES];
  float features[AUDIO_FEATURE_DIM], probs[NUM_CLASSES], average[NUM_CLASSES]={0};
  float best_probs[NUM_CLASSES]={0}, best_confidence=0;
  int best_class=0, windows=0;
  uint32_t bytes=0;
  FILE *f=open_wav_16k_mono(path,&bytes);
  if(!f)return -1;
  memset(r,0,sizeof(*r)); *quiet_windows=0;
  while(bytes) {
    size_t n=bytes/2;
    if(n>AUDIO_NUM_SAMPLES)n=AUDIO_NUM_SAMPLES;
    memset(pcm,0,sizeof(pcm));
    if(fread(pcm,sizeof(*pcm),n,f)!=n){fclose(f);return -1;}
    bytes-=(uint32_t)(n*2); r->samples+=n;
    extract_audio_features(pcm,AUDIO_NUM_SAMPLES,features);
    if(features[0]>r->energy)r->energy=features[0];
    memset(probs,0,sizeof(probs));int pred;
    if(!mean_only && features[0]<2.0e-5f){probs[0]=1;pred=0;(*quiet_windows)++;}
    else pred=model_predict_int8(features,probs);
    for(int i=0;i<NUM_CLASSES;i++)average[i]+=probs[i];
    if(pred!=0 && probs[pred]>best_confidence){best_class=pred;best_confidence=probs[pred];memcpy(best_probs,probs,sizeof(probs));}
    windows++;
  }
  fclose(f);if(!windows)return -1;
  r->analyzed_windows=windows;
  if(!mean_only && best_confidence>=.80f){r->class_index=best_class;r->confidence=best_confidence;memcpy(r->probabilities,best_probs,sizeof(best_probs));}
  else {
    for(int i=0;i<NUM_CLASSES;i++){
      r->probabilities[i]=average[i]/windows;
      if(r->probabilities[i]>r->probabilities[r->class_index])r->class_index=i;
    }
    r->confidence=r->probabilities[r->class_index];
  }
  return 0;
}
int main(int argc,char **argv)
{
  if(argc<3)return 2;
  int mean_only=!strcmp(argv[1],"mean");
  int full=!strcmp(argv[1],"full") || mean_only;
  if(!full && strcmp(argv[1],"legacy"))return 2;
  puts("file\tprediction\tconfidence\twindows\tquiet_windows\tp0\tp1\tp2\tp3\tp4");
  for(int i=2;i<argc;i++){
    aed_result_t r;int quiet=-1;
    int rc=full?full_recording(argv[i],&r,&quiet,mean_only):aed_classify_three_second_wav(argv[i],&r);
    if(rc)return 1;
    printf("%s\t%s\t%.9g\t%d\t%d",argv[i],model_label_name_int8(r.class_index),r.confidence,r.analyzed_windows,quiet);
    for(int j=0;j<NUM_CLASSES;j++)printf("\t%.9g",r.probabilities[j]);
    puts("");
  }
  return 0;
}
