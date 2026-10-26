#include "weights.h"
#include "feature_extractor.h"
#include "wav_reader.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
static void classify(float x[92],float p[5],float *attention) {
 float n[FEATURE_DIM],h[HIDDEN_DIM],l[5];
 for(int i=0;i<FEATURE_DIM;i++){
  int index=FEATURE_INDICES[i];float v=x[index];
#if STUDY_ASINH
  if(index<40)v=asinhf(v/TIME_ASINH_SCALE[index]);
#endif
  n[i]=(v-FEATURE_MEAN[i])/(FEATURE_STD[i]==0?1:FEATURE_STD[i]);
 }
 for(int j=0;j<HIDDEN_DIM;j++){float v=(float)B1_INT8[j]*B1_SCALE;for(int i=0;i<FEATURE_DIM;i++)v+=n[i]*(float)W1_INT8[i][j]*W1_SCALE;h[j]=v>0?v:0;}
 *attention=0;
#ifdef OPT_ATTENTION
 for(int i=0;i<HIDDEN_DIM;i++)*attention+=h[i]*(float)ATT_W_INT8[i]*ATT_W_SCALE;
#endif
 for(int j=0;j<5;j++){float v=(float)B2_INT8[j]*B2_SCALE;for(int i=0;i<HIDDEN_DIM;i++)v+=h[i]*(float)W2_INT8[i][j]*W2_SCALE;l[j]=v;}
 float m=l[0],sum=0;for(int i=1;i<5;i++)if(l[i]>m)m=l[i];for(int i=0;i<5;i++){p[i]=expf(l[i]-m);sum+=p[i];}for(int i=0;i<5;i++)p[i]/=sum;
}
int main(int argc,char **argv){
 puts("file\tprediction\tconfidence\tp0\tp1\tp2\tp3\tp4");
 for(int a=1;a<argc;a++){uint32_t bytes;FILE *f=open_wav_16k_mono(argv[a],&bytes);if(!f)return 1;float total[5]={0};unsigned count=0;double weighted[5]={0},maximum=-INFINITY,weight_sum=0;
 while(bytes){int16_t pcm[16000];float x[92],p[5],attention;size_t n=bytes/2;if(n>16000)n=16000;memset(pcm,0,sizeof(pcm));if(fread(pcm,2,n,f)!=n)return 1;bytes-=(uint32_t)(n*2);extract_audio_features(pcm,16000,x);classify(x,p,&attention);
#ifdef OPT_ATTENTION
 if(attention>maximum){double rescale=isfinite(maximum)?exp(maximum-attention):0;weight_sum*=rescale;for(int i=0;i<5;i++)weighted[i]*=rescale;maximum=attention;}
 double w=exp(attention-maximum);weight_sum+=w;for(int i=0;i<5;i++)weighted[i]+=w*p[i];
#else
 for(int i=0;i<5;i++)total[i]+=p[i];
#endif
 count++;}fclose(f);
 if(!count)return 1;int pred=0;for(int i=0;i<5;i++){
#ifdef OPT_ATTENTION
 total[i]=(float)(weighted[i]/weight_sum);
#else
 total[i]/=count;
#endif
 if(total[i]>total[pred])pred=i;}
 printf("%s\t%s\t%.9g",argv[a],LABEL_NAMES[pred],total[pred]);for(int i=0;i<5;i++)printf("\t%.9g",total[i]);puts("");}return 0;}
