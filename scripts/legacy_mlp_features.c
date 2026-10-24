/* Evaluation-only dump of the unchanged C front-end, one second per row. */
#include "feature_extractor.h"
#include "wav_reader.h"
#include <stdio.h>
#include <string.h>
int main(int argc,char **argv)
{
  for(int a=1;a<argc;a++) {
    uint32_t bytes;FILE *f=open_wav_16k_mono(argv[a],&bytes);if(!f)return 1;
    int index=0;
    while(bytes) {
      int16_t pcm[16000];float x[92];size_t n=bytes/2;if(n>16000)n=16000;
      memset(pcm,0,sizeof(pcm));if(fread(pcm,2,n,f)!=n)return 1;bytes-=(uint32_t)(n*2);
      extract_audio_features(pcm,16000,x);printf("%d %d",a-1,index++);
      for(int i=0;i<92;i++)printf(" %.9g",x[i]);
      puts("");
    }
    fclose(f);
  }
  return 0;
}
