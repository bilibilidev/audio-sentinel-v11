#include "cnn_stream.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const char *names[]={"background","cough","glass_break","baby_cry","dog_bark"};
static uint32_t le32(const unsigned char *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static unsigned le16(const unsigned char *p){return p[0]|(unsigned)p[1]<<8;}
static int wav_data(FILE *f,uint32_t *remaining) {
  unsigned char head[12],part[8],fmt[16];int valid=0;
  if(fread(head,1,12,f)!=12||memcmp(head,"RIFF",4)||memcmp(head+8,"WAVE",4))return -1;
  while(fread(part,1,8,f)==8) {
    uint32_t n=le32(part+4);
    if(!memcmp(part,"fmt ",4)) {
      if(n<16||fread(fmt,1,16,f)!=16)return -1;
      valid=le16(fmt)==1&&le16(fmt+2)==1&&le32(fmt+4)==16000&&le16(fmt+14)==16;
      if(fseek(f,(long)(n-16+(n&1)),SEEK_CUR))return -1;
    }else if(!memcmp(part,"data",4)){if(!valid||n%2)return -1;*remaining=n;return 0;}
    else if(fseek(f,(long)(n+(n&1)),SEEK_CUR))return -1;
  }
  return -1;
}
typedef struct { FILE *debug; int live,stream; } output_context_t;
static void trace(const cnn_state_t *s,void *context) {
  output_context_t *out=(output_context_t *)context;FILE *f=out->debug;
  if(f) {
    fprintf(f,"%" PRIu64 "\t%" PRIu64,s->frames,s->samples);
    for(int i=0;i<5;i++)fprintf(f,"\t%.9g",s->logits[i]);
    for(int i=0;i<5;i++)fprintf(f,"\t%.9g",s->ema[i]);
    fprintf(f,"\t%d\n",s->frames%10==0);
  }
  if(out->live&&s->frames%10==0) {
    float probs[5];cnn_softmax(s->ema,probs);int pred=0;
    for(int i=1;i<5;i++)if(probs[i]>probs[pred])pred=i;
    printf("{\"type\":\"update\",\"stream\":%d,\"time_s\":%.6f,\"prediction\":\"%s\",\"confidence\":%.9g,\"warmup\":%s,\"probabilities\":[",out->stream,s->samples/16000.0,names[pred],probs[pred],s->frames<125?"true":"false");
    for(int i=0;i<5;i++)printf(i?",%.9g":"%.9g",probs[i]);
    puts("]}");fflush(stdout);
  }
}
int main(int argc,char **argv) {
  size_t chunk=160;const char *trace_path=NULL;int first=1,live=0,raw_stdin=0;
  while(first<argc&&!strncmp(argv[first],"--",2)) {
    if(!strcmp(argv[first],"--chunk")&&first+1<argc){chunk=(size_t)strtoul(argv[first+1],NULL,10);first+=2;}
    else if(!strcmp(argv[first],"--trace")&&first+1<argc){trace_path=argv[first+1];first+=2;}
    else if(!strcmp(argv[first],"--live")){live=1;first++;}
    else if(!strcmp(argv[first],"--stdin-raw")){raw_stdin=1;live=1;first++;}
    else return 2;
  }
  if((first==argc&&!raw_stdin)||chunk<1||chunk>65536||(trace_path&&!raw_stdin&&argc-first!=1)||(raw_stdin&&first!=argc))return 2;
  FILE *debug=trace_path?fopen(trace_path,"w"):NULL;if(trace_path&&!debug)return 1;
  if(debug)fputs("frame\tend_sample\tl0\tl1\tl2\tl3\tl4\te0\te1\te2\te3\te4\temit\n",debug);
  cnn_state_t *s=calloc(1,sizeof(*s));int16_t *buffer=malloc(chunk*sizeof(*buffer));
  if(!s||!buffer)return 1;
  if(!live)puts("file\tprediction\tconfidence\tframes\tsamples\tp0\tp1\tp2\tp3\tp4\tema_prediction\tema_confidence");
  int count_files=raw_stdin?1:argc-first;
  for(int a=0;a<count_files;a++) {
    FILE *f=raw_stdin?stdin:fopen(argv[first+a],"rb");uint32_t remaining=0;
    if(!f||(!raw_stdin&&(wav_data(f,&remaining)||remaining==0))){fprintf(stderr,"Invalid PCM16 mono16k WAV\n");return 1;}
    output_context_t context={debug,live,a};
    cnn_reset(s);
    while(raw_stdin||remaining) {
      size_t count=raw_stdin?chunk:remaining/2;if(count>chunk)count=chunk;
      size_t got=fread(buffer,sizeof(*buffer),count,f);
      if(raw_stdin&&got==0){if(ferror(f))return 1;break;}
      if(!raw_stdin&&got!=count)return 1;
      cnn_feed(s,buffer,got,(debug||live)?trace:NULL,&context);
      if(!raw_stdin)remaining-=(uint32_t)(got*2);
    }
    if(!raw_stdin)fclose(f);
    if(s->samples==0)return 1;
    uint64_t actual_samples=s->samples;
    cnn_finish(s,(debug||live)?trace:NULL,&context);float probs[5],ema[5];cnn_result(s,probs);cnn_softmax(s->ema,ema);
    int pred=0,ep=0;for(int i=1;i<5;i++){if(probs[i]>probs[pred])pred=i;if(ema[i]>ema[ep])ep=i;}
    if(live) {
      printf("{\"type\":\"summary\",\"stream\":%d,\"samples\":%" PRIu64 ",\"frames\":%" PRIu64 ",\"file_prediction\":\"%s\",\"file_confidence\":%.9g,\"last_stream_prediction\":\"%s\",\"probabilities\":[",a,actual_samples,s->frames,names[pred],probs[pred],names[ep]);
      for(int i=0;i<5;i++)printf(i?",%.9g":"%.9g",probs[i]);
      puts("]}");fflush(stdout);
    }else {
      printf("%s\t%s\t%.9g\t%" PRIu64 "\t%" PRIu64,argv[first+a],names[pred],probs[pred],s->frames,s->samples);
      for(int i=0;i<5;i++){printf("\t%.9g",probs[i]);}
      printf("\t%s\t%.9g\n",names[ep],ema[ep]);
    }
  }
  if(debug){fclose(debug);}free(s);free(buffer);return 0;
}
