#ifndef CNN_STREAM_H
#define CNN_STREAM_H
#include <stdint.h>
#include <stddef.h>
#include "cnn_weights.h"
typedef struct {
  uint64_t samples,frames;
  int audio_pos,positions[6];
  float previous_raw,audio[400],history[CNN_HISTORY_SIZE];
  float fft_real[512],fft_imag[512],power[257];
  float logits[5],ema[5];
  double sums[5];
} cnn_state_t;
typedef void (*cnn_frame_callback)(const cnn_state_t *state,void *context);
void cnn_reset(cnn_state_t *state);
void cnn_feed(cnn_state_t *state,const int16_t *samples,size_t count,cnn_frame_callback callback,void *context);
void cnn_finish(cnn_state_t *state,cnn_frame_callback callback,void *context);
void cnn_softmax(const float logits[5],float probabilities[5]);
void cnn_result(const cnn_state_t *state,float probabilities[5]);
#endif
