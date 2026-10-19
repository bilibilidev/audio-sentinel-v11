#ifndef AED_CNN_STREAM_V2_H
#define AED_CNN_STREAM_V2_H
#include <stddef.h>
#include <stdint.h>
#include "cnn_model_config_v2.h"
#ifndef CNNV2_SCRATCH_CHANNELS
#define CNNV2_SCRATCH_CHANNELS 32 /* Backward-compatible earlier exports. */
#endif

/* One state per independent audio stream; allocate on the heap on small stacks.
 * Weights are immutable. Call finish only at EOF, never at callback boundaries.
 */
typedef struct {
  uint64_t samples, frames;
  int audio_pos, positions[CNNV2_LAYER_COUNT];
  float previous_raw, previous_energy;
  float audio[400], raw[400], history[CNNV2_HISTORY_SIZE];
  float fft_real[512], fft_imag[512], power[257];
  float logits[5], ema[5], logsumexp_max[5];
  double sums[5], sum_exp[5], energy_sum;
} cnn_v2_state_t;
typedef void (*cnn_v2_callback)(const cnn_v2_state_t *, void *);
void cnn_v2_reset(cnn_v2_state_t *state);
void cnn_v2_feed(cnn_v2_state_t *state, const int16_t *pcm, size_t count,
                 cnn_v2_callback callback, void *context);
void cnn_v2_finish(cnn_v2_state_t *state, cnn_v2_callback callback, void *context);
void cnn_v2_softmax(const float logits[5], float probabilities[5]);
void cnn_v2_result(const cnn_v2_state_t *state, float probabilities[5]);
const char *cnn_v2_label(int index);
#endif
