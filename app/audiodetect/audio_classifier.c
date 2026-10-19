/* Shared CNN v2 path for NSH/host CLI and LVGL capture. No MLP energy gate,
 * one-second voting or truncation: aggregation matches the exported model. */
#include "audio_classifier.h"
#include "cnn_stream_v2.h"
#include "wav_reader.h"
#include <stdlib.h>
#include <string.h>

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
