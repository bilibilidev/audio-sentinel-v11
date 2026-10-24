#ifndef AUDIO_CLASSIFIER_H
#define AUDIO_CLASSIFIER_H

#include <stddef.h>
#include <stdint.h>
#ifndef NUM_CLASSES
#define NUM_CLASSES 5
#endif

typedef struct
{
  int class_index;
  float confidence;
  float energy;
  float probabilities[NUM_CLASSES];
  int analyzed_windows;
  uint64_t analyzed_frames;
  uint64_t samples;
} aed_result_t;

/* Selected at build time: Tiny MLP by default; AED_USE_CNN or the NuttX
 * AE_AUDIODETECT_CNN option retains the CNN v2 backend. */
const char *aed_model_name(void);
const char *aed_model_banner(void);
const char *aed_model_sha256(void);
const char *aed_label_name(int index);

/* Full-file classification; mono signed16 PCM at16kHz. MLP averages all
 * one-second window probabilities; CNN averages causal frame logits. */
int aed_classify_wav(const char *path, aed_result_t *result);
int aed_classify_pcm(const int16_t *pcm, size_t count, aed_result_t *result);
/* Compatibility entry point for the UI's three-second capture files. */
int aed_classify_three_second_wav(const char *path, aed_result_t *result);

#endif
