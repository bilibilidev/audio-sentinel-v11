#ifndef WAV_READER_H
#define WAV_READER_H

#include <stdint.h>
#include <stdio.h>

/* On success, return a file positioned at PCM data. Caller closes it. */
FILE *open_wav_16k_mono(const char *path, uint32_t *data_bytes);
int read_wav_16k_mono(const char *path, int16_t *pcm, int max_samples);

#endif
