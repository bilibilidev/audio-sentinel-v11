#ifndef AUDIO_MFCC_FEATURES_H
#define AUDIO_MFCC_FEATURES_H

#include <stdint.h>

#define MFCC_SAMPLE_RATE 16000
#define MFCC_FRAME_LEN   400
#define MFCC_HOP_LEN     160
#define MFCC_NFFT        512
#define MFCC_NUM_MFCC    13
#define MFCC_NUM_FILTERS 26
#define MFCC_MAX_FRAMES  200
#define MFCC_STATS_DIM (MFCC_NUM_MFCC * 4)

/* Layout: mean[13], std[13], delta_mean[13], delta_std[13]. */
int mfcc_extract_stats(const int16_t *pcm, int num_samples,
                       float stats[MFCC_STATS_DIM]);

#endif
