/* Pure-C MFCC and Delta pooling for the v11 event classifier.
 * Arithmetic retained from the shared v11 front-end; no keyword matching.
 */
#include "mfcc_features.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- Static tables (pre-computed) ---- */

/* Mel filter bank: NUM_FILTERS × (NFFT/2+1) */
static float s_mel_fb[MFCC_NUM_FILTERS][MFCC_NFFT / 2 + 1];
static int s_mel_fb_init = 0;

/* DCT-II matrix: NUM_MFCC × NUM_FILTERS */
static float s_dct_mat[MFCC_NUM_MFCC][MFCC_NUM_FILTERS];

/* Hamming window */
static float s_hamming[MFCC_FRAME_LEN];

/* MFCC workspace for event feature pooling. */
static float s_mfcc_workspace[MFCC_MAX_FRAMES][MFCC_NUM_MFCC];

/* ---- Helper: Hz <-> Mel conversion ---- */

static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/* ---- Initialize static tables (called once) ---- */

static void init_tables(void)
{
    if (s_mel_fb_init) return;

    /* Build Mel filter bank */
    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel((float)MFCC_SAMPLE_RATE / 2.0f);

    float mel_points[MFCC_NUM_FILTERS + 2];
    float hz_points[MFCC_NUM_FILTERS + 2];
    int bin_points[MFCC_NUM_FILTERS + 2];

    for (int i = 0; i < MFCC_NUM_FILTERS + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (MFCC_NUM_FILTERS + 1);
        hz_points[i] = mel_to_hz(mel_points[i]);
        bin_points[i] = (int)floorf((MFCC_NFFT + 1) * hz_points[i] / MFCC_SAMPLE_RATE);
    }

    memset(s_mel_fb, 0, sizeof(s_mel_fb));
    for (int m = 1; m <= MFCC_NUM_FILTERS; m++) {
        int left = bin_points[m - 1];
        int center = bin_points[m];
        int right = bin_points[m + 1];
        for (int k = left; k < center; k++) {
            if (center > left)
                s_mel_fb[m - 1][k] = (float)(k - left) / (center - left);
        }
        for (int k = center; k < right; k++) {
            if (right > center)
                s_mel_fb[m - 1][k] = (float)(right - k) / (right - center);
        }
    }

    /* Build DCT-II matrix */
    for (int k = 0; k < MFCC_NUM_MFCC; k++) {
        for (int n = 0; n < MFCC_NUM_FILTERS; n++) {
            s_dct_mat[k][n] = cosf(M_PI * k * (2 * n + 1) / (2.0f * MFCC_NUM_FILTERS));
            s_dct_mat[k][n] *= sqrtf(2.0f / MFCC_NUM_FILTERS);
        }
    }
    /* Scale k=0 by sqrt(0.5) */
    for (int n = 0; n < MFCC_NUM_FILTERS; n++) {
        s_dct_mat[0][n] *= sqrtf(0.5f);
    }

    /* Build Hamming window */
    for (int i = 0; i < MFCC_FRAME_LEN; i++) {
        s_hamming[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (MFCC_FRAME_LEN - 1));
    }

    s_mel_fb_init = 1;
}

/* ---- In-place radix-2 FFT (N must be power of 2) ---- */

static void fft_radix2(float *real, float *imag, int n)
{
    /* Bit reversal permutation */
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
    }

    /* Cooley-Tukey */
    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * M_PI / len;
        float wr = cosf(angle), wi = sinf(angle);
        for (int i = 0; i < n; i += len) {
            float wmr = 1.0f, wmi = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                float tr = wmr * real[i + k + len/2] - wmi * imag[i + k + len/2];
                float ti = wmr * imag[i + k + len/2] + wmi * real[i + k + len/2];
                real[i + k + len/2] = real[i + k] - tr;
                imag[i + k + len/2] = imag[i + k] - ti;
                real[i + k] += tr;
                imag[i + k] += ti;
                float nwr = wmr * wr - wmi * wi;
                wmi = wmr * wi + wmi * wr;
                wmr = nwr;
            }
        }
    }
}

/* ---- MFCC extraction ---- */

static int extract_mfcc(const int16_t *pcm, int num_samples,
                        float mfcc_out[MFCC_MAX_FRAMES][MFCC_NUM_MFCC])
{
    init_tables();

    /* Pre-emphasis: y[n] = x[n] - 0.97 * x[n-1] */
    static float x[MFCC_SAMPLE_RATE * 2];  /* max 2 seconds */
    int n = (num_samples < MFCC_SAMPLE_RATE * 2) ? num_samples : MFCC_SAMPLE_RATE * 2;

    x[0] = (float)pcm[0] / 32768.0f;
    for (int i = 1; i < n; i++) {
        x[i] = (float)pcm[i] / 32768.0f - 0.97f * (float)pcm[i - 1] / 32768.0f;
    }

    /* Frame the signal */
    int num_frames = 1 + (n - MFCC_FRAME_LEN) / MFCC_HOP_LEN;
    if (num_frames > MFCC_MAX_FRAMES) num_frames = MFCC_MAX_FRAMES;
    if (num_frames < 1) num_frames = 1;

    /* FFT workspace */
    static float fft_real[MFCC_NFFT];
    static float fft_imag[MFCC_NFFT];
    static float power[MFCC_NFFT / 2 + 1];
    static float mel_energy[MFCC_NUM_FILTERS];
    static float log_mel[MFCC_NUM_FILTERS];

    for (int f = 0; f < num_frames; f++) {
        int start = f * MFCC_HOP_LEN;

        /* Copy frame + Hamming window */
        memset(fft_real, 0, sizeof(fft_real));
        memset(fft_imag, 0, sizeof(fft_imag));
        for (int i = 0; i < MFCC_FRAME_LEN; i++) {
            fft_real[i] = x[start + i] * s_hamming[i];
        }

        /* FFT */
        fft_radix2(fft_real, fft_imag, MFCC_NFFT);

        /* Power spectrum */
        for (int i = 0; i <= MFCC_NFFT / 2; i++) {
            power[i] = fft_real[i] * fft_real[i] + fft_imag[i] * fft_imag[i];
        }

        /* Mel filter bank */
        for (int m = 0; m < MFCC_NUM_FILTERS; m++) {
            float sum = 0.0f;
            for (int i = 0; i <= MFCC_NFFT / 2; i++) {
                sum += power[i] * s_mel_fb[m][i];
            }
            mel_energy[m] = (sum < 1e-10f) ? 1e-10f : sum;
            log_mel[m] = logf(mel_energy[m]);
        }

        /* DCT → MFCC */
        for (int k = 0; k < MFCC_NUM_MFCC; k++) {
            float sum = 0.0f;
            for (int m = 0; m < MFCC_NUM_FILTERS; m++) {
                sum += log_mel[m] * s_dct_mat[k][m];
            }
            mfcc_out[f][k] = sum;
        }
    }

    return num_frames;
}

int mfcc_extract_stats(const int16_t *pcm, int num_samples,
                               float stats[MFCC_STATS_DIM])
{
    int frames;

    if (pcm == NULL || stats == NULL || num_samples <= 0) {
        return -1;
    }
    memset(stats, 0, MFCC_STATS_DIM * sizeof(float));
    frames = extract_mfcc(pcm, num_samples, s_mfcc_workspace);
    if (frames <= 0) {
        return -1;
    }

    for (int coefficient = 0; coefficient < MFCC_NUM_MFCC; coefficient++) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        float delta_sum = 0.0f;
        float delta_sum_sq = 0.0f;

        for (int frame = 0; frame < frames; frame++) {
            float value = s_mfcc_workspace[frame][coefficient];
            sum += value;
            sum_sq += value * value;
            if (frame > 0) {
                float delta = value - s_mfcc_workspace[frame - 1][coefficient];
                delta_sum += delta;
                delta_sum_sq += delta * delta;
            }
        }

        float mean = sum / (float)frames;
        stats[coefficient] = mean;
        stats[MFCC_NUM_MFCC + coefficient] =
            sqrtf(fmaxf(0.0f, sum_sq / (float)frames - mean * mean));

        if (frames > 1) {
            int delta_frames = frames - 1;
            float delta_mean = delta_sum / (float)delta_frames;
            stats[2 * MFCC_NUM_MFCC + coefficient] = delta_mean;
            stats[3 * MFCC_NUM_MFCC + coefficient] =
                sqrtf(fmaxf(0.0f, delta_sum_sq / (float)delta_frames -
                                      delta_mean * delta_mean));
        }
    }
    return 0;
}

