#include "wav_reader.h"
#include <limits.h>
#include <string.h>

static uint16_t read_u16_le(const unsigned char *p)
{
  return (uint16_t)((uint16_t)p[0] | (uint16_t)p[1] << 8);
}
static uint32_t read_u32_le(const unsigned char *p)
{
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
         (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static int skip_chunk(FILE *fp, uint32_t size)
{
  uint64_t count = (uint64_t)size + (size & 1U);
  if (count > LONG_MAX) return -1;
  return fseek(fp, (long)count, SEEK_CUR);
}
FILE *open_wav_16k_mono(const char *path, uint32_t *data_bytes)
{
  unsigned char header[12], chunk[8], fmt[16];
  int valid = 0;
  FILE *fp;
  if (!path || !data_bytes) return NULL;
  fp = fopen(path, "rb");
  if (!fp) { perror(path); return NULL; }
  if (fread(header, 1, 12, fp) != 12 || memcmp(header, "RIFF", 4) ||
      memcmp(header + 8, "WAVE", 4)) goto invalid;
  while (fread(chunk, 1, 8, fp) == 8)
    {
      uint32_t size = read_u32_le(chunk + 4);
      if (!memcmp(chunk, "fmt ", 4))
        {
          if (size < 16 || fread(fmt, 1, 16, fp) != 16) goto invalid;
          valid = read_u16_le(fmt) == 1 && read_u16_le(fmt + 2) == 1 &&
                  read_u32_le(fmt + 4) == 16000 && read_u16_le(fmt + 14) == 16 &&
                  read_u16_le(fmt + 12) == 2;
          if (skip_chunk(fp, size - 16)) goto invalid;
        }
      else if (!memcmp(chunk, "data", 4))
        {
          if (!valid || !size || size % 2) goto invalid;
          *data_bytes = size;
          return fp;
        }
      else if (skip_chunk(fp, size)) goto invalid;
    }
invalid:
  fprintf(stderr, "Invalid WAV (requires PCM16 mono 16 kHz): %s\n", path);
  fclose(fp);
  return NULL;
}
int read_wav_16k_mono(const char *path, int16_t *pcm, int max_samples)
{
  uint32_t bytes;
  FILE *fp;
  size_t count;
  if (!pcm || max_samples <= 0) return -1;
  memset(pcm, 0, (size_t)max_samples * sizeof(*pcm));
  fp = open_wav_16k_mono(path, &bytes);
  if (!fp) return -1;
  count = bytes / 2;
  if (count > (size_t)max_samples) count = (size_t)max_samples;
  if (fread(pcm, sizeof(*pcm), count, fp) != count)
    { fclose(fp); return -1; }
  fclose(fp);
  return (int)count;
}
