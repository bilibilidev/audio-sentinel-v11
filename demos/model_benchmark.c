#define _POSIX_C_SOURCE 200809L
#include "audio_classifier.h"
#include "mlp_weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

/* Linux-only demo harness: executes the unchanged shared classifier. */
int main(int argc, char **argv)
{
  aed_result_t result;
  struct rusage usage;
  double times[200];
  char *end;
  if (argc != 3) return 2;
  long repeats = strtol(argv[2], &end, 10);
  if (*end || repeats < 1 || repeats > 200) return 2;
  for (int i = 0; i < 3; ++i)
    if (aed_classify_wav(argv[1], &result)) return 1;
  for (long i = 0; i < repeats; ++i)
    {
      struct timespec start, finish;
      if (clock_gettime(CLOCK_MONOTONIC, &start)) return 1;
      if (aed_classify_wav(argv[1], &result)) return 1;
      if (clock_gettime(CLOCK_MONOTONIC, &finish)) return 1;
      times[i] = (finish.tv_sec - start.tv_sec) * 1000.0 +
                 (finish.tv_nsec - start.tv_nsec) / 1000000.0;
    }
  if (getrusage(RUSAGE_SELF, &usage)) return 1;
  long rss = -1, peak_rss = -1;
  FILE *status = fopen("/proc/self/status", "r");
  if (!status) return 1;
  char line[256];
  while (fgets(line, sizeof(line), status))
    {
      if (strncmp(line, "VmRSS:", 6) == 0) sscanf(line + 6, "%ld", &rss);
      if (strncmp(line, "VmHWM:", 6) == 0) sscanf(line + 6, "%ld", &peak_rss);
    }
  fclose(status);
  size_t weights = sizeof(W1_INT8) + sizeof(B1_INT8) + sizeof(W2_INT8) + sizeof(B2_INT8);
  size_t parameters = sizeof(W1_INT8) / sizeof(int8_t) + sizeof(B1_INT8) / sizeof(int8_t) +
                      sizeof(W2_INT8) / sizeof(int8_t) + sizeof(B2_INT8) / sizeof(int8_t);
  size_t window_macs = (sizeof(W1_INT8) + sizeof(W2_INT8)) / sizeof(int8_t);
  size_t scales = sizeof(FEATURE_MEAN) + sizeof(FEATURE_STD) + sizeof(TIME_ASINH_SCALE) +
                  sizeof(W1_SCALE) + sizeof(B1_SCALE) + sizeof(W2_SCALE) + sizeof(B2_SCALE);
  printf("{\"input_dim\":%d,\"hidden_dim\":%d,\"classes\":%d,", FEATURE_DIM,HIDDEN_DIM,NUM_CLASSES);
  printf("\"parameters\":%zu,\"weight_bytes\":%zu,\"scale_bytes\":%zu,",parameters,weights,scales);
  printf("\"float32_weight_bytes\":%zu,\"prediction\":\"%s\",",parameters*sizeof(float),aed_label_name(result.class_index));
  printf("\"confidence\":%.9f,\"samples\":%llu,\"windows\":%d,",result.confidence,(unsigned long long)result.samples,result.analyzed_windows);
  printf("\"network_macs\":%llu,",(unsigned long long)window_macs*(unsigned long long)result.analyzed_windows);
  printf("\"rss_kib\":%ld,\"peak_rss_kib\":%ld,",rss,peak_rss);
  printf("\"getrusage_peak_rss_kib\":%ld,\"warmup\":3,\"times_ms\":[",usage.ru_maxrss);
  for (long i = 0; i < repeats; ++i) printf("%s%.9f", i ? "," : "", times[i]);
  puts("]}");
  return 0;
}
