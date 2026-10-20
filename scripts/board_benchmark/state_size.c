/* Compile only; nm -S reports the ARM ABI sizeof, with no board execution. */
#include "cnn_stream_v2.h"
const unsigned char cnn_v2_state_bytes[sizeof(cnn_v2_state_t)] = {0};
