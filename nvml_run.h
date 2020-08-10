#ifndef GPU_NVML_RUN
#define GPU_NVML_RUN

#include <nvml.h>

void nvml_setup();
void nvml_reset();

unsigned int assign_clock(char *deviceName);

#endif
