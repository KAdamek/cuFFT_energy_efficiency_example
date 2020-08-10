#include "nvml_run.h"
#include <stdio.h>
#include "results.h"
#include "debug.h"

nvmlReturn_t nvmlResult;
nvmlDevice_t nvmlDeviceID;
char deviceNameStr[128];

void nvml_setup(int device){

	// run the nvml Init phase
	nvmlResult = nvmlInit();
	if (NVML_SUCCESS != nvmlResult){
                printf("NVML init fail: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
        }

	// get the Device ID string for NVML
	nvmlResult =  nvmlDeviceGetHandleByIndex(device, &nvmlDeviceID);
	if (NVML_SUCCESS != nvmlResult){
                printf("NVML get Device ID fail: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
        }

	nvmlResult = nvmlDeviceGetName(nvmlDeviceID, deviceNameStr, sizeof(deviceNameStr)/sizeof(deviceNameStr[0]));
	if (NVML_SUCCESS != nvmlResult){
                printf("NVML get Device name fail: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
        }


	//set the desired min and max GPU clock
	unsigned int gpu_clock;
	gpu_clock = assign_clock(deviceNameStr);
	nvmlResult = nvmlDeviceSetGpuLockedClocks(nvmlDeviceID, gpu_clock, gpu_clock);
	if (NVML_SUCCESS != nvmlResult){
                printf("NVML set GPU clock fail: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
        }

	if(DEBUG){
		printf("----------- DEBUG -----------\n");
		printf("GPU name is %s;\n", deviceNameStr);
		printf("GPU clock set to %d;\n", gpu_clock);
		printf("-----------------------------\n");
	}

}

void nvml_reset(){

	nvmlResult = nvmlDeviceResetGpuLockedClocks(nvmlDeviceID);
	if (NVML_SUCCESS != nvmlResult){
                printf("NVML reset GPU fail: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
        }

}

// function to set the near optimal frequency by device Name
unsigned int assign_clock(char *deviceName){
	unsigned int set_clock;
	if ( strcmp(deviceName, "Tesla V100-SXM2-32GB")  == 0 ){
		set_clock = 255;
	} 
	else {
		set_clock = 262;
	}

	return set_clock;
}

