#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>
#include <iomanip>

#include "debug.h"
#include "params.h"
#include "results.h"


//-----------------------------------------------
//---------- Signal generation
void Generate_signal_noise(float2 *h_input, int nElements, int nSeries){
	for(int f=0; f<nSeries; f++){
		for(int x=0; x<nElements; x++){
			h_input[nElements*f + x].y=rand() / (float)RAND_MAX;
			h_input[nElements*f + x].x=rand() / (float)RAND_MAX;
		}
	}
}
//------------------------------------------------<



int Calculate_GPU_HRMS(float2 *h_input, float *h_output, Performance_results *HRMS_results, int device);

int main(int argc, char* argv[]) {
	
	if(argc!=6) {
		printf("Argument error!\n");
		printf("1) Length of the time series\n");
		printf("2) Number of harmonics summed\n");
		printf("3) Number of time-series to process\n");
		printf("4) Number of runs of the kernel\n");
		printf("5) Device id\n");
        return (1);
    }
	char * pEnd;
	
	int nElements = strtol(argv[1],&pEnd,10);
	int nHarmonics = strtol(argv[2],&pEnd,10);
	int nSeries = strtol(argv[3],&pEnd,10);
	int nRuns = strtol(argv[4],&pEnd,10);
	int device = strtol(argv[5],&pEnd,10);
	
	if(DEBUG){
		printf("----------- DEBUG -----------\n");
		printf("Selected parameters:\n");
		printf("Number of elements per time-series = %d;\n", nElements);
		printf("Number of harmonics summed = %d;\n", nHarmonics);
		printf("Number of series to process = %d;\n", nSeries);
		printf("Number of runs: %d;\n", nRuns);
		printf("Device: %d;\n", device);
		printf("-----------------------------<\n");
	}
	
	// Performance measurements
	Performance_results HRMS_results;
	HRMS_results.Assign(nElements, nHarmonics, nSeries, nRuns, "HRMS_results.dat");
	
	float2 *h_input;
	float *h_output;
	h_input = new float2[nElements*nSeries];
	h_output = new float[nElements*nSeries];
	
	Generate_signal_noise(h_input, nElements, nSeries);
	
	Calculate_GPU_HRMS(h_input, h_output, &HRMS_results, device);
	
	delete [] h_input;
	delete [] h_output;
	
	if(VERBOSE) printf("     cuFFT Execution time:\033[32m%0.3f\033[0mms\n", HRMS_results.GPU_FFT_time);
	if(VERBOSE) printf("     MSD Execution time  :\033[32m%0.3f\033[0mms\n", HRMS_results.GPU_MSD_time);
	if(VERBOSE) printf("     HRMS Execution time :\033[32m%0.3f\033[0mms\n", HRMS_results.GPU_HRMS_time);
	if(VERBOSE) printf("     Total Execution time:\033[32m%0.3f\033[0mms\n", HRMS_results.GPU_total_time);
	if(VERBOSE) printf("     cuFFT represent     :\033[32m%0.3f\033[0m%% of total execution time\n", (HRMS_results.GPU_FFT_time/(HRMS_results.GPU_total_time/100.0)));
	HRMS_results.Save();

	return (0);
}
