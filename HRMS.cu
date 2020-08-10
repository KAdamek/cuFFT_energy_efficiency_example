#include <iostream>
#include <fstream>
#include <vector>


#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "debug.h"
#include "timer.h"
#include "utils_cuda.h"

#include <cufftXt.h>
#include <cuda_fp16.h>

#include "params.h"
#include "results.h"
#include "MSD_GPU_library.h"

#include "nvml_run.h"

#define PHS_NTHREADS 64
#define CT_CORNER_BLOCKS 1
#define CT_ROWS_PER_WARP 2
#define CT_NTHREADS 512
#define WARP 32

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

__global__ void GPU_simple_power_and_interbin_kernel(float2 *d_input_complex, float *d_output_power, int nTimesamples, float norm){
    int pos_x = blockIdx.x*blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y*nTimesamples;
	
    float2 A;
    A.x = 0; A.y = 0;
	
    if( pos_x < nTimesamples ) {
		A = d_input_complex[pos_y + pos_x];
		d_output_power[pos_y + pos_x] = (A.x*A.x + A.y*A.y)*norm;
    }
}


__global__ void corner_turn_SM_kernel(float const* __restrict__ d_input, float *d_output, int primary_size, int secondary_size) {
    __shared__ float s_input[WARP*(WARP+1)*CT_CORNER_BLOCKS];
	
	int i, spos, itemp, pc, sc;
	size_t gpos;
	
	int warp_id = threadIdx.x>>5;
	int local_id = threadIdx.x & (WARP - 1);
	
	gpos=(size_t)((size_t)(blockIdx.y*(blockDim.x>>5)) + (size_t)warp_id)*CT_ROWS_PER_WARP*primary_size + (size_t)(blockIdx.x*CT_CORNER_BLOCKS*WARP) + (size_t)local_id;
	for(int by=0; by<CT_ROWS_PER_WARP; by++){
		spos=local_id*WARP + local_id + warp_id*CT_ROWS_PER_WARP + by;
		for(int bx=0; bx<CT_CORNER_BLOCKS; bx++){ // temporary 
			if(gpos<primary_size){
				s_input[spos]=d_input[gpos];
			}
			gpos=gpos + (size_t)WARP;
			spos=spos + WARP*(WARP+1);
		}
		gpos=gpos + (size_t)primary_size - (size_t)(CT_CORNER_BLOCKS*WARP);
	}
	
	__syncthreads();
	
	itemp=warp_id*CT_ROWS_PER_WARP*CT_CORNER_BLOCKS;
	for(i=0; i<CT_ROWS_PER_WARP*CT_CORNER_BLOCKS; i++){
		pc = (blockIdx.x*CT_CORNER_BLOCKS*WARP + itemp + i);
		sc = WARP*blockIdx.y + local_id;
		if( pc<primary_size && sc<secondary_size ) {
			gpos=(size_t)(pc*secondary_size) + (size_t)sc;
			spos=(itemp + i)*(WARP+1) + local_id;
			d_output[gpos]=s_input[spos];
		}
	}
}

__global__ void PHS_GPU_kernel(float const* __restrict__ d_input, float *d_output_SNR, ushort *d_output_harmonics, float *d_MSD, int nTimesamples, int nSpectra, int nHarmonics){
	float HS_value, temp_SNR, SNR;
	ushort max_SNR_harmonic;
	int pos;

	// reading 0th harmonic, i.e. fundamental frequency
	pos = blockIdx.x*nSpectra + blockIdx.y*blockDim.x + threadIdx.x;
	if( (blockIdx.y*blockDim.x + threadIdx.x)<nSpectra ){
		HS_value = __ldg(&d_input[pos]);
		SNR = (HS_value - __ldg(&d_MSD[0]))/(__ldg(&d_MSD[1]));
		max_SNR_harmonic = 0;
		
		if(blockIdx.x>0) {
			for(int f=1; f<nHarmonics; f++) {
				if( (blockIdx.x + f*blockIdx.x)<nTimesamples ) {
					pos = (blockIdx.x + f*blockIdx.x)*nSpectra + blockIdx.y*blockDim.x + threadIdx.x;
					HS_value = HS_value + __ldg(&d_input[pos]);
					temp_SNR = (HS_value - __ldg(&d_MSD[f*2]))/(__ldg(&d_MSD[2*f+1])); //assuming white noise 
					if(temp_SNR > SNR) {
						SNR = temp_SNR;
						max_SNR_harmonic = f;
					}
				}
			}
		}
		
		pos = blockIdx.x*nSpectra + blockIdx.y*blockDim.x + threadIdx.x;
		d_output_SNR[pos] = SNR;
		d_output_harmonics[pos] = max_SNR_harmonic;
	}
}
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------



int Initiate_device(int device){
	int devCount;
	cudaGetDeviceCount(&devCount);
	if(device<devCount) {
		cudaSetDevice(device);
		return(0);
	}
	else return(1);	
}

int Check_free_memory(size_t total_input_FFT_size, size_t total_output_FFT_size){
	cudaError_t err_code;
	size_t free_mem, total_mem;
	err_code = cudaMemGetInfo(&free_mem,&total_mem);
	if(err_code!=cudaSuccess) {
		printf("CUDA ERROR!\n");
		return(1);
	}
	
	if(free_mem<(total_input_FFT_size+total_output_FFT_size)) {
		printf("ERROR: Not enough GPU memory\n");
		return(1);
	}
	
	return(0);
}

double stdev(std::vector<double> *times, double mean_time){
	double sum = 0;
	for(size_t i=0; i<times->size(); i++){
		double x = (times->operator[](i)-mean_time);
		sum = sum + x*x;
	}
	double stdev = sqrt( sum/((double) times->size()) );
	return(stdev);
}

// ***********************************************************************************
// ***********************************************************************************
// ***********************************************************************************

int Calculate_GPU_HRMS(float2 *h_input, float *h_output, Performance_results *HRMS_results, int device){
	int nElements  = HRMS_results->nElements;
	int nHarmonics = HRMS_results->nHarmonics;
	int nSeries    = HRMS_results->nSeries;
	int nRuns      = HRMS_results->nRuns;
	size_t input_size = nElements*nSeries*sizeof(float2);
	size_t power_size = nElements*nSeries*sizeof(float);
	size_t output_size = nElements*nSeries;
	GpuTimer timer, total_timer;
	total_timer.Start();
	
	Initiate_device(device);
	
	float2 *d_input;
	float *d_power;
	float *d_output_SNR;
	ushort *d_output_harmonics;
	
	cudaError_t err_code;
	err_code = cudaMalloc((void **) &d_input, input_size);
	if(err_code!=cudaSuccess) {
		printf("\nError in allocation of the device memory!\n");
		return(1);
	}
	err_code = cudaMalloc((void **) &d_power, power_size);
	if(err_code!=cudaSuccess) {
		printf("\nError in allocation of the device memory!\n");
		return(1);
	}
	err_code = cudaMalloc((void **) &d_output_SNR, output_size*sizeof(float));
	if(err_code!=cudaSuccess) {
		printf("\nError in allocation of the device memory!\n");
		return(1);
	}
	err_code = cudaMalloc((void **) &d_output_harmonics, output_size*sizeof(ushort));
	if(err_code!=cudaSuccess) {
		printf("\nError in allocation of the device memory!\n");
		return(1);
	}	
	
	err_code = cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
	if(err_code!=cudaSuccess) {
		printf("\nError in allocation of the device memory!\n");
		return(1);
	}
	
	//-------------------- cuFFT ----------------->
	cufftHandle plan;
	cufftResult cuFFT_error;
	cuFFT_error = cufftPlan1d(&plan, nElements, CUFFT_C2C, nSeries);
	double FFT_execution_time = 0;
	if (CUFFT_SUCCESS == cuFFT_error) {
		nvml_setup(device);
		for(int f=0; f<nRuns; f++){
			timer.Start();
			cufftExecC2C(plan, (cufftComplex *) d_input, (cufftComplex *) d_input, CUFFT_FORWARD);
			timer.Stop();
			FFT_execution_time += timer.Elapsed();
		}
		// stop before reset to default; kernel call is async
		cudaDeviceSynchronize();
		nvml_reset();
		FFT_execution_time = FFT_execution_time/((double) nRuns);
		HRMS_results->GPU_FFT_time = FFT_execution_time;
	}
	else printf("CUFFT error: Plan creation failed");
	cufftDestroy(plan);
	//--------------------------------------------<
	
	//------------- Power calculation ------------>
    int power_blocks_x, power_blocks_y;
	
    power_blocks_x = (nElements + 256 - 1)/256;
    power_blocks_y = nSeries;
	
    dim3 power_blockDim(256, 1, 1);
    dim3 power_gridSize(power_blocks_x ,power_blocks_y , 1);
	
	FFT_execution_time = 0;
	for(int f=0; f<nRuns; f++){
		timer.Start();
		GPU_simple_power_and_interbin_kernel<<< power_gridSize , power_blockDim >>>(d_input, d_power, nElements, 1);	
		timer.Stop();
		FFT_execution_time += timer.Elapsed();
	}
	FFT_execution_time = FFT_execution_time/((double) nRuns);
	HRMS_results->GPU_MSD_time = FFT_execution_time;	
	//--------------------------------------------<	

	//--------------------- MSD ------------------>
	int nBatches = 1;
	int MSD_size = MSD_RESULTS_SIZE*nBatches*sizeof(float);
	int MSD_elements_size = nBatches*sizeof(size_t);
	
	float *d_MSD;
	size_t *d_MSD_nElements;
	if ( cudaSuccess != cudaMalloc((void **) &d_MSD, MSD_size)) {
		printf("CUDA API error while allocating GPU memory\n");
	}
	if ( cudaSuccess != cudaMalloc((void **) &d_MSD_nElements, MSD_elements_size)) {
		printf("CUDA API error while allocating GPU memory\n");
	}
	
	MSD_Error MSD_error;
	MSD_Configuration MSD_conf;
	std::vector<size_t> dimensions={ (size_t) nSeries, (size_t) nElements};
	bool outlier_rejection = false;
	int offset = 0;
	double outlier_rejection_sigma = 3.0;
	MSD_error = MSD_conf.Create_MSD_Plan(dimensions, offset, outlier_rejection, outlier_rejection_sigma, nBatches);
	if(MSD_error!=MSDSuccess) Get_MSD_Error(MSD_error);
	FFT_execution_time = 0;
	for(int f=0; f<nRuns; f++){
		timer.Start();
		MSD_error = MSD_GetMeanStdev(d_MSD, d_MSD_nElements, d_power, MSD_conf);
		timer.Stop();
		FFT_execution_time += timer.Elapsed();
	}
	FFT_execution_time = FFT_execution_time/((double) nRuns);
	HRMS_results->GPU_MSD_time += FFT_execution_time;
	if(MSD_error!=MSDSuccess) Get_MSD_Error(MSD_error);
	//--------------------------------------------<	
	
	//--------------- Harmonic Sum --------------->
	int CT_nBlocks_x, CT_nBlocks_y;
	int Elements_per_block=CT_CORNER_BLOCKS*WARP;
	CT_nBlocks_x = (nElements + Elements_per_block - 1)/Elements_per_block;
	CT_nBlocks_y = (nSeries + WARP + 1)/WARP;
	dim3 CT_gridSize(CT_nBlocks_x, CT_nBlocks_y, 1);
    dim3 CT_blockSize(CT_NTHREADS, 1, 1);
	FFT_execution_time = 0;
	for(int f=0; f<nRuns; f++){
		timer.Start();
		corner_turn_SM_kernel<<< CT_gridSize, CT_blockSize >>>(d_power, (float *) d_input, nElements, nSeries);
		timer.Stop();
		FFT_execution_time += timer.Elapsed();
	}
	FFT_execution_time = FFT_execution_time/((double) nRuns);
	HRMS_results->GPU_HRMS_time = FFT_execution_time;	
	
	
	int HRMS_nBlocks_x, HRMS_nBlocks_y;
	HRMS_nBlocks_x = nElements;
    HRMS_nBlocks_y = (nSeries + PHS_NTHREADS - 1)/PHS_NTHREADS;
    dim3 HRMS_gridSize(HRMS_nBlocks_x, HRMS_nBlocks_y, 1);
    dim3 HRMS_blockSize(PHS_NTHREADS, 1, 1);
	
	FFT_execution_time = 0;
	for(int f=0; f<nRuns; f++){
		timer.Start();
		PHS_GPU_kernel<<< HRMS_gridSize, HRMS_blockSize >>>((float *) d_input, d_output_SNR, d_output_harmonics, d_MSD, nElements, nSeries, nHarmonics);
		timer.Stop();
		FFT_execution_time += timer.Elapsed();
	}
	FFT_execution_time = FFT_execution_time/((double) nRuns);
	HRMS_results->GPU_HRMS_time += FFT_execution_time;
	//--------------------------------------------<
	
	total_timer.Stop();
	HRMS_results->GPU_total_time = HRMS_results->GPU_HRMS_time + HRMS_results->GPU_MSD_time + HRMS_results->GPU_FFT_time;

	cudaFree(d_MSD);
	cudaFree(d_MSD_nElements);
	cudaFree(d_input);
	cudaFree(d_power);
	cudaFree(d_output_SNR);
	cudaFree(d_output_harmonics);
	
	return(0);
}
