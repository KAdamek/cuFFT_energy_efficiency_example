INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcuda -lcufft -L. -lMSDGPU

# use this compilers
# g++ just because the file write
GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_70 --ptxas-options=-v -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -Wextra $(INC)

ANALYZE = HRMS_benchmark.exe


ifdef reglim
NVCCFLAGS += --maxrregcount=$(reglim)
endif

ifdef fastmath
NVCCFLAGS += --use_fast_math
endif

all: clean msdlib analyze

analyze: HRMS_benchmark.o HRMS.o Makefile
	$(NVCC) -o $(ANALYZE) HRMS_benchmark.o HRMS.o $(LIB) $(NVCCFLAGS) 

HRMS.o: timer.h utils_cuda.h
	$(NVCC) -c HRMS.cu $(NVCCFLAGS)
	
HRMS_benchmark.o: HRMS_benchmark.cpp
	$(GCC) -c HRMS_benchmark.cpp $(GCC_OPTS)

msdlib: MSD-library.o 
	
MSD-library.o: timer.h MSD_Configuration.h MSD_GPU_kernels_2d.cu
	$(NVCC) -c MSD_GPU_host_code.cu $(NVCCFLAGS)
	ar rsv libMSDGPU.a MSD_GPU_host_code.o
	rm *.o


clean:	
	rm -f *.o *.~ $(ANALYZE)


