#!/bin/bash

#to run the script: sh call.sh memory_file.txt
#where the memory_file is a list of permited memory settings on the gpu

RED='\033[0;31m'
NC='\033[0m'

echo "Starting"
# setup your id of the card and the base of memory frequency
CARD=1
FREQ_MEM=877
TYPE=C2C
ID=V100
PREC=d
CORE_CLOCK=255
NRUNS=200

#for LENGTH in 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152
#for LENGTH in 25 27 49 81 121 232 243 343 625 729 1331 2187 2401 3125 6561 14641 16807 19683 59049 117649 161051 531441 823543 1771561 139 19321 2685619
for HARM in 2 4 8 16 32  #27 81 177147 1594323 125 3125 15625 78125 390625 1953125 121 
#for LENGTH in 48 96 192 384 768 1536 3072 6144 12288 24576 49152 98304 196608 393216 786432 1572864
do
rm timing-${ID}-${PREC}-${TYPE}-${HARM}-${CORE_CLOCK}.txt

#run the logger on the energy profiling
	nvidia-smi --query-gpu=timestamp,power.draw,clocks.current.sm,clocks.current.memory --format=csv,noheader,nounits -i $CARD -f nvidiasmi-${ID}-${PREC}-${TYPE}-${HARM}-${CORE_CLOCK}.txt -lms 10 &
	PID=$(echo $!)
	echo "Logging process id: $PID"
	sleep 5

printf "${RED}------ Running the ${HARM} ------${NC}\n"	
#run for each permitted memory a defined cuFFT
#need to change the length of cuFFT, now is setup to run on 2GB of data, i.e. number of FFT auto computed to fit 2GB with the set length
#        echo "Text read from file: $line"
#	CORE_MEM=$(echo $line | awk '{print $3}')
#	echo $CORE_MEM
#	nvidia-smi -i $CARD -ac $FREQ_MEM,$CORE_MEM
	TIMESTAMP=$(date +"%Y/%m/%d %H:%M:%S.%3N")
#	nvprof --print-gpu-trace -u ms --csv ../cuFFT_benchmark.exe ${LENGTH} 0 0 -2 200 ${PREC} ${TYPE} ${CARD} 2>out.nvprof
	nvprof --print-gpu-trace -u ms --csv ../HRMS_benchmark.exe 500000 8 1024 ${NRUNS} ${CORE_CLOCK} ${CARD} 2>out.nvprof
	START_LINE=$[$(awk '/Start/{ print NR; exit }' out.nvprof) + 2]
	#in csv mode there is no line with Regs
	#	END_LINE=$[$(awk '/Regs:/{ print NR; exit }' out.nvprof) - 3]
	END_LINE=$[$(cat out.nvprof | wc -l) - 0]
	START_TIME=$(head -n $START_LINE out.nvprof | tail -n 1 | awk -F "," '{print $1}')
	END_TIME=$(head -n $END_LINE out.nvprof | tail -n 1 | awk -F "," '{print $1}')
	END_PLUS_TIME=$(head -n $END_LINE out.nvprof | tail -n 1 | awk -F "," '{print $2}')
	sleep 5
	echo ${TIMESTAMP}"," ${START_TIME}"," ${END_TIME}"," ${END_PLUS_TIME}"," ${HARM}>> timing-${ID}-${PREC}-${TYPE}-${HARM}-${CORE_CLOCK}.txt

# clean-up 
#nvidia-smi -i $CARD -rac
kill -2 $PID
done
