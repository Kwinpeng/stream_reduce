CC = nvcc

MGPU_DIR := external/moderngpu

INCLUDES := -I $(MGPU_DIR)/include

GENCODE_SM50  := -gencode arch=compute_30,code=sm_30
#GENCODE_SM52  := -gencode arch=compute_52,code=sm_52
#GENCODE_SM60  := -gencode arch=compute_60,code=sm_60

GENCODE_FLAGS := $(GENCODE_SM50) $(GENCODE_SM52) $(GENCODE_SM60)

NVCCFLAGS	  += -std=c++11 $(GENCODE_FLAGS) $(INCLUDES)

all: Reduce

Reduce: mgpucontext.o mgpuutil.o reduce.o
	$(CC) $(NVCCFLAGS) -o $@ $+

reduce.o: reduce.cu
	$(CC) $(NVCCFLAGS) -o $@ -c $<

mgpucontext.o: $(MGPU_DIR)/src/mgpucontext.cu
	$(CC) $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: $(MGPU_DIR)/src/mgpuutil.cpp
	$(CC) $(NVCCFLAGS) -o $@ -c $<

clean:
	@rm *.o Reduce
