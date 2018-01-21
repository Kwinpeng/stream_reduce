CC = nvcc
FLAGS = -gencode=arch=compute_50,code=sm_50 \
		-gencode=arch=compute_52,code=sm_52 \
		-gencode=arch=compute_60,code=sm_60 \
		-gencode=arch=compute_61,code=sm_61

reduce: reduce.cu
	$(CC) $(FLAGS) reduce.cu -o reduce

clean:
	@rm reduce
