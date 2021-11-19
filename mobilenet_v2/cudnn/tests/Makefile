all:
	make conv
	make gemm
	make pool

conv:
	nvcc -o testConv.o -lcudnn testConv.cu

gemm:
	nvcc -o testGemm.o -lcublas testGemm.cu

pool:
	nvcc -o testPool.o -lcudnn testPool.cu
	
clean:
	rm *.o