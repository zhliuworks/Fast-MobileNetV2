all:
	make conv
	make gemm
	make pool
	make add
	make im2col

conv:
	nvcc -o testConv.o --std=c++11 -O3 testConv.cu

gemm:
	nvcc -o testGemm.o --std=c++11 -O3 testGemm.cu

pool:
	nvcc -o testPool.o --std=c++11 -O3 testPool.cu

add:
	nvcc -o testAdd.o --std=c++11 -O3 testAdd.cu

im2col:
	nvcc -o testIm2Col.o --std=c++11 -O3 testIm2Col.cu

clean:
	rm *.o