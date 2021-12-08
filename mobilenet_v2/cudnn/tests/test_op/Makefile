all:
	make conv
	make gemm
	make pool
	make add

conv:
	nvcc -o testConv.o -lcudnn --std=c++11 testConv.cu

gemm:
	nvcc -o testGemm.o -lcublas --std=c++11 testGemm.cu

pool:
	nvcc -o testPool.o -lcudnn --std=c++11 testPool.cu

add:
	nvcc -o testAdd.o -lcudnn --std=c++11 testAdd.cu

clean:
	rm *.o