all:
	make mobile

mobile:
	nvcc -o testCudnnMobileNetV2.o -lcudnn -lcublas --std=c++11 -O3 \
		testCudnnMobileNetV2.cu    \
		../../../nn/MobileNetV2.cc \

clean:
	rm *.o