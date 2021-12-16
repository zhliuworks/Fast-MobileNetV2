all:
	make mobile

mobile:
	nvcc -o testFastMobileNetV2.o --std=c++11 -O3 \
		testFastMobileNetV2.cu    \
		../../../nn/MobileNetV2.cc \

clean:
	rm *.o