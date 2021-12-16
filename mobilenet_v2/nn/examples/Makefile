BASE = ..

all:
	make show
	make check

show:
	g++ -o show.out --std=c++11 -O3                \
				showMobileNetV2.cc                 \
				${BASE}/MobileNetV2.h              \
				${BASE}/MobileNetV2.cc             \
				${BASE}/layers/Layer.h             \
				${BASE}/layers/Conv2d.h            \
				${BASE}/layers/GlobalAveragePool.h \
				${BASE}/layers/Linear.h            \
				${BASE}/layers/ReLU6.h             \
				${BASE}/layers/ResidualAdd.h       \

check:
	g++ -o check.out --std=c++11 -O3               \
				checkWeights.cc                    \
				${BASE}/MobileNetV2.h              \
				${BASE}/MobileNetV2.cc             \
				${BASE}/layers/Layer.h             \
				${BASE}/layers/Conv2d.h            \
				${BASE}/layers/GlobalAveragePool.h \
				${BASE}/layers/Linear.h            \
				${BASE}/layers/ReLU6.h             \
				${BASE}/layers/ResidualAdd.h	

clean:
	rm *.out