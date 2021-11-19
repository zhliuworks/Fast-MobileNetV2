# cuda
cat /usr/local/cuda/version.txt
# cudnn < 8.0
cat /usr/local/cuda/include/cudnn.h | grep '#define CUDNN_MAJOR' -A 2 || \
# cudnn >= 8.0
cat /usr/local/cuda/include/cudnn_version.h | grep '#define CUDNN_MAJOR' -A 2