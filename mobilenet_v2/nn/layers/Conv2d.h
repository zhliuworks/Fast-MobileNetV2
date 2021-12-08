#ifndef _NN_CONV_2D
#define _NN_CONV_2D

#include "Layer.h"

class Conv2d : public Layer {
public:
    explicit Conv2d(int in_channels=1,
                    int out_channels=1,
                    int kernel_size=1,
                    int stride=1,
                    int padding=0,
                    int group=1) :
            in_channels(in_channels / group),
            out_channels(out_channels),
            kernel_size(kernel_size),
            stride(stride),
            padding(padding),
            group(group) {

        size = out_channels * in_channels * kernel_size * kernel_size;
        weight = new float[size];
        bias = new float[out_channels];
    }

    Conv2d &operator=(Conv2d conv2d) {
        in_channels = conv2d.getInChannels();
        out_channels = conv2d.getOutChannels();
        kernel_size = conv2d.getKernelSize();
        stride = conv2d.getStride();
        padding = conv2d.getPadding();
        group = conv2d.getGroup();

        delete []weight;
        delete []bias;

        size = out_channels * in_channels * kernel_size * kernel_size;
        weight = new float[size];
        bias = new float[out_channels];
        return *this;
    }

    ~Conv2d() {
        delete []weight;
        delete []bias;
    }

    void show() {
        std::cout << "\033[36mConv2d(\033[0m"
                  << in_channels << ", "
                  << out_channels << ", "
                  << kernel_size << ", "
                  << stride << ", "
                  << padding << ", "
                  << group
                  << "\033[36m)\033[0m";
    }

    int getType() {
        return 0;
    }

    std::vector<std::pair<float*, std::vector<int>>> getParameters() {
        return {
            std::make_pair(weight, std::vector<int>{out_channels, in_channels, kernel_size, kernel_size}),
            std::make_pair(bias, std::vector<int>{out_channels})
        };
    }

    float *getWeight() { return weight; }
    float *getBias() { return bias; }
    int getInChannels() { return in_channels; }
    int getOutChannels() { return out_channels; }
    int getKernelSize() { return kernel_size; }
    int getStride() { return stride; }
    int getPadding() { return padding; }
    int getGroup() { return group; }
    int getSize() { return size; }

private:
    float *weight;
    float *bias;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int group;
    int size;
};

#endif // _NN_CONV_2D