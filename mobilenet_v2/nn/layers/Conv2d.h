#ifndef _NN_CONV_2D
#define _NN_CONV_2D

#include "Layer.h"

class Conv2d : public Layer {
public:
    explicit Conv2d(int in_channels = 1,
                    int out_channels = 1,
                    int kernel_size = 1,
                    int stride = 1,
                    int padding = 0,
                    int groups = 1)
            : in_channels(in_channels / groups),
            out_channels(out_channels),
            kernel_size(kernel_size),
            stride(stride),
            padding(padding),
            groups(groups) {

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
        groups = conv2d.getGroups();

        delete []weight;
        delete []bias;

        size = out_channels * in_channels * kernel_size * kernel_size;
        weight = new float[size];
        bias = new float[out_channels];
    }

    ~Conv2d() {
        delete []weight;
        delete []bias;
    }
    
    std::vector<std::pair<float*, std::vector<int>>> getParameters() {
        return {
            std::make_pair(weight, std::vector<int>{out_channels, in_channels, kernel_size, kernel_size}),
            std::make_pair(bias, std::vector<int>{out_channels})
        };
    }

    // show
    void show() {
        std::cout << "\033[36mConv2d(\033[0m"
                  << in_channels << ", "
                  << out_channels << ", "
                  << kernel_size << ", "
                  << stride << ", "
                  << padding << ", "
                  << groups
                  << "\033[36m)\033[0m";
    }

    float *getWeight() { return weight; }
    float *getBias() { return bias; }
    int getInChannels() { return in_channels; }
    int getOutChannels() { return out_channels; }
    int getKernelSize() { return kernel_size; }
    int getStride() { return stride; }
    int getPadding() { return padding; }
    int getGroups() { return groups; }
    int getSize() { return size; }

private:
    float *weight;
    float *bias;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int groups;
    int size;
};

#endif // _NN_CONV_2D