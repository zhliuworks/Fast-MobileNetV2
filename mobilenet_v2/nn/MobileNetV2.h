#ifndef _NN_MOBILE_NET_V2
#define _NN_MOBILE_NET_V2

#include "layers/Conv2d.h"
#include "layers/GlobalAveragePool.h"
#include "layers/Linear.h"
#include "layers/ReLU6.h"
#include "layers/ResidualAdd.h"
#include <fstream>
#include <cassert>
#include <string>

#define NUM_CONV 52
#define NUM_RELU 35
#define NUM_ADD  10

class MobileNetV2 {
public:
    // build model topology and load parameters
    explicit MobileNetV2(std::string weights_path = "./weights/");

    // get entry
    Layer *getEntry() { return entry; }

    // get layers
    Conv2d *getConv2dLayers() { return Conv2dLayers; }
    ReLU6 *getReLU6Layers() { return ReLU6Layers; }
    ResidualAdd *getResidualAddLayers() { return ResidualAddLayers; }
    GlobalAveragePool getGlobalAveragePoolLayer() { return GlobalAveragePoolLayer; }
    Linear getLinearLayer() { return LinearLayer; }

    // show
    void show();

private:
    std::string weights_path;
    Layer *entry;
    Layer *curr;
    Conv2d Conv2dLayers[NUM_CONV];
    ReLU6 ReLU6Layers[NUM_RELU];
    ResidualAdd ResidualAddLayers[NUM_ADD];
    GlobalAveragePool GlobalAveragePoolLayer;
    Linear LinearLayer;

    void GetConvWeights(float *weight, float *bias, int idx);
    void GetGemmWeight(float *weight, float *bias);
    void BuildInvertedResidual(int input_dim, int output_dim, int stride, int expansion_ratio,
                               int &i_conv, int &i_relu, int &i_add);
};

#endif // _NN_MOBILE_NET_V2