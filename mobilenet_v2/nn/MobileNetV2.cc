#include "MobileNetV2.h"
#include <iomanip>

MobileNetV2::MobileNetV2(std::string weights_path) : weights_path(weights_path) {

    int i_conv = 0, i_relu = 0, i_add = 0;

    // First Conv2d + ReLU6
    Conv2dLayers[i_conv] = Conv2d(3, 32, 3, 2, 1);
    GetConvWeights(Conv2dLayers[i_conv].getWeight(), Conv2dLayers[i_conv].getBias(), i_conv);
    curr = entry = &Conv2dLayers[i_conv++];
    curr = curr->setNext(ReLU6Layers[i_relu++]);

    // Inverted Residuals
    std::vector<std::vector<int>> inverted_residual_settings{
        /*
        t, c, n, s
        # t: expansion ratio
        # c: number of output channels
        # n: repeated times of identical layers
        # s: stride of first layer
        */
        {1,  16, 1, 1},
        {6,  24, 2, 2},
        {6,  32, 3, 2},
        {6,  64, 4, 2},
        {6,  96, 3, 1},
        {6, 160, 3, 2},
        {6, 320, 1, 1},
    };

    int input_dim = 32;
    int stride;

    for (auto &setting : inverted_residual_settings) {
        for (int i = 0; i < setting[2]; i++) {
            stride = (i == 0) ? setting[3] : 1;
            BuildInvertedResidual(input_dim, setting[1], setting[0], stride,
                                  i_conv, i_relu, i_add, i);
            input_dim = setting[1];
        }
    }

    // Last Conv2d + ReLU6
    Conv2dLayers[i_conv] = Conv2d(320, 1280, 1, 1, 0);
    GetConvWeights(Conv2dLayers[i_conv].getWeight(), Conv2dLayers[i_conv].getBias(), i_conv);
    curr = curr->setNext(Conv2dLayers[i_conv++]);
    curr = curr->setNext(ReLU6Layers[i_relu++]);

    // Global Average Pool
    curr = curr->setNext(GlobalAveragePoolLayer);

    // linear
    LinearLayer = Linear(1280, 1000);
    GetGemmWeight(LinearLayer.getWeight(), LinearLayer.getBias());
    curr = curr->setNext(LinearLayer);

    // assert index
    assert(i_conv == NUM_CONV);
    assert(i_relu == NUM_RELU);
    assert(i_add == NUM_ADD);
}


void MobileNetV2::show() {
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "                 MobileNetV2                 " << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    curr = entry;
    int i = 0;
    while (curr) {
        std::cout << std::setw(2) << i++;
        std::cout << ": ";
        curr->show();
        std::cout << std::endl;
        curr = curr->getNext();
    }
    std::cout << "---------------------------------------------" << std::endl;
}


void MobileNetV2::GetConvWeights(float *weight, float *bias, int idx) {
    std::ifstream infile;
    int k;

    // weight
    infile.open(weights_path + "conv/" + std::to_string(idx) + ".w");
    assert(infile.is_open());
    k = 0;
    while (!infile.eof()) {
        infile >> weight[k++];
    }
    infile.close();

    // bias
    infile.open(weights_path + "conv/" + std::to_string(idx) + ".b");
    assert(infile.is_open());
    k = 0;
    while (!infile.eof()) {
        infile >> bias[k++];
    }
    infile.close();
}


void MobileNetV2::GetGemmWeight(float *weight, float *bias) {
    std::ifstream infile;
    int k;

    // weight
    infile.open(weights_path + "gemm/0.w");
    assert(infile.is_open());
    k = 0;
    while (!infile.eof()) {
        infile >> weight[k++];
    }
    infile.close();

    // bias
    infile.open(weights_path + "gemm/0.b");
    assert(infile.is_open());
    k = 0;
    while (!infile.eof()) {
        infile >> bias[k++];
    }
    infile.close();
}


void MobileNetV2::BuildInvertedResidual(int input_dim, int output_dim, int expansion_ratio, int stride,
                                        int &i_conv, int &i_relu, int &i_add, int i) {

    int hidden_dim = input_dim * expansion_ratio;

    if (expansion_ratio == 1) {
        // depthwise convolution
        Conv2dLayers[i_conv] = Conv2d(hidden_dim, hidden_dim, 3, stride, 1, hidden_dim);
        GetConvWeights(Conv2dLayers[i_conv].getWeight(), Conv2dLayers[i_conv].getBias(), i_conv);
        curr = curr->setNext(Conv2dLayers[i_conv++]);
        curr = curr->setNext(ReLU6Layers[i_relu++]);
        // pointwise convolution, linear
        Conv2dLayers[i_conv] = Conv2d(hidden_dim, output_dim, 1, 1, 0);
        GetConvWeights(Conv2dLayers[i_conv].getWeight(), Conv2dLayers[i_conv].getBias(), i_conv);
        curr = curr->setNext(Conv2dLayers[i_conv++]);
    } else {
        // pointwise convolution, ReLU6
        Conv2dLayers[i_conv] = Conv2d(input_dim, hidden_dim, 1, 1, 0);
        GetConvWeights(Conv2dLayers[i_conv].getWeight(), Conv2dLayers[i_conv].getBias(), i_conv);
        curr = curr->setNext(Conv2dLayers[i_conv++]);
        curr = curr->setNext(ReLU6Layers[i_relu++]);
        // depthwise convolution
        Conv2dLayers[i_conv] = Conv2d(hidden_dim, hidden_dim, 3, stride, 1, hidden_dim);
        GetConvWeights(Conv2dLayers[i_conv].getWeight(), Conv2dLayers[i_conv].getBias(), i_conv);
        curr = curr->setNext(Conv2dLayers[i_conv++]);
        curr = curr->setNext(ReLU6Layers[i_relu++]);
        // pointwise convolution, linear
        Conv2dLayers[i_conv] = Conv2d(hidden_dim, output_dim, 1, 1, 0);
        GetConvWeights(Conv2dLayers[i_conv].getWeight(), Conv2dLayers[i_conv].getBias(), i_conv);
        curr = curr->setNext(Conv2dLayers[i_conv++]);
    }

    if (stride == 1 && input_dim == output_dim) {
        if (i == 1) {
            ResidualAddLayers[i_add].setResidual(Conv2dLayers[i_conv - 4]);
            Conv2dLayers[i_conv - 4].setIsBypass();
        } else {
            ResidualAddLayers[i_add].setResidual(ResidualAddLayers[i_add - 1]);
            ResidualAddLayers[i_add - 1].setIsBypass();
        }
        curr = curr->setNext(ResidualAddLayers[i_add++]);
    }
}