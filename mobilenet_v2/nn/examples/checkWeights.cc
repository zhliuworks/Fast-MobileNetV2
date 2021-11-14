#include "../MobileNetV2.h"
#define EPS 1e-6

int main() {
    std::string weights_path = "../weights/";
    MobileNetV2 *mobileNetV2 = new MobileNetV2(weights_path);
    auto convLayers = mobileNetV2->getConv2dLayers();
    float *weight, *bias;
    int size;
    std::ifstream infile;
    float curr;

    for (int i = 0; i < NUM_CONV; i++) {
        weight = convLayers[i].getWeight();
        size = convLayers[i].getSize();
        infile.open(weights_path + "conv/" + std::to_string(i) + ".w");
        assert(infile.is_open());
        for (int j = 0; j < size; j++) {
            infile >> curr;
            if (abs(curr - weight[j]) > EPS) {
                std::cout << "conv w diff: " << curr << '\t' << weight[j] << std::endl;
            }
        }
        infile.close();

        bias = convLayers[i].getBias();
        size = convLayers[i].getOutChannels();
        infile.open(weights_path + "conv/" + std::to_string(i) + ".b");
        assert(infile.is_open());
        for (int j = 0; j < size; j++) {
            infile >> curr;
            if (abs(curr - bias[j]) > EPS) {
                std::cout << "conv b diff: " << curr << '\t' << bias[j] << std::endl;
            }
        }
        infile.close();
    }

    auto linearLayer = mobileNetV2->getLinearLayer();
    weight = linearLayer.getWeight();
    size = linearLayer.getSize();
    infile.open(weights_path + "gemm/0.w");
    assert(infile.is_open());    
    for (int j = 0; j < size; j++) {
        infile >> curr;
        if (abs(curr - weight[j]) > EPS) {
            std::cout << "gemm w diff: " << curr << '\t' << weight[j] << std::endl;
        }
    }
    infile.close();

    bias = linearLayer.getBias();
    size = linearLayer.getOutFeatures();
    infile.open(weights_path + "gemm/0.b");
    assert(infile.is_open());
    for (int j = 0; j < size; j++) {
        infile >> curr;
        if (abs(curr - bias[j]) > EPS) {
            std::cout << "gemm b diff: " << curr << '\t' << bias[j] << std::endl;
        }
    }
    infile.close();

    return 0;
}