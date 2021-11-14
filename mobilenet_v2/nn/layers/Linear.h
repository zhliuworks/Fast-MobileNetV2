#ifndef _NN_LINEAR
#define _NN_LINEAR

#include "Layer.h"

class Linear : public Layer {
public:
    explicit Linear(int in_features = 1, int out_features = 1)
            : in_features(in_features), out_features(out_features) {
        
        size = out_features * in_features;
        weight = new float[size];
        bias = new float[out_features];
    }

    Linear &operator=(Linear linear) {
        in_features = linear.getInFeatures();
        out_features = linear.getOutFeatures();

        delete []weight;
        delete []bias;

        size = out_features * in_features;
        weight = new float[size];
        bias = new float[out_features];
    }

    ~Linear() {
        delete []weight;
        delete []bias;
    }

    std::vector<std::pair<float*, std::vector<int>>> getParameters() {
        return {
            std::make_pair(weight, std::vector<int>{out_features, in_features}),
            std::make_pair(bias, std::vector<int>{out_features})
        };        

    }

    // show
    void show() {
        std::cout << "\033[32mLinear(\033[0m"
                  << in_features << ", "
                  << out_features
                  << "\033[32m)\033[0m";
    }

    float *getWeight() { return weight; }
    float *getBias() { return bias; }
    int getInFeatures() { return in_features; }
    int getOutFeatures() { return out_features; }
    int getSize() { return size; }

private:
    float *weight;
    float *bias;
    int in_features;
    int out_features;
    int size;
};

#endif // _NN_LINEAR