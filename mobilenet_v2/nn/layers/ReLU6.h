#ifndef _NN_RELU6
#define _NN_RELU6

#include "Layer.h"

class ReLU6 : public Layer {
public:
    explicit ReLU6() {}
    ~ReLU6() {}

    // show
    void show() {
        std::cout << "\033[35mReLU6\033[0m";
    }
};

#endif // _NN_RELU6