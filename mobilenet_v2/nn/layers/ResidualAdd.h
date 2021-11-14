#ifndef _NN_RESIDUAL_ADD
#define _NN_RESIDUAL_ADD

#include "Layer.h"

class ResidualAdd : public Layer {
public:
    explicit ResidualAdd() { residual = nullptr; }
    ~ResidualAdd() { residual = nullptr; }

    // get residual layer
    Layer *getResidual() { return residual; }

    // set residual layer
    void setResidual(Layer &resLayer) { residual = &resLayer; }

    // show
    void show() {
        std::cout << "\033[33mResidualAdd(\033[0m";
        if (residual) {
            residual->show();
        }
        std::cout << "\033[33m)\033[0m";
    }

private:
    Layer *residual;
};

#endif // _NN_RESIDUAL_ADD