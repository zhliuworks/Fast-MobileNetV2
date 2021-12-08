#ifndef _NN_LAYER
#define _NN_LAYER

#include <memory>
#include <vector>
#include <cassert>
#include <iostream>

class Layer {
public:
    Layer() {
        next = nullptr;
        bypass = false;
    }
    ~Layer() { next = nullptr; }
    
    // get next layer
    Layer *getNext() { return next; }

    // set next layer
    Layer *setNext(Layer &nextLayer) {
        next = &nextLayer;
        return next;
    }

    // whether added by ResidualAdd
    bool getIsBypass() { return bypass; }

    // when added by ResidualAdd
    void setIsBypass() { bypass = true; }

    // get residual layer
    Layer *getResidual() {
        throw std::runtime_error("not implement error\n");
    }

    // set residual layer
    void setResidual(Layer &resLayer) {
        throw std::runtime_error("not implement error\n");
    }

    // show
    virtual void show() {
        throw std::runtime_error("not implement error\n");
    }

    // layer type
    virtual int getType() {
        throw std::runtime_error("not implement error\n");
    }

    // get parameters
    virtual std::vector<std::pair<float*, std::vector<int>>> getParameters() {
        throw std::runtime_error("not implement error\n");
    };
    virtual float *getWeight() {
        throw std::runtime_error("not implement error\n");
    }
    virtual float *getBias() {
        throw std::runtime_error("not implement error\n");
    }
    virtual int getInChannels() {
        throw std::runtime_error("not implement error\n");
    }
    virtual int getOutChannels() {
        throw std::runtime_error("not implement error\n");
    }
    virtual int getKernelSize() {
        throw std::runtime_error("not implement error\n");
    }
    virtual int getStride() {
        throw std::runtime_error("not implement error\n");
    }
    virtual int getPadding() {
        throw std::runtime_error("not implement error\n");
    }
    virtual int getGroup() {
        throw std::runtime_error("not implement error\n");
    }
    virtual int getSize() {
        throw std::runtime_error("not implement error\n");
    }
    virtual int getInFeatures() {
        throw std::runtime_error("not implement error\n");
    }
    virtual int getOutFeatures() {
        throw std::runtime_error("not implement error\n");
    }

protected:
    Layer *next;
    bool bypass;
};

#endif // _NN_LAYER