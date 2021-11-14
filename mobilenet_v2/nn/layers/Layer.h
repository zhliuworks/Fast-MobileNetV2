#ifndef _NN_LAYER
#define _NN_LAYER

#include <memory>
#include <vector>
#include <cassert>
#include <iostream>

class Layer {
public:
    Layer() { next = nullptr; }
    ~Layer() { next = nullptr; }

    // get parameters and shapes
    virtual std::vector<std::pair<float*, std::vector<int>>> getParameters() {
        throw std::runtime_error("not implement error\n");
    };
    
    // get next layer
    Layer *getNext() { return next; }

    // set next layer
    Layer *setNext(Layer &nextLayer) {
        next = &nextLayer;
        return next;
    }

    // show
    virtual void show() {
        throw std::runtime_error("not implement error\n");
    }

protected:
    Layer *next;
};

#endif // _NN_LAYER