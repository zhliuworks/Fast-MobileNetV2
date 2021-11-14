#ifndef _NN_GLOBAL_AVERAGE_POOL
#define _NN_GLOBAL_AVERAGE_POOL

#include "Layer.h"

class GlobalAveragePool : public Layer {
public:
    explicit GlobalAveragePool() {}
    ~GlobalAveragePool() {}

    // show
    void show() {
        std::cout << "\033[34mGlobalAveragePool\033[0m";
    }
};

#endif // _NN_GLOBAL_AVERAGE_POOL