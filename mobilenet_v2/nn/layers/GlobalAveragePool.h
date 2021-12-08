#ifndef _NN_GLOBAL_AVERAGE_POOL
#define _NN_GLOBAL_AVERAGE_POOL

#include "Layer.h"

class GlobalAveragePool : public Layer {
public:
    explicit GlobalAveragePool() {}
    ~GlobalAveragePool() {}

    void show() {
        std::cout << "\033[34mGlobalAveragePool\033[0m";
    }

    int getType() {
        return 1;
    }
};

#endif // _NN_GLOBAL_AVERAGE_POOL