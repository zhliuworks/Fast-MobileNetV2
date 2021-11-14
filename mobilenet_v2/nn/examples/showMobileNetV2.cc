#include "../MobileNetV2.h"

int main() {
    MobileNetV2 *mobileNetV2 = new MobileNetV2("../weights/");
    mobileNetV2->show();
    return 0;
}