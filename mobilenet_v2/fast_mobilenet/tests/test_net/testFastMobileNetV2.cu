#include "../../fastMobileNetV2.cuh"

const int NUM_TEST = 10;
const std::string inputs_path = "./inputs/";
const std::string outputs_path = "./outputs/";

__host__ int main() {
    TensorShape input_shape(1, 3, 244, 244);
    float *input_data = (float*)malloc(1 * 3 * 244 * 244 * sizeof(float));
    int output_shape = 1000;
    float *output_data = (float*)malloc(1 * 1000 * sizeof(float));

    std::ifstream infile;
    std::ofstream outfile;
    int k;

    for (int i = 0; i < NUM_TEST; i++) {
        infile.open(inputs_path + std::to_string(i));
        assert(infile.is_open());
        k = 0;
        while (!infile.eof()) {
            infile >> input_data[k++];
        }
        infile.close();

        fastMobileNetV2(input_data, input_shape, "../../../nn/weights/", output_data, output_shape);

        outfile.open(outputs_path + std::to_string(i));
        assert(outfile.is_open());
        for (int j = 0; j < 1000; j++) {
            outfile << output_data[j] << ' ';
        }
        outfile.close();
        std::cout << "Finished " + inputs_path + std::to_string(i) << std::endl;
    }

    return 0;
}