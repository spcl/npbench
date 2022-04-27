#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "HalideBuffer.h"
#include "halide_image_io.h"

#include "iir_blur.h"
#include "iir_blur_auto_schedule.h"

#include "../../benchmark.h"

using namespace Halide::Runtime;

int main(int argc, char **argv) {
    const std::string input_image_path = "../image_processing/images/rgb.png";
    const std::string output_image_path = "./iir_blur_output.png";

    Buffer<uint8_t, 3> input = Halide::Tools::load_image(input_image_path);
    Buffer<uint8_t, 3> output(input.width(), input.height(), input.channels());

    // Timing code
    const float alpha = 0.1f;

    /*
    double base_time = benchmark(30, [&]() {
        iir_blur(input, alpha, output);
        output.device_sync();
    });
    printf("Base time: %gms\n", base_time * 1e3);
    */

    double auto_time = benchmark(30, [&]() {
        iir_blur_auto_schedule(input, alpha, output);
        output.device_sync();
    });
    printf("Auto time: %gms\n", auto_time * 1e3);

    Halide::Tools::save_image(output, output_image_path);

    printf("Success!\n");
    return 0;
}
