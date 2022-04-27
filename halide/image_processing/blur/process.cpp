#include <chrono>
#include <cstdio>

#include "blur.h"
#include "blur_auto_schedule.h"

#include "HalideBuffer.h"
#include "halide_image_io.h"

#include "../../benchmark.h"

using namespace Halide::Runtime;

int main(int argc, char **argv) {
    const std::string input_image_path = "../image_processing/images/gray.png";
    const std::string output_image_path = "./blur_output.png";

    Buffer<uint8_t, 2> input = Halide::Tools::load_image(input_image_path);
    Buffer<uint8_t, 2> output(input.width() - 2, input.height() - 2);
    output.set_min(1, 1);

    // Timing code

    /*
    double base_time = benchmark(30, [&]() {
        blur(input, output);
        output.device_sync();
    });
    printf("Base: %gms\n", base_time * 1e3);
    */

    // Auto-scheduled version
    double auto_time = benchmark(30, [&]() {
        blur_auto_schedule(input, output);
        output.device_sync();
    });
    printf("Auto time: %gms\n", auto_time * 1e3);

    Halide::Tools::save_image(output, output_image_path);

    printf("Success!\n");
    return 0;
}
