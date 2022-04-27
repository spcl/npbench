#include <chrono>
#include <cstdio>

#include "conv2d_bias.h"
#include "conv2d_bias_auto_schedule.h"

#include "HalideBuffer.h"

#include "../../benchmark.h"

using namespace Halide::Runtime;

int main(int argc, char **argv) {
    const int N = 8, CI = 3, CO = 16, W = 256, H = 256, K = 20;

    const int border = K - 1;

    Buffer<float, 4> input(CI, W, H, N);
    Buffer<float, 4> filter(CI, K, K, CO);
    Buffer<float, 1> bias(CO);
    Buffer<float, 4> output(CO, W - border, H - border, N);

    for (int n = 0; n < N; n++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                for (int c = 0; c < CI; c++) {
                    input(c, x, y, n) = rand();
                }
            } 
        }
    }

    for (int co = 0; co < CO; co++) {
        for (int y = 0; y < K; y++) {
            for (int x = 0; x < K; x++) {
                for (int ci = 0; ci < CI; ci++) {
                    filter(ci, x, y, co) = rand();
                }
            }
        }
    }

    for (int x = 0; x < CO; x++) {
        bias(x) = rand();
    }

    // Timing code

    /*
    double base_time = benchmark(30, [&]() {
        conv2d_bias(input, filter, bias, output);
        output.device_sync();
    });
    printf("Base time: %gms\n", base_time * 1e3);
    */

    double auto_time = benchmark(30, [&]() {
        conv2d_bias_auto_schedule(input, filter, bias, output);
        output.device_sync();
    });
    printf("Auto time: %gms\n", auto_time * 1e3);

    printf("Success!\n");
    return 0;
}
