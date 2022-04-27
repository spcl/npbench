#include <chrono>
#include <cstdio>

#include "mlp.h"
#include "mlp_auto_schedule.h"

#include "HalideBuffer.h"

#include "../../benchmark.h"

using namespace Halide::Runtime;

int main(int argc, char **argv) {
    const int N = 8, C = 3, H1 = 30000, H2 = 10000, H3 = 1000;

    Buffer<float, 2> input(C, N);
    Buffer<float, 2> w1(C, H1);
    Buffer<float, 1> b1(H1);
    Buffer<float, 2> w2(H1, H2);
    Buffer<float, 1> b2(H2);
    Buffer<float, 2> w3(H2, H3);
    Buffer<float, 1> b3(H3);
    Buffer<float, 2> output(H3, N);

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            input(c, n) = rand();
        }
    }

    for (int h1 = 0; h1 < H1; h1++) {
        for (int c = 0; c < C; c++) {
            w1(c, h1) = rand();
        }
    }
    for (int x = 0; x < H1; x++) {
        b1(x) = rand();
    }

    for (int h2 = 0; h2 < H2; h2++) {
        for (int h1 = 0; h1 < H1; h1++) {
            w2(h1, h2) = rand();
        }
    }
    for (int x = 0; x < H2; x++) {
        b2(x) = rand();
    }

    for (int h3 = 0; h3 < H3; h3++) {
        for (int h2 = 0; h2 < H2; h2++) {
            w3(h2, h3) = rand();
        }
    }
    for (int x = 0; x < H3; x++) {
        b3(x) = rand();
    }

    // Timing code

    /*
    double base_time = benchmark(30, [&]() {
        mlp(input, w1, b1, w2, b2, w3, b3, output);
        output.device_sync();
    });
    printf("Base time: %gms\n", base_time * 1e3);
    */

    double auto_time = benchmark(30, [&]() {
        mlp_auto_schedule(input, w1, b1, w2, b2, w3, b3, output);
        output.device_sync();
    });
    printf("Auto time: %gms\n", auto_time * 1e3);

    printf("Success!\n");
    return 0;
}
