#include <chrono>
#include <cstdio>

#include "softmax.h"
#include "softmax_auto_schedule.h"

#include "HalideBuffer.h"

#include "../../benchmark.h"

using namespace Halide::Runtime;

int main(int argc, char **argv) {
    const int S1 = 64, S2 = 16, S3 = 512, S4 = 512;

    Buffer<float, 4> input(S4, S3, S2, S1);
    Buffer<float, 4> output(S4, S3, S2, S1);

    for (int s1 = 0; s1 < S1; s1++) {
        for (int s2 = 0; s2 < S2; s2++) {
            for (int s3 = 0; s3 < S3; s3++) {
                for (int s4 = 0; s4 < S4; s4++) {
                    input(s4, s3, s2, s1) = rand();
                }
            } 
        }
    }


    // Timing code

    /*
    double base_time = benchmark(30, [&]() {
        softmax(input, output);
        output.device_sync();
    });
    printf("Base time: %gms\n", base_time * 1e3);
    */

    double auto_time = benchmark(30, [&]() {
        softmax_auto_schedule(input, output);
        output.device_sync();
    });
    printf("Auto time: %gms\n", auto_time * 1e3);

    printf("Success!\n");
    return 0;
}
