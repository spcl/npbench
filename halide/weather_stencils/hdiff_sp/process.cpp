#include <chrono>
#include <cstdio>

#include "hdiff_sp.h"
#include "hdiff_sp_auto_schedule.h"

#include "HalideBuffer.h"
#include "halide_image_io.h"

#include "../../benchmark.h"

using namespace Halide::Runtime;

int main(int argc, char **argv) {
    const int I = 256, J = 256, K = 160;
    
    Buffer<float, 3> in_field(K, J + 4, I + 4);
    Buffer<float, 3> coeff(K, J, I);    
    Buffer<float, 3> output(K, J, I);

    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J + 4; j++) {
            for (int k = 0; k < K + 4; k++) {
                in_field(k, j, i) = rand();
            } 
        }
    }

    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            for (int k = 0; k < K; k++) {
                coeff(k, j, i) = rand();
            } 
        }
    }

    // Timing code

    /*
    double base_time = benchmark(30, [&]() {
        hdiff_sp(in_field, coeff, output);
        output.device_sync();
    });
    printf("Base: %gms\n", base_time * 1e3);
    */

    // Auto-scheduled version
    double auto_time = benchmark(30, [&]() {
        hdiff_sp_auto_schedule(in_field, coeff, output);
        output.device_sync();
    });
    printf("Auto time: %gms\n", auto_time * 1e3);

    printf("Success!\n");
    return 0;
}
