#include <chrono>
#include <cstdio>
#include <omp.h>

#include "conv2d_bias.h"
#include "conv2d_bias_auto_schedule.h"

#include "HalideBuffer.h"

#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

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

    std::vector<double> runtimes;
    for (int i = 0; i < 30; i++)
    {
        Buffer<float, 4> input_ = input.copy();
        Buffer<float, 4> filter_ = filter.copy();
        Buffer<float, 1> bias_ = bias.copy();
        Buffer<float, 4> output_ = output.copy();

        double t_start = omp_get_wtime();

        conv2d_bias_auto_schedule(input_, filter_, bias_, output_);
        output_.device_sync();
    
        double t_end = omp_get_wtime();
        runtimes.push_back(t_end - t_start);
    }

    auto n = runtimes.size() / 2;
    nth_element(runtimes.begin(), runtimes.begin()+n, runtimes.end());
    
    auto med = runtimes[n];
    if(!(runtimes.size() & 1)) {
        auto max_it = max_element(runtimes.begin(), runtimes.begin()+n);
        med = (*max_it + med) / 2.0;
    }
    printf("Runtime: %f\n", med);

    Buffer<float, 4> input_ = input.copy();
    Buffer<float, 4> filter_ = filter.copy();
    Buffer<float, 1> bias_ = bias.copy();
    Buffer<float, 4> output_ = output.copy();
    
    LIKWID_MARKER_INIT;
    LIKWID_MARKER_THREADINIT;

    LIKWID_MARKER_START("Compute");
    
    conv2d_bias_auto_schedule(input_, filter_, bias_, output_);
    output_.device_sync();

	LIKWID_MARKER_STOP("Compute");

    LIKWID_MARKER_CLOSE;

    printf("Success!\n");
    return 0;
}
