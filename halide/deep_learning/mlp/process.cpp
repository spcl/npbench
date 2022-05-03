#include <chrono>
#include <cstdio>
#include <omp.h>

#include "mlp.h"
#include "mlp_auto_schedule.h"

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

    std::vector<double> runtimes;
    for (int i = 0; i < 30; i++)
    {
        Buffer<float, 2> input_ = input.copy();
        Buffer<float, 2> w1_ = w1.copy();
        Buffer<float, 1> b1_ = b1.copy();
        Buffer<float, 2> w2_ = w2.copy();
        Buffer<float, 1> b2_ = b2.copy();
        Buffer<float, 2> w3_ = w3.copy();
        Buffer<float, 1> b3_ = b3.copy();
        Buffer<float, 2> output_ = output.copy();
        
        double t_start = omp_get_wtime();

        mlp_auto_schedule(input_, w1_, b1_, w2_, b2_, w3_, b3_, output_);
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

    Buffer<float, 2> input_ = input.copy();
    Buffer<float, 2> w1_ = w1.copy();
    Buffer<float, 1> b1_ = b1.copy();
    Buffer<float, 2> w2_ = w2.copy();
    Buffer<float, 1> b2_ = b2.copy();
    Buffer<float, 2> w3_ = w3.copy();
    Buffer<float, 1> b3_ = b3.copy();
    Buffer<float, 2> output_ = output.copy();
    
    LIKWID_MARKER_INIT;
    LIKWID_MARKER_THREADINIT;

    LIKWID_MARKER_START("Compute");
    
    mlp_auto_schedule(input_, w1_, b1_, w2_, b2_, w3_, b3_, output_);
    output_.device_sync();

	LIKWID_MARKER_STOP("Compute");

    LIKWID_MARKER_CLOSE;

    printf("Success!\n");
    return 0;
}
