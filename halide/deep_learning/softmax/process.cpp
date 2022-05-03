#include <chrono>
#include <cstdio>
#include <omp.h>

#include "softmax.h"
#include "softmax_auto_schedule.h"

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

    std::vector<double> runtimes;
    for (int i = 0; i < 30; i++)
    {
        Buffer<float, 4> input_ = input.copy();
        Buffer<float, 4> output_ = output.copy();
        
        double t_start = omp_get_wtime();

        softmax_auto_schedule(input_, output_);
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
    Buffer<float, 4> output_ = output.copy();
    
    LIKWID_MARKER_INIT;
    LIKWID_MARKER_THREADINIT;

    LIKWID_MARKER_START("Compute");
    
    softmax_auto_schedule(input_, output_);
    output_.device_sync();

	LIKWID_MARKER_STOP("Compute");

    LIKWID_MARKER_CLOSE;

    printf("Success!\n");
    return 0;
}
