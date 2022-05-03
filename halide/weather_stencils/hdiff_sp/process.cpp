#include <chrono>
#include <cstdio>
#include <omp.h>

#include "hdiff_sp.h"
#include "hdiff_sp_auto_schedule.h"

#include "HalideBuffer.h"
#include "halide_image_io.h"

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

    std::vector<double> runtimes;
    for (int i = 0; i < 30; i++)
    {
        Buffer<float, 3> in_field_ = in_field.copy();
        Buffer<float, 3> coeff_ = coeff.copy();
        Buffer<float, 3> output_ = output.copy();

        double t_start = omp_get_wtime();

        hdiff_sp_auto_schedule(in_field_, coeff_, output_);
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


    Buffer<float, 3> in_field_ = in_field.copy();
    Buffer<float, 3> coeff_ = coeff.copy();    
    Buffer<float, 3> output_ = output.copy();

    LIKWID_MARKER_INIT;
    LIKWID_MARKER_THREADINIT;

    LIKWID_MARKER_START("Compute");
    
    hdiff_sp_auto_schedule(in_field_, coeff_, output_);
    output_.device_sync();
    
	LIKWID_MARKER_STOP("Compute");

    LIKWID_MARKER_CLOSE;

    printf("Success!\n");
    return 0;
}
