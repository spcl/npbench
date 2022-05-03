#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

#include "HalideBuffer.h"
#include "halide_image_io.h"

#include "iir_blur.h"
#include "iir_blur_auto_schedule.h"

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
    const std::string input_image_path = "../image_processing/images/rgb.png";
    const std::string output_image_path = "./iir_blur_output.png";

    Buffer<uint8_t, 3> input = Halide::Tools::load_image(input_image_path);
    Buffer<uint8_t, 3> output(input.width(), input.height(), input.channels());

    // Timing code
    const float alpha = 0.1f;

    std::vector<double> runtimes;
    for (int i = 0; i < 30; i++)
    {
        Buffer<uint8_t, 3> input_ = input.copy();
        Buffer<uint8_t, 3> output_ = output.copy();    

        double t_start = omp_get_wtime();

        iir_blur_auto_schedule(input_, alpha, output_);
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

    Buffer<uint8_t, 3> input_ = input.copy();
    Buffer<uint8_t, 3> output_ = output.copy();

    LIKWID_MARKER_INIT;
    LIKWID_MARKER_THREADINIT;

    LIKWID_MARKER_START("Compute");
    
    iir_blur_auto_schedule(input_, alpha, output_);
    output_.device_sync();

	LIKWID_MARKER_STOP("Compute");

    LIKWID_MARKER_CLOSE;

    Halide::Tools::save_image(output_, output_image_path);

    printf("Success!\n");
    return 0;
}
