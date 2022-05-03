#include "Halide.h"

namespace {

using namespace Halide;

class Conv2dBias : public Halide::Generator<Conv2dBias> {
public:
    Input<Buffer<float, 4>> input{"input"};
    Input<Buffer<float, 4>> filter{"filter"};
    Input<Buffer<float, 1>> bias{"bias"};
    Output<Buffer<float, 4>> output{"output"};

    void generate() {
        const int N = 8, CI = 3, CO = 16, W = 256, H = 256, K = 20;
        const int border = K - 1;
        /* THE ALGORITHM */

        Var x("x"), y("y"), c("c"), n("n");

        RDom r(0, CI, 0, K, 0, K);

        output(c, x, y, n) = bias(c);
        output(c, x, y, n) += filter(r.x, r.y, r.z, c) * input(r.x, x + r.y, y + r.z, n);

        /* THE SCHEDULE */

        // Ask Halide to compile for this specific size:

        input.dim(0).set_bounds(0, CI).set_stride(1);
        input.dim(1).set_bounds(0, W).set_stride(CI);
        input.dim(2).set_bounds(0, H).set_stride(CI * W);
        input.dim(3).set_bounds(0, N).set_stride(CI * W * H);

        filter.dim(0).set_bounds(0, CI).set_stride(1);
        filter.dim(1).set_bounds(0, K).set_stride(CI);
        filter.dim(2).set_bounds(0, K).set_stride(CI * K);
        filter.dim(3).set_bounds(0, CO).set_stride(CI * K * K);

        bias.dim(0).set_bounds(0, CO).set_stride(1);

        output.dim(0).set_bounds(0, CO).set_stride(1);
        output.dim(1).set_bounds(0, W - border).set_stride(CO);
        output.dim(2).set_bounds(0, H - border).set_stride(CO * (W - border));
        output.dim(3).set_bounds(0, N).set_stride(CO * (W - border) * (H - border));

        // estimates

        input.dim(0).set_estimate(0, CI);
        input.dim(1).set_estimate(0, W); 
        input.dim(2).set_estimate(0, H);
        input.dim(3).set_estimate(0, N);

        output.dim(0).set_estimate(0, CO);
        output.dim(1).set_estimate(0, W - border);
        output.dim(2).set_estimate(0, H - border);
        output.dim(3).set_estimate(0, N);

        filter.dim(0).set_estimate(0, CI);
        filter.dim(1).set_estimate(0, K);
        filter.dim(2).set_estimate(0, K);
        filter.dim(3).set_estimate(0, CO);

        bias.dim(0).set_estimate(0, CO);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Conv2dBias, conv2d_bias)
