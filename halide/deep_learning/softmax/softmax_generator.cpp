#include "Halide.h"

namespace {

using namespace Halide;

class Softmax : public Halide::Generator<Softmax> {
public:
    Input<Buffer<float, 4>> input{"input"};
    Output<Buffer<float, 4>> output{"output"};

    void generate() {
        const int S1 = 64, S2 = 16, S3 = 512, S4 = 512;

        /* THE ALGORITHM */

        Var s4, s3, s2, s1;

        RDom a(0, S4);
        Func maxi;
        maxi(s3, s2, s1) = maximum(input(a.x, s3, s2, s1));

        Func expo;
        expo(s4, s3, s2, s1) = exp(input(s4, s3, s2, s1) - maxi(s3, s2, s1));

        RDom b(0, S4);
        Func norm;
        norm(s3, s2, s1) += expo(b.x, s3, s2, s1);

        output(s4, s3, s2, s1) = expo(s4, s3, s2, s1) / norm(s3, s2, s1);

        // Bounds

        input.dim(0).set_bounds(0, S4).set_stride(1);
        input.dim(1).set_bounds(0, S3).set_stride(S4);
        input.dim(2).set_bounds(0, S2).set_stride(S4 * S3);
        input.dim(3).set_bounds(0, S1).set_stride(S4 * S3 * S2);

        output.dim(0).set_bounds(0, S4).set_stride(1);
        output.dim(1).set_bounds(0, S3).set_stride(S4);
        output.dim(2).set_bounds(0, S2).set_stride(S4 * S3);
        output.dim(3).set_bounds(0, S1).set_stride(S4 * S3 * S2);

        // Estimates

        input.dim(0).set_estimate(0, S4);
        input.dim(1).set_estimate(0, S3);
        input.dim(2).set_estimate(0, S2);
        input.dim(3).set_estimate(0, S1);

        output.dim(0).set_estimate(0, S4);
        output.dim(1).set_estimate(0, S3);
        output.dim(2).set_estimate(0, S2);
        output.dim(3).set_estimate(0, S1);

    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Softmax, softmax)
