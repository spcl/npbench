#include "Halide.h"

namespace {

using namespace Halide;

class MLP : public Halide::Generator<MLP> {
public:
    Input<Buffer<float, 2>> input{"input"};
    Input<Buffer<float, 2>> w1{"w1"};
    Input<Buffer<float, 1>> b1{"b1"};
    Input<Buffer<float, 2>> w2{"w2"};
    Input<Buffer<float, 1>> b2{"b2"};
    Input<Buffer<float, 2>> w3{"w3"};
    Input<Buffer<float, 1>> b3{"b3"};
    Output<Buffer<float, 2>> output{"output"};

    void generate() {
        const int N = 8, C = 3, H1 = 30000, H2 = 10000, H3 = 1000;
        
        /* THE ALGORITHM */
        
        Var n;

        // Layer 1

        Var h1;
        RDom r1(0, C);
        Func layer1;
        layer1(h1, n) = b1(h1);
        layer1(h1, n) += input(r1.x, n) * w1(r1.x, h1);

        Func relu1;
        relu1(h1, n) = max(0.0f, layer1(h1, n));

        // Layer 2

        Var h2;
        RDom r2(0, H1);
        Func layer2;
        layer2(h2, n) = b2(h2);
        layer2(h2, n) += layer1(r2.x, n) * w2(r2.x, h2);

        Func relu2;
        relu2(h2, n) = max(0.0f, layer2(h2, n));

        // Layer 3

        Var h3;
        RDom r3(0, H2);
        Func layer3;
        layer3(h3, n) = b3(h3);
        layer3(h3, n) += layer2(r3.x, n) * w3(r3.x, h3);

        // Softmax

        RDom a(0, H3);
        Func maxi;
        maxi(n) = maximum(layer3(a.x, n));

        Func expo;
        expo(h3, n) = exp(layer3(h3, n) - maxi(n));

        RDom b(0, H3);
        Func norm;
        norm(n) += expo(b.x, n);

        output(h3, n) = expo(h3, n) / norm(n);


        input.dim(0).set_bounds(0, C).set_stride(1);
        input.dim(1).set_bounds(0, N).set_stride(C);

        w1.dim(0).set_bounds(0, C).set_stride(1);
        w1.dim(1).set_bounds(0, H1).set_stride(C);
        b1.dim(0).set_bounds(0, H1).set_stride(1);

        w2.dim(0).set_bounds(0, H1).set_stride(1);
        w2.dim(1).set_bounds(0, H2).set_stride(H1);
        b2.dim(0).set_bounds(0, H2).set_stride(1);

        w3.dim(0).set_bounds(0, H2).set_stride(1);
        w3.dim(1).set_bounds(0, H3).set_stride(H2);
        b3.dim(0).set_bounds(0, H3).set_stride(1);

        output.dim(0).set_bounds(0, H3).set_stride(1);
        output.dim(1).set_bounds(0, N).set_stride(H3);

        // Estimates

        input.dim(0).set_estimate(0, C);
        input.dim(1).set_estimate(0, N);

        w1.dim(0).set_estimate(0, C);
        w1.dim(1).set_estimate(0, H1);
        b1.dim(0).set_estimate(0, H1);

        w2.dim(0).set_estimate(0, H1);
        w2.dim(1).set_estimate(0, H2);
        b2.dim(0).set_estimate(0, H2);

        w3.dim(0).set_estimate(0, H2);
        w3.dim(1).set_estimate(0, H3);
        b3.dim(0).set_estimate(0, H3);

        output.dim(0).set_estimate(0, H3);
        output.dim(1).set_estimate(0, N);

    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(MLP, mlp)
