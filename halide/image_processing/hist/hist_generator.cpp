#include "Halide.h"

namespace {

using namespace Halide::ConciseCasts;

class Hist : public Halide::Generator<Hist> {
// Benchmark: Histogram equalization

public:
    Input<Buffer<uint8_t, 3>> input{"input"};
    Output<Buffer<uint8_t, 3>> output{"output"};

    void generate() {
        const int width = 1536;
        const int height = 2560;

        // Algorithm
        Var x, y, c;

        Func input_f;
        input_f(x, y, c) = Halide::cast<float>(input(x, y, c));

        Func gray("gray");
        gray(x, y) = (0.299f * input_f(x, y, 0) +
                   0.587f * input_f(x, y, 1) +
                   0.114f * input_f(x, y, 2));

        Func Cr("Cr");
        Expr R = input_f(x, y, 0);
        Cr(x, y) = (R - gray(x, y)) * 0.713f + 128;

        Func Cb("Cb");
        Expr B = input_f(x, y, 2);
        Cb(x, y) = (B - gray(x, y)) * 0.564f + 128;

        Func hist_rows("hist_rows");
        hist_rows(x, y) = 0;
        RDom rx(0, input.width());
        Expr bin = cast<int>(clamp(gray(rx, y), 0, 255));
        hist_rows(bin, y) += 1;

        Func hist("hist");
        hist(x) = 0;
        RDom ry(0, input.height());
        hist(x) += hist_rows(x, ry);

        Func cdf("cdf");
        cdf(x) = hist(0);
        RDom b(1, 255);
        cdf(b.x) = cdf(b.x - 1) + hist(b.x);

        Func cdf_bin("cdf_bin");
        cdf_bin(x, y) = u8(clamp(gray(x, y), 0, 255));

        Func eq("equalize");
        eq(x, y) = clamp(cdf(cdf_bin(x, y)) * (255.0f / (input.height() * input.width())), 0, 255);

        Expr red = u8(clamp(eq(x, y) + (Cr(x, y) - 128) * 1.4f, 0, 255));
        Expr green = u8(clamp(eq(x, y) - 0.343f * (Cb(x, y) - 128) - 0.711f * (Cr(x, y) - 128), 0, 255));
        Expr blue = u8(clamp(eq(x, y) + 1.765f * (Cb(x, y) - 128), 0, 255));
        
        output(x, y, c) = mux(c, {red, green, blue});

        // Bounds

        input.dim(0).set_estimate(0, width).set_stride(1);
        input.dim(1).set_estimate(0, height).set_stride(width);
        input.dim(2).set_estimate(0, 3).set_stride(width * height);
        
        output.dim(0).set_estimate(0, width).set_stride(1);
        output.dim(1).set_estimate(0, height).set_stride(width);
        output.dim(2).set_estimate(0, 3).set_stride(width * height);

        // Estimates

        input.dim(0).set_estimate(0, width);
        input.dim(1).set_estimate(0, height);
        input.dim(2).set_estimate(0, 3);

        output.dim(0).set_estimate(0, width);
        output.dim(1).set_estimate(0, height);
        output.dim(2).set_estimate(0, 3);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Hist, hist)
