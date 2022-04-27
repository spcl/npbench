// This file defines a generator for a first order IIR low pass filter
// for a 2D image.

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

Var x, y, c;

// Defines a func to blur the columns of an input with a first order low
// pass IIR filter, followed by a transpose.
Func blur_cols_transpose(Func input, Expr height, Expr alpha) {
    Func blur("blur");

    // Pure definition: do nothing.
    blur(x, y, c) = undef<float>();
    // Update 0: set the top row of the result to the input.
    blur(x, 0, c) = input(x, 0, c);
    // Update 1: run the IIR filter down the columns.
    RDom ry(1, height - 1);
    blur(x, ry, c) =
        (1 - alpha) * blur(x, ry - 1, c) + alpha * input(x, ry, c);
    // Update 2: run the IIR blur up the columns.
    Expr flip_ry = height - ry - 1;
    blur(x, flip_ry, c) =
        (1 - alpha) * blur(x, flip_ry + 1, c) + alpha * blur(x, flip_ry, c);

    // Transpose the blur.
    Func transpose("transpose");
    transpose(x, y, c) = blur(y, x, c);

    return transpose;
}

class IirBlur : public Generator<IirBlur> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};
    Output<Buffer<uint8_t, 3>> output{"output"};
    Input<float> alpha{"alpha"};

    void generate() {
        const int width = 1536;
        const int height = 2560;

        Expr w = input.width();
        Expr h = input.height();

        Func input_f;
        input_f(x, y, c) = Halide::cast<float>(input(x, y, c));

        // First, blur the columns of the input.
        Func blury_T = blur_cols_transpose(input_f, h, alpha);

        // Blur the columns again (the rows of the original).
        Func blur = blur_cols_transpose(blury_T, w, alpha);

        Func clip;
        clip(x, y, c) = cast<uint8_t>(min(max(blur(x, y, c), 0), 255));

        output = clip;

        // Bounds

        input.dim(0).set_bounds(0, width).set_stride(1);
        input.dim(1).set_bounds(0, height).set_stride(width);
        input.dim(2).set_bounds(0, 3).set_stride(width * height);
        
        output.dim(0).set_bounds(0, width).set_stride(1);
        output.dim(1).set_bounds(0, height).set_stride(width);
        output.dim(2).set_bounds(0, 3).set_stride(width * height);

        // Estimates

        input.dim(0).set_estimate(0, width);
        input.dim(1).set_estimate(0, height);
        input.dim(2).set_estimate(0, 3);

        alpha.set_estimate(0.1f);
        
        output.dim(0).set_estimate(0, width);
        output.dim(1).set_estimate(0, height);
        output.dim(2).set_estimate(0, 3);
    }
};

HALIDE_REGISTER_GENERATOR(IirBlur, iir_blur)
