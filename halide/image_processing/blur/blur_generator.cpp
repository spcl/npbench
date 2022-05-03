#include "Halide.h"

namespace {

using namespace Halide;

class Blur : public Halide::Generator<Blur> {
public:
    Input<Buffer<uint8_t, 2>> input{"input"};
    Output<Buffer<uint8_t, 2>> output{"output"};

    void generate() {
        const int width = 1536, height = 2560;

        /** Algorithm **/
        Var x, y;

        Func input_f;
        input_f(x, y) = Halide::cast<float>(input(x, y));

        Func blur_x;        
        blur_x(x, y) = (input_f(x - 1, y) + input_f(x, y) + input_f(x + 1, y)) / 3;
        
        Func blur_y;
        blur_y(x, y) = (blur_x(x, y - 1) + blur_x(x, y) + blur_x(x, y + 1)) / 3;

        Func clip;
        clip(x, y) = cast<uint8_t>(min(max(blur_y(x, y), 0), 255));

        output = clip;

        /** Compile **/

        // Ask Halide to compile for this specific size:

        input.dim(0).set_bounds(0, width).set_stride(1);
        input.dim(1).set_bounds(0, height).set_stride(width);

        output.dim(0).set_bounds(1, width - 2).set_stride(1);
        output.dim(1).set_bounds(1, height - 2).set_stride(width - 2);

        /** Estimates **/

        input.dim(0).set_estimate(0, width);
        input.dim(1).set_estimate(0, height);

        output.dim(0).set_estimate(1, width - 2);
        output.dim(1).set_estimate(1, height - 2);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Blur, blur)
