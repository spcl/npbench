#include "Halide.h"

namespace {

using namespace Halide;

class Hdiff_SP : public Halide::Generator<Hdiff_SP> {
public:
    Input<Buffer<float, 3>> in_field{"in_field"};
    Input<Buffer<float, 3>> coeff{"coeff"};    
    Output<Buffer<float, 3>> output{"output"};

    void generate() {
        const int I = 256, J = 256, K = 160;

        /** Algorithm **/
        Var i, j, k;

        Func lap_field;
        lap_field(k, j, i) = 4.0f * in_field(k, j + 1, i + 1) - (in_field(k, j + 1, i + 2) + in_field(k, j + 1, i) + in_field(k, j + 2, i + 1) + in_field(k, j, i + 1));

        Func res_flx;
        res_flx(k, j, i) = lap_field(k, j + 1, i + 1) - lap_field(k, j + 1, i);

        Func condition_flx;
        condition_flx(k, j, i) = res_flx(k, j, i) * (in_field(k, j + 2, i + 2) - in_field(k, j + 2, i + 1));

        Func flx_field;
        flx_field(k, j, i) = Halide::select(condition_flx(k, j, i) > 0, 0, res_flx(k, j, i));

        Func res_fly;
        res_fly(k, j, i) = lap_field(k, j + 1, i + 1) - lap_field(k, j, i + 1);

        Func condition_fly;
        condition_fly(k, j, i) = res_flx(k, j, i) * (in_field(k, j + 2, i + 2) - in_field(k, j + 1, i + 2));

        Func fly_field;
        fly_field(k, j, i) = Halide::select(condition_fly(k, j, i) > 0, 0, res_fly(k, j, i));

        Func out_field;
        out_field(k, j, i) = in_field(k, j + 2, i + 2) - coeff(k, j, i) * ((flx_field(k, j, i + 1) - flx_field(k, j, i)) + (fly_field(k, j + 1, i) - fly_field(k, j , i)));

        output = out_field;

        /** Compile **/

        // Ask Halide to compile for this specific size:

        in_field.dim(0).set_bounds(0, K).set_stride(1);
        in_field.dim(1).set_bounds(0, J + 4).set_stride(K);
        in_field.dim(2).set_bounds(0, I + 4).set_stride(K * (J + 4));

        coeff.dim(0).set_bounds(0, K).set_stride(1);
        coeff.dim(1).set_bounds(0, J).set_stride(K);
        coeff.dim(2).set_bounds(0, I).set_stride(K * J);

        output.dim(0).set_bounds(0, K).set_stride(1);
        output.dim(1).set_bounds(0, J).set_stride(K);
        output.dim(2).set_bounds(0, I).set_stride(K * J);

        /** Estimates **/

        in_field.dim(0).set_estimate(0, K);
        in_field.dim(1).set_estimate(0, J + 4);
        in_field.dim(2).set_estimate(0, I + 4);

        coeff.dim(0).set_estimate(0, K);
        coeff.dim(1).set_estimate(0, J);
        coeff.dim(2).set_estimate(0, I);

        output.dim(0).set_estimate(0, K);
        output.dim(1).set_estimate(0, J);
        output.dim(2).set_estimate(0, I);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Hdiff_SP, hdiff_sp)
