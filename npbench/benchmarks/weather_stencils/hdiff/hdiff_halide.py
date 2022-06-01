import numpy as np
import halide as hl

def hdiff(in_field, coeff):
    I = 256
    J = 256
    K = 160

    i, j, k = hl.Var("i"), hl.Var("j"), hl.Var("k")

    lap_field = hl.Func("lap_field")
    lap_field[k, j, i] = 4.0 * in_field[k, j + 1, i + 1] - (in_field[k, j + 1, i + 2] + in_field[k, j + 1, i] + in_field[k, j + 2, i + 1] + in_field[k, j, i + 1])

    res_flx = hl.Func("res_flx")
    res_flx[k, j, i] = lap_field[k, j + 1, i + 1] - lap_field[k, j + 1, i]

    condition_flx = hl.Func("condition_flx")
    condition_flx[k, j, i] = res_flx[k, j, i] * (in_field[k, j + 2, i + 2] - in_field[k, j + 2, i + 1])

    flx_field = hl.Func("flx_field")
    flx_field[k, j, i] = hl.select(condition_flx[k, j, i] > 0, 0, res_flx[k, j, i])

    res_fly = hl.Func("res_fly")
    res_fly[k, j, i] = lap_field[k, j + 1, i + 1] - lap_field[k, j, i + 1]

    condition_fly = hl.Func("condition_fly")
    condition_fly[k, j, i] = res_flx[k, j, i] * (in_field[k, j + 2, i + 2] - in_field[k, j + 1, i + 2])

    fly_field = hl.Func("fly_field")
    fly_field[k, j, i] = hl.select(condition_fly[k, j, i] > 0, 0, res_fly[k, j, i])

    out_field = hl.Func("out_field")
    out_field[k, j, i] = in_field[k, j + 2, i + 2] - coeff[k, j, i] * ((flx_field[k, j, i + 1] - flx_field[k, j, i]) + (fly_field[k, j + 1, i] - fly_field[k, j , i]))

    # Set bounds
    
    in_field.dim(0).set_bounds(0, K).set_stride(1)
    in_field.dim(1).set_bounds(0, J + 4).set_stride(K)
    in_field.dim(2).set_bounds(0, I + 4).set_stride(K * (J + 4))

    coeff.dim(0).set_bounds(0, K).set_stride(1)
    coeff.dim(1).set_bounds(0, J).set_stride(K)
    coeff.dim(2).set_bounds(0, I).set_stride(K * J)

    in_field.dim(0).set_estimate(0, K)
    in_field.dim(1).set_estimate(0, J + 4)
    in_field.dim(2).set_estimate(0, I + 4)

    coeff.dim(0).set_estimate(0, K)
    coeff.dim(1).set_estimate(0, J)
    coeff.dim(2).set_estimate(0, I)

    out_field.set_estimate(k, 0, K)
    out_field.set_estimate(j, 0, J)
    out_field.set_estimate(i, 0, I)

    return out_field
