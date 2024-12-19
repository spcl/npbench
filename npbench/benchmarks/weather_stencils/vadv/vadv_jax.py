import jax
import jax.numpy as jnp
from jax import lax

# Sample constants
BET_M = 0.5
BET_P = 0.5

# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L111
@jax.jit
def vadv(utens_stage, u_stage, wcon, u_pos, utens, dtr_stage):
    I, J, K = utens_stage.shape[0], utens_stage.shape[1], utens_stage.shape[2]
    ccol = jnp.empty((I, J, K), dtype=utens_stage.dtype)
    dcol = jnp.empty((I, J, K), dtype=utens_stage.dtype)
    data_col = jnp.empty((I, J), dtype=utens_stage.dtype)

    def loop1(k, loop_vars):
        ccol, dcol = loop_vars
        gcv = 0.25 * (wcon[1:, :, 0 + 1] + wcon[:-1, :, 0 + 1])
        cs = gcv * BET_M
        ccol = ccol.at[:, :, k].set(gcv * BET_P)
        bcol = dtr_stage - ccol[:, :, k]

        # update the d column
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol = dcol.at[:, :, k].set((dtr_stage * u_pos[:, :, k] + utens[:, :, k] +
                         utens_stage[:, :, k] + correction_term))

        # Thomas forward
        divided = 1.0 / bcol
        ccol = ccol.at[:, :, k].set(ccol[:, :, k] * divided)
        dcol = dcol.at[:, :, k].set(dcol[:, :, k] * divided)

        return ccol, dcol
    
    ccol, dcol = lax.fori_loop(0, 1, loop1, (ccol, dcol))

    def loop2(k, loop_vars):
        ccol, dcol = loop_vars
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])

        as_ = gav * BET_M
        cs = gcv * BET_M

        acol = gav * BET_P
        ccol = ccol.at[:, :, k].set(gcv * BET_P)
        bcol = dtr_stage - acol - ccol[:, :, k]

        # update the d column
        correction_term = -as_ * (u_stage[:, :, k - 1] -
                                  u_stage[:, :, k]) - cs * (
                                      u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol = dcol.at[:, :, k].set((dtr_stage * u_pos[:, :, k] + utens[:, :, k] +
                                     utens_stage[:, :, k] + correction_term))

        # Thomas forward
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol = ccol.at[:, :, k].set(ccol[:, :, k] * divided)
        dcol = dcol.at[:, :, k].set((dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided)

        return ccol, dcol
    
    ccol, dcol = lax.fori_loop(1, K - 1, loop2, (ccol, dcol))

    def loop3(k, dcol):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        as_ = gav * BET_M
        acol = gav * BET_P
        bcol = dtr_stage - acol

        # update the d column
        correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol = dcol.at[:, :, k].set((dtr_stage * u_pos[:, :, k] + utens[:, :, k] +
                         utens_stage[:, :, k] + correction_term))
        
        # Thomas forward
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol = dcol.at[:, :, k].set((dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided)

        return dcol
    
    dcol = lax.fori_loop(K - 1, K, loop3, dcol)

    def loop4(k, loop_vars):
        data_col, utens_stage = loop_vars
        datacol = dcol[:, :, k]
        data_col = data_col.at[:].set(datacol)
        utens_stage = utens_stage.at[:, :, k].set(dtr_stage * (datacol - u_pos[:, :, k]))

        return data_col, utens_stage
    
    data_col, utens_stage = lax.fori_loop(K - 1, K, loop4, (data_col, utens_stage))

    def loop5(k, loop_vars):
        data_col, utens_stage = loop_vars
        K = utens_stage.shape[2]
        k = K - 2 - k 
        datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col = data_col.at[:].set(datacol)
        utens_stage = utens_stage.at[:, :, k].set(dtr_stage * (datacol - u_pos[:, :, k]))

        return data_col, utens_stage
    
    data_col, utens_stage = lax.fori_loop(0, K - 1, loop5, (data_col, utens_stage))

    return ccol, dcol, data_col, utens_stage
