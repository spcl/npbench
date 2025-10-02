import numpy as np
import numba as nb
BET_M = 0.5
BET_P = 0.5

@nb.jit(nopython=True, parallel=True, fastmath=True)
def vadv(utens_stage, u_stage, wcon, u_pos, utens, dtr_stage):
    I, J, K = (utens_stage.shape[0], utens_stage.shape[1], utens_stage.shape[2])
    ccol = np.empty((I, J, K), dtype=utens_stage.dtype)
    dcol = np.empty((I, J, K), dtype=utens_stage.dtype)
    data_col = np.empty((I, J), dtype=utens_stage.dtype)
    for k in range(1):
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * BET_M
        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - ccol[:, :, k]
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
        divided = 1.0 / bcol
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = dcol[:, :, k] * divided
    for k in range(1, K - 1):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        as_ = gav * BET_M
        cs = gcv * BET_M
        acol = gav * BET_P
        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - acol - ccol[:, :, k]
        correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = (dcol[:, :, k] - dcol[:, :, k - 1] * acol) * divided
    for k in range(K - 1, K):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        as_ = gav * BET_M
        acol = gav * BET_P
        bcol = dtr_stage - acol
        correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol[:, :, k] = (dcol[:, :, k] - dcol[:, :, k - 1] * acol) * divided
    for k in range(K - 1, K - 2, -1):
        datacol = dcol[:, :, k]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])
    for k in range(K - 2, -1, -1):
        datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])
    return (utens_stage, u_stage, wcon, u_pos, utens, dtr_stage)