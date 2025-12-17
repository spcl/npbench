import triton
import triton.language as tl
import torch


@triton.jit
def qr_step_kernel(
    A_ptr, Q_ptr, R_ptr,
    M, N,
    k,
    BLOCK_SIZE: tl.constexpr
):
    # Row indices for this block
    offs_m = tl.arange(0, BLOCK_SIZE)
    mask = offs_m < M

    # nrm = np.dot(A[:, k], A[:, k])
    a_k = tl.load(
        A_ptr + offs_m * N + k,
        mask=mask,
        other=0
    )
    nrm = tl.sum(a_k * a_k, axis=0)

    # R[k, k] = sqrt(nrm)
    rkk = tl.sqrt(nrm)
    tl.store(R_ptr + k * N + k, rkk)

    # Q[:, k] = A[:, k] / R[k, k]
    q_k = a_k / rkk
    tl.store(
        Q_ptr + offs_m * N + k,
        q_k,
        mask=mask
    )

    # For j in range(k+1, N):
    #    R[k, j] = dot(Q[:, k], A[:, j])
    #    A[:, j] -= Q[:, k] * R[k, j]
    for j in range(N):
        if j > k:

            # load A[:, j]
            a_j = tl.load(
                A_ptr + offs_m * N + j,
                mask=mask,
                other=0
            )

            # R[k, j] = dot(Q[:, k], A[:, j])
            rkj = tl.sum(q_k * a_j, axis=0)
            tl.store(R_ptr + k * N + j, rkj)

            # A[:, j] -= Q[:, k] * R[k, j]
            a_j = a_j - q_k * rkj
            tl.store(
                A_ptr + offs_m * N + j,
                a_j,
                mask=mask
            )


def kernel(A: torch.Tensor):
    M, N = A.shape

    Q = torch.empty_like(A)
    R = torch.zeros((N, N), dtype=A.dtype)
    # Cannot autotune, BLOCK_SIZE must be >= M
    BLOCK_SIZE = triton.next_power_of_2(M)

    grid = (1,)

    for k in range(N):
        qr_step_kernel[grid](
            A, Q, R,
            M, N, k,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return Q, R

