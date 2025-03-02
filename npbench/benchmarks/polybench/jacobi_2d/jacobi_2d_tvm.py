
import inspect
import tvm
from tvm import te
from tvm import relay
from tvm import auto_scheduler
import tvm.testing
from tvm import autotvm
from npbench.infrastructure.tvm_framework import TVMFramework

@auto_scheduler.register_workload("jacobi_2d_combined_gpu")
def jacobi_2d_combined_gpu(A, B, TSTEPS, N, dtype):
    A = te.placeholder((N, N), name="A", dtype=dtype)
    B = te.placeholder((N, N), name="B", dtype=dtype)
    T = te.create_iteration_var(te.Range(1, TSTEPS), "t", "serial")

    def compute_step1(A):
        return te.compute(
            (N, N),
            lambda i, j:
            te.if_then_else(
                te.all(i > 0, i < N-1, j > 0, j < N-1),
                0.2 * (
                    A[i, j] +   # center
                    A[i, j - 1] + # left
                    A[i, j + 1] + # right
                    A[i - 1, j] + # top
                    A[i + 1, j]   # bottom
                ),
                A[i, j]
            ),
            name="B_comp"
        )


    def compute_step2(B):
        return te.compute(
            (N, N),
            lambda i, j:
            te.if_then_else(
                te.all(i > 0, i < N-1, j > 0, j < N-1),
                0.2 * (
                    B[i, j] +   # center
                    B[i, j - 1] + # left
                    B[i, j + 1] + # right
                    B[i - 1, j] + # top
                    B[i + 1, j]   # bottom
                ),
                B[i, j]
            ),
            name="A_comp"
        )

    # Initialize first iteration manually
    B_1 = compute_step1(A)
    A_1 = compute_step2(B_1)

    # Create scan state - we need to store only the latest A values
    # Shape: [1, N, N] - the first dimension is for the state
    state_placeholder = te.placeholder((1, N, N), dtype=dtype, name="state_placeholder")

    # Initialize the scan state with A_1 (result of first iteration)
    init_state = te.compute((1, N, N), lambda t, i, j: A_1[i, j], name="init_state")

    # Define scan body function
    # This will be applied for each time step
    def scan_body(t, states):
        # Get the latest state (A values from previous iteration)
        A_prev = states[0]

        # Compute next B values from previous A values
        B_next = compute_step1(A_prev)

        # Compute next A values from the B values
        A_next = compute_step2(B_next)

        # Return the new state (updated A values)
        return [A_next]

    # Apply scan operator to run the remaining TSTEPS-1 iterations
    # Note: We already did the first iteration manually, so we need TSTEPS-1 more
    # The state is initialized with the results of the first iteration
    if TSTEPS > 1:
        # scan_outputs shape: [TSTEPS-1, N, N]
        scan_outputs = tvm.te.scan(
            init_state,        # Initial state
            scan_body,         # Function to apply at each step
            TSTEPS-1,          # Number of iterations
            name="time_scan"
        )

        # Get the final A values from the last iteration
        final_A = scan_outputs[0][TSTEPS-2]  # Get the last element
    else:
        # If TSTEPS = 1, just return the result of the first iteration
        final_A = A_1

    return [final_A, B]



_kernel1 = None

def autotuner(TSTEPS, A, B):
    global _kernel1

    if _kernel1 is not None:
        return

    dtype = A.dtype
    M = int(A.shape[0])
    N = int(A.shape[1])
    assert M == N

    _kernel1 = TVMFramework.autotune(func=jacobi_2d_combined_gpu, name="jacobi_2d_combined_gpu", args=(A, B, TSTEPS, N, dtype), target=tvm.target.cuda())


def kernel(TSTEPS, A, B):
    global _kernel1

    _kernel1(A, B, TSTEPS)

    return A