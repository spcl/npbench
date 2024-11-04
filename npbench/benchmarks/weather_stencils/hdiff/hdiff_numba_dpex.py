import dpnp as np
import numba_dpex as dpex

@dpex.kernel
def hdiff_kernel(in_field, out_field, coeff, I, J, K):
    i = dpex.get_global_id(0)
    j = dpex.get_global_id(1)
    k = dpex.get_global_id(2)

    if i < I and j < J and k < K:
        # Compute laplacian field
        lap_field = 4.0 * in_field[i + 1, j + 1, k] - (
            in_field[i + 2, j + 1, k] + in_field[i, j + 1, k] +
            in_field[i + 1, j + 2, k] + in_field[i + 1, j, k])

        # Compute flux in x-direction (flx_field)
        res_x = lap_field - (4.0 * in_field[i, j + 1, k] - (
            in_field[i + 1, j + 1, k] + in_field[i - 1, j + 1, k] +
            in_field[i, j + 2, k] + in_field[i, j, k]))
        flx_field = res_x if (res_x * (in_field[i + 1, j + 1, k] - in_field[i, j + 1, k])) <= 0 else 0

        # Compute flux in y-direction (fly_field)
        res_y = lap_field - (4.0 * in_field[i + 1, j, k] - (
            in_field[i + 2, j, k] + in_field[i, j, k] +
            in_field[i + 1, j + 1, k] + in_field[i + 1, j - 1, k]))
        fly_field = res_y if (res_y * (in_field[i + 1, j, k] - in_field[i + 1, j - 1, k])) <= 0 else 0

        # Update the output field
        out_field[i, j, k] = in_field[i + 1, j + 1, k] - coeff[i, j, k] * (
            flx_field + fly_field)

def hdiff(in_field, out_field, coeff):
    '''
    Horizontal diffusion kernel using numba-dpex for device execution.
    '''
    I, J, K = out_field.shape[0], out_field.shape[1], out_field.shape[2]
    
    # Convert arrays to dpnp arrays for device execution
    in_field = np.asarray(in_field)
    out_field = np.asarray(out_field)
    coeff = np.asarray(coeff)

    # Define global size for parallelization over I, J, and K
    global_size = (I, J, K)

    # Launch the kernel
    hdiff_kernel[global_size](in_field, out_field, coeff, I, J, K)


