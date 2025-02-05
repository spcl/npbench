import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=["N", "R", "K"])
def stockham_fft(N, R, K, x, y):

    # Generate DFT matrix for radix R.
    # Define transient variable for matrix.
    i_coord, j_coord = jnp.mgrid[0:R, 0:R]
    # dft_mat = jnp.empty((R, R), dtype=jnp.complex128)
    dft_mat = jnp.exp(-2.0j * jnp.pi * i_coord * j_coord / R)
    y = x

    ii_coord, jj_coord = jnp.mgrid[0:R, 0:R**K]

    # Main Stockham loop
    for i in range(K):
        # Stride permutation
        yv = jnp.reshape(y, (R**i, R, R**(K - i - 1)))
        tmp_perm = jnp.transpose(yv, axes=(1, 0, 2))

        # Twiddle Factor multiplication
        tmp = jnp.exp(-2.0j * jnp.pi * ii_coord[:, :R**i] * jj_coord[:, :R**i] / R**(i + 1))
        D = jnp.repeat(jnp.reshape(tmp, (R, R**i, 1)), R**(K - i - 1), axis=2)
        tmp_twid = jnp.reshape(tmp_perm, (N, )) * jnp.reshape(D, (N, ))

        # Product with Butterfly
        y = jnp.reshape(dft_mat @ jnp.reshape(tmp_twid, (R, R**(K - 1))),(N, ))

    return y