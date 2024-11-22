import jax
import jax.numpy as jnp

@jax.jit
def scattering_self_energies(neigh_idx, dH, G, D, Sigma):
    def body_fun(sigma, idx):
        k, E, q, w, a, b, i, j = idx

        dHG = G[k, E - w, neigh_idx[a, b]] @ dH[a, b, i]
        dHD = dH[a, b, j] * D[q, w, a, b, i, j]

        update = jnp.where(E >= w, dHG @ dHD, 0.0)

        return sigma.at[k, E, a].add(update), None

    k_range = jnp.arange(G.shape[0])
    E_range = jnp.arange(G.shape[1])
    q_range = jnp.arange(D.shape[0])
    w_range = jnp.arange(D.shape[1])
    a_range = jnp.arange(neigh_idx.shape[0])
    b_range = jnp.arange(neigh_idx.shape[1])
    i_range = jnp.arange(D.shape[-2])
    j_range = jnp.arange(D.shape[-1])

    indices = jnp.meshgrid( # Create meshgrid of indices
        k_range, E_range, q_range, w_range,
        a_range, b_range, i_range, j_range,
        indexing='ij'
    )

    indices = jnp.stack([idx.ravel() for idx in indices], axis=1) # Reshape indices into a single array of 8-tuples

    result, _ = jax.lax.scan(body_fun, Sigma, indices) # Use scan to iterate over all index combinations

    return result
