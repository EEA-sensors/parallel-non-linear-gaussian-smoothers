import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import vmap


@vmap
def filtering_operator(elem1, elem2):
    """
    Associative operator described in TODO: put the reference

    Parameters
    ----------
    elem1: tuple of array
        a_i, b_i, C_i, eta_i, J_i
    elem2: tuple of array
        a_j, b_j, C_j, eta_j, J_j

    Returns
    -------

    """
    A1, b1, C1, eta1, J1 = elem1
    A2, b2, C2, eta2, J2 = elem2
    dim = b1.shape[0]

    I_dim = jnp.eye(dim)

    IpCJ = I_dim + jnp.dot(C1, J2)
    IpJC = I_dim + jnp.dot(J2, C1)

    AIpCJ_inv = jlinalg.solve(IpCJ.T, A2.T, sym_pos=False).T
    AIpJC_inv = jlinalg.solve(IpJC.T, A1, sym_pos=False).T

    A = jnp.dot(AIpCJ_inv, A1)
    b = jnp.dot(AIpCJ_inv, b1 + jnp.dot(C1, eta2)) + b2
    C = jnp.dot(AIpCJ_inv, jnp.dot(C1, A2.T)) + C2
    eta = jnp.dot(AIpJC_inv, eta2 - jnp.dot(J2, b1)) + eta1
    J = jnp.dot(AIpJC_inv, jnp.dot(J2, A1)) + J1
    return A, b, C, eta, J


@vmap
def smoothing_operator(elem1, elem2):
    """
    Associative operator described in TODO: put the reference

    Parameters
    ----------
    elem1: tuple of array
        g_i, E_i, L_i
    elem2: tuple of array
        g_j, E_j, L_j

    Returns
    -------

    """
    g1, E1, L1 = elem1
    g2, E2, L2 = elem2

    g = E2 @ g1 + g2
    E = E2 @ E1
    L = E2 @ L1 @ E2.T + L2
    return g, E, L
