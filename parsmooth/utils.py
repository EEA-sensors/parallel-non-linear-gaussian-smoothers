from collections import namedtuple
from typing import NamedTuple

import jax.numpy as jnp
import jax.scipy.linalg as jlnialg
import numpy as np

__all__ = ["MVNParams", "make_matrices_parameters"]


class MVNParams(NamedTuple):
    mean: jnp.ndarray
    cov: jnp.ndarray or None = None
    chol: jnp.ndarray or None = None


def make_matrices_parameters(matrix: jnp.ndarray or np.array, n_observations: int) -> jnp.array:
    """ Processes a matrix (or "list" thereof) to be able to be iterated over n_observations times

    Parameters
    ----------
    matrix: array
        Matrix to be processed
    n_observations: int
        First dimension of the returned array

    Returns
    -------

    """
    if jnp.ndim(matrix) <= 2:
        return jnp.tile(matrix, (n_observations, 1, 1))
    elif jnp.ndim(matrix) == 3:
        if matrix.shape[0] == 1:
            return jnp.repeat(matrix, n_observations, 0)
        if matrix.shape[0] == n_observations:
            return matrix
        raise ValueError("if matrix has 3 dimensions, its first dimension must be of size 1 or n_observations")
    raise ValueError("matrix has more than 3 dimensions")


# The real logic
def _make_associative_filtering_params(args):
    Hk, Rk, Fk_1, Qk_1, uk_1, yk, dk, I_dim = args

    # FIRST TERM
    ############

    # temp variable
    HQ = jnp.dot(Hk, Qk_1)  # Hk @ Qk_1

    Sk = jnp.dot(HQ, Hk.T) + Rk
    Kk = jlnialg.solve(Sk, HQ, sym_pos=True).T  # using the fact that S and Q are symmetric

    # temp variable:
    I_KH = I_dim - jnp.dot(Kk, Hk)  # I - Kk @ Hk

    Ck = jnp.dot(I_KH, Qk_1)

    residual = (yk - jnp.dot(Hk, uk_1) - dk)

    bk = uk_1 + jnp.dot(Kk, residual)
    Ak = jnp.dot(I_KH, Fk_1)

    # SECOND TERM
    #############
    HF = jnp.dot(Hk, Fk_1)
    FHS_inv = jsolve(Sk, HF).T

    etak = jnp.dot(FHS_inv, residual)
    Jk = jnp.dot(FHS_inv, HF)

    return Ak, bk, Ck, etak, Jk
