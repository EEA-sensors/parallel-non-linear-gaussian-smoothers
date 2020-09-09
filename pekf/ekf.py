from typing import Callable

import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax, jacfwd, vmap

from pekf.utils import MVNormalParameters


@vmap
def filtering_operator(elem1, elem2):
    """
    Wrapper for the associative operator described in TODO: put the reference
    It is implemented in :func:`_filtering_operator`

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

    AIpCJ_inv = jlinalg.solve(IpCJ, A2.T, sym_pos=True).T
    AIpJC_inv = jlinalg.solve(IpJC, A1, sym_pos=True).T

    A = jnp.dot(AIpCJ_inv, A1)
    b = jnp.dot(AIpCJ_inv, b1 + jnp.dot(C1, eta2)) + b2
    C = jnp.dot(AIpCJ_inv, jnp.dot(C1, A2.T)) + C2
    eta = jnp.dot(AIpJC_inv, eta2 - jnp.dot(J2, b1)) + eta1
    J = jnp.dot(AIpJC_inv, jnp.dot(J2, A1)) + J1
    return A, b, C, eta, J


def make_associative_filtering_params(observation_function, Rk, transition_function, Qk_1, yk, i, m0, P0):
    predicate = i == 0

    generic_operand = (Rk, Qk_1, yk, m0)
    initial_operand = (Rk, P0, yk, jnp.zeros_like(m0))

    def _generic_function(args):
        r, q, y, m = args[0]
        inp = (observation_function, r, transition_function, q, y, m)
        return _make_associative_filtering_params(inp)

    def _initial_function(args):
        r, q, y, m = args[1]
        inp = (observation_function, r, lambda z: z, q, y, m)
        return _make_associative_filtering_params(inp)

    return lax.cond(predicate,
                    _initial_function,  # take initial
                    _generic_function,  # take generic
                    (initial_operand, generic_operand))


def _make_associative_filtering_params(args):
    observation_function, Rk, transition_function, Qk_1, yk, u = args

    # FIRST TERM
    ############

    dim_x = Qk_1.shape[0]

    zero_x = jnp.zeros(dim_x)
    I_dim = jnp.eye(dim_x)

    # Jacobians
    Hk = jacfwd(observation_function, 0)(zero_x)
    Fk_1 = jacfwd(transition_function, 0)(zero_x)

    # temp variable
    HQ = jnp.dot(Hk, Qk_1)  # Hk @ Qk_1

    Sk = jnp.dot(HQ, Hk.T) + Rk
    Kk = jlinalg.solve(Sk, HQ, sym_pos=True).T  # using the fact that S and Q are symmetric

    # temp variable:
    I_KH = I_dim - jnp.dot(Kk, Hk)  # I - Kk @ Hk

    Ck = jnp.dot(I_KH, Qk_1)

    residual = yk - jnp.dot(Hk, u)

    bk = u + jnp.dot(Kk, residual)
    Ak = jnp.dot(I_KH, Fk_1)

    # SECOND TERM
    #############
    HF = jnp.dot(Hk, Fk_1)
    FHS_inv = jlinalg.solve(Sk, HF).T

    etak = jnp.dot(FHS_inv, residual)
    Jk = jnp.dot(FHS_inv, HF)

    return Ak, bk, Ck, etak, Jk


def filter_routine(initial_state: MVNormalParameters,
                   observations: jnp.ndarray,
                   transition_function: Callable,
                   transition_covariance: jnp.ndarray,
                   observation_function: Callable,
                   observation_covariance: jnp.ndarray):
    """ Computes the predict-update routine of the Extended Kalman Filter equations
    using temporal parallelization and returns a series of filtered_states TODO:reference

    Parameters
    ----------
    initial_state: MVNormalParameters
        prior belief on the initial state distribution
    observations: (n, K) array
        array of n observations of dimension K
    transition_function: callable
        transition function of the state space model
    transition_covariance: (D, D) array
        transition covariance for each time step
    observation_function: callable
        observation function of the state space model
    observation_covariance: (K, K) array
        observation error covariances for each time step

    Returns
    -------
    filtered_states: MVNormalParameters
        list of filtered states

    """
    n_observations = observations.shape[0]

    @vmap
    def make_params(observations, i):
        return make_associative_filtering_params(observation_function, observation_covariance,
                                                 transition_function, transition_covariance, observations,
                                                 i, initial_state.mean,
                                                 initial_state.cov)

    As, bs, Cs, etas, Js = make_params(observations, jnp.arange(n_observations))

    _, filtered_means, filtered_covariances, _, _ = lax.associative_scan(filtering_operator, (As, bs, Cs, etas, Js))

    return vmap(MVNormalParameters)(filtered_means, filtered_covariances)
