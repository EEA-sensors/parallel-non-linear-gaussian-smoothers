from typing import Callable

import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax, vmap, jacfwd

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

    AIpCJ_inv = jlinalg.solve(IpCJ.T, A2.T, sym_pos=False).T
    AIpJC_inv = jlinalg.solve(IpJC.T, A1, sym_pos=False).T

    A = jnp.dot(AIpCJ_inv, A1)
    b = jnp.dot(AIpCJ_inv, b1 + jnp.dot(C1, eta2)) + b2
    C = jnp.dot(AIpCJ_inv, jnp.dot(C1, A2.T)) + C2
    eta = jnp.dot(AIpJC_inv, eta2 - jnp.dot(J2, b1)) + eta1
    J = jnp.dot(AIpJC_inv, jnp.dot(J2, A1)) + J1
    return A, b, C, eta, J


def make_associative_filtering_params(observation_function, Rk, transition_function, Qk_1, yk, i, m0, P0, x_k_1, x_k):
    predicate = i == 0

    jac_obs = jacfwd(observation_function, 0)
    jac_trans = jacfwd(transition_function, 0)

    def _first(_):
        return _make_associative_filtering_params_first(observation_function, jac_obs, Rk, transition_function,
                                                        jac_trans, Qk_1, m0, P0, yk)

    def _generic(_):
        return _make_associative_filtering_params_generic(observation_function, jac_obs, Rk, transition_function,
                                                          jac_trans, x_k_1, x_k, Qk_1, yk)

    return lax.cond(predicate,
                    _first,  # take initial
                    _generic,  # take generic
                    None)


def _make_associative_filtering_params_first(observation_function, jac_observation_function, R, transition_function,
                                             jac_transition_function, Q, m0, P0, y):
    F = jac_transition_function(m0)

    m1 = transition_function(m0)
    P1 = F @ P0 @ F.T + Q

    H = jac_observation_function(m1)

    S = H @ P1 @ H.T + R
    K = jlinalg.solve(S, H @ P1, sym_pos=True).T
    A = jnp.zeros(F.shape)
    b = m1 + K @ (y - observation_function(m1))
    C = P1 - (K @ S @ K.T)

    eta = jnp.zeros(F.shape[0])
    J = jnp.zeros(F.shape)

    return A, b, C, eta, J


def _make_associative_filtering_params_generic(observation_function, jac_observation_function, Rk, transition_function,
                                               jac_transition_function, x_k_1, x_k, Qk_1, yk):
    F = jac_transition_function(x_k_1)
    H = jac_observation_function(x_k)

    alpha = observation_function(x_k) + H @ transition_function(x_k_1) - H @ F @ x_k_1 - H @ x_k
    residual = yk - alpha
    HQ = H @ Qk_1

    S = HQ @ H.T + Rk
    K = jlinalg.solve(S, HQ, sym_pos=True).T
    A = F - K @ H @ F
    b = K @ residual
    C = Qk_1 - K @ H @ Qk_1

    HF = H @ F

    temp = jlinalg.solve(S, HF, sym_pos=True).T
    eta = temp @ residual
    J = temp @ HF

    return A, b, C, eta, J


def filter_routine(initial_state: MVNormalParameters,
                   observations: jnp.ndarray,
                   transition_function: Callable,
                   transition_covariance: jnp.ndarray,
                   observation_function: Callable,
                   observation_covariance: jnp.ndarray,
                   linearisation_points: jnp.ndarray = None):
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
    linearisation_points: (n, D) array, optional
        points at which to compute the jacobians.

    Returns
    -------
    filtered_states: MVNormalParameters
        list of filtered states

    """
    n_observations = observations.shape[0]
    x_dim = initial_state.mean.shape[0]
    dtype = initial_state.mean.dtype
    if linearisation_points is None:
        linearisation_points = jnp.zeros((n_observations, x_dim), dtype=dtype)

    @vmap
    def make_params(observations, i, x_k_1, x_k):
        return make_associative_filtering_params(observation_function, observation_covariance,
                                                 transition_function, transition_covariance, observations,
                                                 i, initial_state.mean,
                                                 initial_state.cov, x_k_1, x_k)

    x_k_1_s = jnp.concatenate((initial_state.mean.reshape(1, -1), linearisation_points[1:]), 0)
    As, bs, Cs, etas, Js = make_params(observations, jnp.arange(n_observations), x_k_1_s, linearisation_points)
    _, filtered_means, filtered_covariances, _, _ = lax.associative_scan(filtering_operator, (As, bs, Cs, etas, Js))

    return vmap(MVNormalParameters)(filtered_means, filtered_covariances)
