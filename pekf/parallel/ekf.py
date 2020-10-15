from typing import Callable

import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax, vmap, jacfwd

from pekf.utils import MVNormalParameters
from .operators import filtering_operator


def make_associative_filtering_params(observation_function, Rk, transition_function, Qk_1, yk, i, m0, P0, x_k_1, x_k):
    predicate = i == 0

    jac_obs = jacfwd(observation_function, 0)
    jac_trans = jacfwd(transition_function, 0)

    def _first(_):
        return _make_associative_filtering_params_first(observation_function, jac_obs, Rk, transition_function,
                                                        jac_trans, Qk_1, m0, P0, x_k, yk)

    def _generic(_):
        return _make_associative_filtering_params_generic(observation_function, jac_obs, Rk, transition_function,
                                                          jac_trans, x_k_1, x_k, Qk_1, yk)

    return lax.cond(predicate,
                    _first,  # take initial
                    _generic,  # take generic
                    None)


def _make_associative_filtering_params_first(observation_function, jac_observation_function, R, transition_function,
                                             jac_transition_function, Q, m0, P0, x_k, y):
    F = jac_transition_function(m0)

    m1 = transition_function(m0)
    P1 = F @ P0 @ F.T + Q

    H = jac_observation_function(x_k)

    S = H @ P1 @ H.T + R
    K = jlinalg.solve(S, H @ P1, sym_pos=True).T
    A = jnp.zeros(F.shape)

    alpha = observation_function(x_k) + H @ (m1 - x_k)

    b = m1 + K @ (y - alpha)
    C = P1 - (K @ S @ K.T)

    eta = jnp.zeros(F.shape[0])
    J = jnp.zeros(F.shape)

    return A, b, C, eta, J


def _make_associative_filtering_params_generic(observation_function, jac_observation_function, Rk, transition_function,
                                               jac_transition_function, x_k_1, x_k, Qk_1, yk):
    F = jac_transition_function(x_k_1)
    H = jac_observation_function(x_k)

    F_x_k_1 = F @ x_k_1
    x_k_hat = transition_function(x_k_1)

    alpha = observation_function(x_k) + H @ (x_k_hat - F_x_k_1 - x_k)
    residual = yk - alpha
    HQ = H @ Qk_1

    S = HQ @ H.T + Rk
    S_invH = jlinalg.solve(S, H, sym_pos=True)
    K = (S_invH @ Qk_1).T
    A = F - K @ H @ F
    b = K @ residual + x_k_hat - F_x_k_1
    C = Qk_1 - K @ H @ Qk_1

    HF = H @ F

    temp = (S_invH @ F).T
    eta = temp @ residual
    J = temp @ HF

    return A, b, C, eta, J


def filter_routine(initial_state: MVNormalParameters,
                   observations: jnp.ndarray,
                   transition_function: Callable,
                   transition_covariance: jnp.ndarray,
                   observation_function: Callable,
                   observation_covariance: jnp.ndarray,
                   linearization_points: jnp.ndarray = None):
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
    linearization_points: (n, D) array, optional
        points at which to compute the jacobians.

    Returns
    -------
    filtered_states: MVNormalParameters
        list of filtered states

    """
    n_observations = observations.shape[0]
    x_dim = initial_state.mean.shape[0]
    dtype = initial_state.mean.dtype
    if linearization_points is None:
        linearization_points = jnp.zeros((n_observations, x_dim), dtype=dtype)

    @vmap
    def make_params(obs, i, x_k_1, x_k):
        return make_associative_filtering_params(observation_function, observation_covariance,
                                                 transition_function, transition_covariance, obs,
                                                 i, initial_state.mean,
                                                 initial_state.cov, x_k_1, x_k)

    x_k_1_s = jnp.concatenate((initial_state.mean.reshape(1, -1), linearization_points[:-1]), 0)
    As, bs, Cs, etas, Js = make_params(observations, jnp.arange(n_observations), x_k_1_s, linearization_points)
    _, filtered_means, filtered_covariances, _, _ = lax.associative_scan(filtering_operator, (As, bs, Cs, etas, Js))

    return vmap(MVNormalParameters)(filtered_means, filtered_covariances)
