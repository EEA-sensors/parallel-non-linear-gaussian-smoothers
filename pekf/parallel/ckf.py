from typing import Callable

import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax, vmap

from pekf.utils import MVNormalParameters, make_matrices_parameters
from .operators import filtering_operator
from ..cubature_common import get_sigma_points, get_mv_normal_parameters, covariance_sigma_points, SigmaPoints


def make_associative_filtering_params(observation_function, Rk, transition_function, Qk_1, yk, i, initial_state,
                                      prev_linearization_state, linearization_state):
    predicate = i == 0

    def _first(_):
        return _make_associative_filtering_params_first(observation_function, Rk, transition_function, Qk_1,
                                                        initial_state, linearization_state, yk)

    def _generic(_):
        return _make_associative_filtering_params_generic(observation_function, Rk, transition_function, Qk_1,
                                                          prev_linearization_state, linearization_state, yk)

    return lax.cond(predicate,
                    _first,  # take initial
                    _generic,  # take generic
                    None)


def _make_associative_filtering_params_first(observation_function, R, transition_function, Q, initial_state,
                                             linearization_state, y):
    # Prediction part
    initial_sigma_points = get_sigma_points(initial_state)
    propagated_points = transition_function(initial_sigma_points.points)
    propagated_sigma_points = SigmaPoints(propagated_points, initial_sigma_points.wm, initial_sigma_points.wc)
    propagated_state = get_mv_normal_parameters(propagated_sigma_points)

    pred_cross_covariance = covariance_sigma_points(initial_sigma_points, initial_state.mean, propagated_sigma_points,
                                                    propagated_state.mean)

    F = jlinalg.solve(initial_state.cov, pred_cross_covariance, sym_pos=True).T  # Linearized transition function

    m1 = propagated_state.mean
    P1 = propagated_state.cov + Q

    # Update part
    linearization_points = get_sigma_points(linearization_state)
    obs_points = observation_function(linearization_points.points)
    obs_sigma_points = SigmaPoints(obs_points, linearization_points.wm, linearization_points.wc)
    obs_mvn = get_mv_normal_parameters(obs_sigma_points)
    update_cross_covariance = covariance_sigma_points(linearization_points, linearization_state.mean,
                                                      obs_sigma_points, obs_mvn.mean)

    H = jlinalg.solve(linearization_state.cov, update_cross_covariance, sym_pos=True).T
    d = obs_mvn.mean - jnp.dot(H, linearization_state.mean)
    predicted_observation = H @ m1 + d

    S = H @ (P1 - linearization_state.cov) @ H.T + R + obs_mvn.cov
    K = jlinalg.solve(S, H @ P1, sym_pos=True).T
    A = jnp.zeros(F.shape)
    b = m1 + K @ (y - predicted_observation)
    C = P1 - K @ S @ K.T

    eta = jnp.zeros(F.shape[0])
    J = jnp.zeros(F.shape)

    return A, b, 0.5 * (C + C.T), eta, J


def _make_associative_filtering_params_generic(observation_function, Rk, transition_function, Qk_1,
                                               prev_linearization_state, linearization_state, yk):
    # Prediction part
    sigma_points = get_sigma_points(prev_linearization_state)

    propagated_points = transition_function(sigma_points.points)
    propagated_sigma_points = SigmaPoints(propagated_points, sigma_points.wm, sigma_points.wc)
    propagated_state = get_mv_normal_parameters(propagated_sigma_points)

    pred_cross_covariance = covariance_sigma_points(sigma_points, prev_linearization_state.mean,
                                                    propagated_sigma_points,
                                                    propagated_state.mean)

    F = jlinalg.solve(prev_linearization_state.cov, pred_cross_covariance,
                      sym_pos=True).T  # Linearized transition function
    pred_mean_residual = propagated_state.mean - F @ prev_linearization_state.mean
    pred_cov_residual = propagated_state.cov - F @ prev_linearization_state.cov @ F.T + Qk_1

    # Update part
    linearization_points = get_sigma_points(linearization_state)
    obs_points = observation_function(linearization_points.points)
    obs_sigma_points = SigmaPoints(obs_points, linearization_points.wm, linearization_points.wc)
    obs_mvn = get_mv_normal_parameters(obs_sigma_points)
    update_cross_covariance = covariance_sigma_points(linearization_points,
                                                      linearization_state.mean,
                                                      obs_sigma_points,
                                                      obs_mvn.mean)

    H = jlinalg.solve(linearization_state.cov, update_cross_covariance, sym_pos=True).T
    obs_mean_residual = obs_mvn.mean - jnp.dot(H, linearization_state.mean)
    obs_cov_residual = obs_mvn.cov - H @ linearization_state.cov @ H.T

    S = H @ pred_cov_residual @ H.T + Rk + obs_cov_residual  # total residual covariance
    total_obs_residual = (yk - H @ pred_mean_residual - obs_mean_residual)
    S_invH = jlinalg.solve(S, H, sym_pos=True)

    K = (S_invH @ pred_cov_residual).T
    A = F - K @ H @ F
    b = pred_mean_residual + K @ total_obs_residual
    C = pred_cov_residual - K @ S @ K.T

    temp = (S_invH @ F).T
    HF = H @ F

    eta = temp @ total_obs_residual
    J = temp @ HF
    return A, b, 0.5 * (C + C.T), eta, J


def filter_routine(initial_state: MVNormalParameters,
                   observations: jnp.ndarray,
                   transition_function: Callable,
                   transition_covariance: jnp.ndarray,
                   observation_function: Callable,
                   observation_covariance: jnp.ndarray,
                   linearization_states: MVNormalParameters = None):
    """ Computes the predict-update routine of the Cubature Kalman Filter equations
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
    linearization_states: MVNormalParameters, optional
        in the case of Sigma-Point .

    Returns
    -------
    filtered_states: MVNormalParameters
        list of filtered states

    """
    n_observations = observations.shape[0]
    x_dim = initial_state.mean.shape[0]
    dtype = initial_state.mean.dtype

    if linearization_states is None:
        linearization_mean = jnp.zeros((n_observations, x_dim), dtype=dtype)
        linearization_cov = make_matrices_parameters(jnp.eye(x_dim, dtype=dtype), n_observations)
        linearization_states = MVNormalParameters(linearization_mean, linearization_cov)

    @vmap
    def make_params(obs, i, prev_linearization_state, linearisation_state):
        return make_associative_filtering_params(observation_function, observation_covariance, transition_function,
                                                 transition_covariance, obs, i, initial_state,
                                                 prev_linearization_state, linearisation_state)

    x_k_1_s = jnp.concatenate((initial_state.mean.reshape(1, -1), linearization_states.mean[:-1]), 0)
    P_k_1_s = jnp.concatenate((initial_state.cov.reshape(1, x_dim, x_dim), linearization_states.cov[:-1]), 0)
    prev_linearization_states = MVNormalParameters(x_k_1_s, P_k_1_s)

    As, bs, Cs, etas, Js = make_params(observations, jnp.arange(n_observations), prev_linearization_states,
                                       linearization_states)
    _, filtered_means, filtered_covariances, _, _ = lax.associative_scan(filtering_operator, (As, bs, Cs, etas, Js))

    return vmap(MVNormalParameters)(filtered_means, filtered_covariances)
