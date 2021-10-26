from typing import Callable

import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax, vmap

from parsmooth.parallel.ckf import filter_routine
from parsmooth.utils import MVNormalParameters
from .operators import smoothing_operator
from ..cubature_common import get_sigma_points, SigmaPoints, get_mv_normal_parameters, covariance_sigma_points


def make_associative_smoothing_params(transition_function, Qk, i, n, filtered_state, linearization_state):
    predicate = i == n - 1

    def _last(_):
        return filtered_state.mean, jnp.zeros_like(filtered_state.cov), filtered_state.cov

    def _generic(_):
        return _make_associative_smoothing_params_generic(transition_function, Qk, filtered_state, linearization_state)

    return lax.cond(predicate,
                    _last,  # take initial
                    _generic,  # take generic
                    None)


def _make_associative_smoothing_params_generic(transition_function, Qk, filtered_state, linearization_state):
    # Prediction part
    sigma_points = get_sigma_points(linearization_state)

    propagated_points = transition_function(sigma_points.points)
    propagated_sigma_points = SigmaPoints(propagated_points, sigma_points.wm, sigma_points.wc)
    propagated_state = get_mv_normal_parameters(propagated_sigma_points)

    pred_cross_covariance = covariance_sigma_points(sigma_points,
                                                    linearization_state.mean,
                                                    propagated_sigma_points,
                                                    propagated_state.mean)

    F = jlinalg.solve(linearization_state.cov, pred_cross_covariance,
                      sym_pos=True).T  # Linearized transition function

    Pp = Qk + propagated_state.cov + F @ (filtered_state.cov - linearization_state.cov) @ F.T

    E = jlinalg.solve(Pp, F @ filtered_state.cov, sym_pos=True).T

    g = filtered_state.mean - E @ (propagated_state.mean + F @ (filtered_state.mean - linearization_state.mean))
    L = filtered_state.cov - E @ F @ filtered_state.cov

    return g, E, 0.5 * (L + L.T)


def smoother_routine(transition_function: Callable,
                     transition_covariance: jnp.ndarray,
                     filtered_states: MVNormalParameters,
                     linearization_states: MVNormalParameters = None):
    """ Computes the predict-update routine of the Extended Kalman Filter equations
    using temporal parallelization and returns a series of filtered_states TODO:reference

    Parameters
    ----------
    transition_function: callable
        transition function of the state space model
    transition_covariance: (D, D) array
        transition covariance for each time step
        observation error covariances for each time step
    filtered_states: MVNormalParameters
        states resulting from (iterated) EKF
    linearization_states: MVNormalParameters, optional
        states at which to compute the cubature linearized functions

    Returns
    -------
    filtered_states: MVNormalParameters
        list of filtered states

    """
    n_observations = filtered_states.mean.shape[0]

    @vmap
    def make_params(i, filtered_state, linearization_state):
        if linearization_state is None:
            linearization_state = filtered_state
        return make_associative_smoothing_params(transition_function, transition_covariance,
                                                 i, n_observations, filtered_state, linearization_state)

    gs, Es, Ls = make_params(jnp.arange(n_observations), filtered_states, linearization_states)

    smoothed_means, _, smoothed_covariances = lax.associative_scan(smoothing_operator, (gs, Es, Ls), reverse=True)

    return vmap(MVNormalParameters)(smoothed_means, smoothed_covariances)


def iterated_smoother_routine(initial_state: MVNormalParameters,
                              observations: jnp.ndarray,
                              transition_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                              transition_covariance: jnp.ndarray,
                              observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                              observation_covariance: jnp.ndarray,
                              initial_linearization_states: MVNormalParameters = None,
                              n_iter: int = 100,
                              propagate_first: bool = True):
    """
    Computes the Gauss-Newton iterated cubature Kalman smoother

    Parameters
    ----------
    initial_state: MVNormalParameters
        prior belief on the initial state distribution
    observations: (n, K) array
        array of n observations of dimension K
    transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
        transition function of the state space model
    transition_covariance: (D, D) array
        transition covariances for each time step, if passed only one, it is repeated n times
    observation_function: callable :math:`h(x_t,\epsilon_t)\mapsto y_t`
        observation function of the state space model
    observation_covariance: (K, K)  array
        observation error covariances for each time step, if passed only one, it is repeated n times
    initial_linearization_states: MVNormalParameters, optional
        points at which to compute the jacobians durning the first pass.
    n_iter: int
        number of times the filter-smoother routine is computed
    propagate_first: bool, optional
        Is the first step a transition or an update? i.e. False if the initial time step has
        an associated observation. Default is True.
    Returns
    -------
    iterated_smoothed_trajectories: MVNormalParameters
        The result of the smoothing routine

    """

    def body(linearization_points, _):
        filtered_states = filter_routine(initial_state, observations, transition_function, transition_covariance,
                                         observation_function, observation_covariance, linearization_points,
                                         propagate_first)
        return smoother_routine(transition_function, transition_covariance, filtered_states,
                                linearization_points), None

    if initial_linearization_states is None:
        initial_linearization_states = body(None, None)[0]

    iterated_smoothed_trajectories, _ = lax.scan(body, initial_linearization_states, jnp.arange(n_iter))
    return iterated_smoothed_trajectories
