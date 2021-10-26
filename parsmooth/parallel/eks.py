from typing import Callable

import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax, vmap, jacfwd

from parsmooth.utils import MVNormalParameters
from .ekf import filter_routine
from .operators import smoothing_operator


def make_associative_smoothing_params(transition_function, Qk, i, n, mk, Pk, xk):
    predicate = i == n - 1

    jac_trans = jacfwd(transition_function, 0)

    def _last(_):
        return mk, jnp.zeros_like(Pk), Pk

    def _generic(_):
        return _make_associative_smoothing_params_generic(transition_function, jac_trans, Qk, mk, Pk, xk)

    return lax.cond(predicate,
                    _last,  # take initial
                    _generic,  # take generic
                    None)


def _make_associative_smoothing_params_generic(transition_function, jac_transition_function, Qk, mk, Pk, xk):
    F = jac_transition_function(xk)
    Pp = F @ Pk @ F.T + Qk

    E = jlinalg.solve(Pp, F @ Pk, sym_pos=True).T

    g = mk - E @ (transition_function(xk) + F @ (mk - xk))
    L = Pk - E @ Pp @ E.T

    return g, E, L


def smoother_routine(transition_function: Callable,
                     transition_covariance: jnp.ndarray,
                     filtered_states: MVNormalParameters,
                     linearisation_points: jnp.ndarray = None):
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
    linearisation_points: (n, D) array, optional
        points at which to compute the jacobians, typically previous run.

    Returns
    -------
    filtered_states: MVNormalParameters
        list of filtered states

    """
    n_observations = filtered_states.mean.shape[0]

    if linearisation_points is None:
        linearisation_points = filtered_states.mean

    @vmap
    def make_params(i, mk, Pk, xk):
        return make_associative_smoothing_params(transition_function, transition_covariance,
                                                 i, n_observations, mk, Pk, xk)

    gs, Es, Ls = make_params(jnp.arange(n_observations), filtered_states.mean,
                             filtered_states.cov, linearisation_points)

    smoothed_means, _, smoothed_covariances = lax.associative_scan(smoothing_operator, (gs, Es, Ls), reverse=True)
    return MVNormalParameters(smoothed_means, smoothed_covariances)


def iterated_smoother_routine(initial_state: MVNormalParameters,
                              observations: jnp.ndarray,
                              transition_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                              transition_covariance: jnp.ndarray,
                              observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                              observation_covariance: jnp.ndarray,
                              initial_linearization_points: jnp.ndarray = None,
                              n_iter: int = 100,
                              propagate_first: bool = True):
    """
    Computes the Gauss-Newton iterated extended Kalman smoother

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
    initial_linearization_points: (N, D) array, optional
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
        if linearization_points is not None:
            linearization_points = linearization_points.mean if isinstance(linearization_points,
                                                                           MVNormalParameters) else linearization_points
        filtered_states = filter_routine(initial_state, observations, transition_function, transition_covariance,
                                         observation_function, observation_covariance, linearization_points,
                                         propagate_first)
        return smoother_routine(transition_function, transition_covariance, filtered_states,
                                linearization_points), None

    if initial_linearization_points is None:
        initial_linearization_points = body(None, None)[0]

    iterated_smoothed_trajectories, _ = lax.scan(body, initial_linearization_points, jnp.arange(n_iter))
    return iterated_smoothed_trajectories
