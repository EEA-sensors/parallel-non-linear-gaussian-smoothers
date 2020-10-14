from typing import Callable

import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax, vmap, jacfwd

from pekf.utils import MVNormalParameters
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
    L = Pk - E @ F @ Pk

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

    return vmap(MVNormalParameters)(smoothed_means, smoothed_covariances)
