from typing import Callable

import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax, vmap, jacfwd

from parsmooth.utils import MVNParams
from .ekf_sqrt import filter_routine
from .operators_sqrt import smoothing_operator



def Tria(A):
    tria_A = jlinalg.qr(A.T, mode = 'economic')[1].T
    return tria_A


def make_associative_smoothing_params(F, c, W, i, n, mk, Nk):   #Note Nk
    predicate = i == n - 1

    def _last(_):
        return mk, jnp.zeros_like(Nk), Nk

    def _generic(_):
        return _make_associative_smoothing_params_generic(F, c, W, mk, Nk)

    return lax.cond(predicate,
                    _last,  # take initial
                    _generic,  # take generic
                    None)


def _make_associative_smoothing_params_generic(F, c, W, m, N):
    
    nx = F.shape[0]
    Phi = jnp.block([ [ F @ N , W ], [N , jnp.zeros((N.shape[0] , W.shape[1]))] ])
    Tria_Phi = Tria(Phi)
    Phi11 = Tria_Phi[:nx , :nx]
    Phi21 = Tria_Phi[nx: nx + N.shape[0] , :nx]
    Phi22 = Tria_Phi[nx: nx + N.shape[0] , nx:]  

    E = jlinalg.solve(Phi11.T, Phi21.T).T
    D = Phi22
    g = m - E @ (F @ m + c)

    return g, E, D


def smoother_routine(filtered_states: MVNParams,
                     linear_param: tuple):
    """ Computes the predict-update routine of the Extended Kalman Filter equations
    using temporal parallelization and returns a series of filtered_states TODO:reference

    Parameters
    ----------
    filtered_states: MVNParams
        states resulting from (iterated) EKF
    linearisation_points: (n, D) array, optional
        points at which to compute the jacobians, typically previous run.

    Returns
    -------
    filtered_states: MVNParams
        list of filtered states

    """
    n_observations = filtered_states.mean.shape[0]


    @vmap
    def make_params(i, mk, Nk, F, c, W, H, d, V):
        return make_associative_smoothing_params(F, c, W, i, n_observations, mk, Nk)

    gs, Es, Ds = make_params(jnp.arange(n_observations), filtered_states.mean,
                             filtered_states.cov, *linear_param)

    smoothed_means, _, smoothed_covariances = lax.associative_scan(smoothing_operator, (gs, Es, Ds), reverse=True)

    return vmap(MVNParams)(smoothed_means, smoothed_covariances)


def iterated_smoother_routine(initial_state: MVNParams,
                              observations: jnp.ndarray,
                              transition_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                              transition_covariance: jnp.ndarray,
                              observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                              observation_covariance: jnp.ndarray,
                              initial_linearization_states: MVNParams = None,
                              n_iter: int = 100):
    """
    Computes the Gauss-Newton iterated extended Kalman smoother

    Parameters
    ----------
    initial_state: MVNParams
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
    initial_linearization_states: (N, D) array, optional
        points at which to compute the jacobians durning the first pass.
    n_iter: int
        number of times the filter-smoother routine is computed

    Returns
    -------
    iterated_smoothed_trajectories: MVNParams
        The result of the smoothing routine

    """

    def body(linearization_points, _):
        if linearization_points is not None:
            linearization_points = linearization_points.mean
        filtered_states, linear_param = filter_routine(initial_state, observations, transition_function, transition_covariance,
                                         observation_function, observation_covariance, linearization_points)
        return smoother_routine(filtered_states, linear_param), None

    if initial_linearization_states is None:
        initial_linearization_states = body(None, None)[0]

    iterated_smoothed_trajectories, _ = lax.scan(body, initial_linearization_states, jnp.arange(n_iter))
    return iterated_smoothed_trajectories
