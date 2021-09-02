from typing import Callable, Tuple
from enum import Enum
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax, vmap

from parsmooth.utils import MVNParams

class LinearizationMethod(Enum):
    EXTENDED = "EXTENDED"

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


def make_associative_filtering_params(observation_function, Rk, transition_function, Qk_1, yk, i, m0, P0, x_k_1, x_k,
                                      linearization_method):
    predicate = i == 0

    def _first(_):
        return _make_associative_filtering_params_first(observation_function, Rk, transition_function,
                                                        Qk_1, m0, P0, x_k, yk, linearization_method)

    def _generic(_):
        return _make_associative_filtering_params_generic(observation_function, Rk, transition_function,
                                                          x_k_1, x_k, Qk_1, yk, linearization_method)

    return lax.cond(predicate,
                    _first,  # take initial
                    _generic,  # take generic
                    None)


def _make_associative_filtering_params_first(observation_function, obs_cov, transition_function, noise_cov, m0, P0, x_k,
                                             y, linearization_method):
    F, c, Q = linearization_method(transition_function, x_k, noise_cov)
    H, d, R = linearization_method(observation_function, x_k, obs_cov)

    m1 = F @ m0 + c
    P1 = F @ P0 @ F.T + Q

    S = H @ P1 @ H.T + R
    K = jlinalg.solve(S, H @ P1, sym_pos=True).T
    A = jnp.zeros(F.shape)

    b = m1 + K @ (y - H @ m1 - d)
    C = P1 - (K @ S @ K.T)

    eta = jnp.zeros(F.shape[0])
    J = jnp.zeros(F.shape)

    return A, b, C, eta, J


def _make_associative_filtering_params_generic(observation_function, obs_cov, transition_function,
                                               x_k_1, x_k, noise_cov, yk, linearization_method):
    F, c, Q = linearization_method(transition_function, x_k_1, noise_cov)
    H, d, R = linearization_method(observation_function, x_k, obs_cov)

    S = H @ Q @ H.T + R
    S_invH = jlinalg.solve(S, H, sym_pos=True)
    K = (S_invH @ Q).T
    A = F - K @ H @ F
    b = c + K @ (yk - H @ c - d)
    C = Q - K @ H @ Q

    temp = (S_invH @ F).T
    eta = temp @ (yk - H @ c - d)
    J = temp @ H @ F

    return A, b, C, eta, J


def filter_routine(initial_state: MVNParams,
                   observations: jnp.ndarray,
                   transition_function: Callable or Tuple[Callable],
                   transition_covariance: jnp.ndarray,
                   observation_function: Callable or Tuple[Callable],
                   observation_covariance: jnp.ndarray,
                   linearization_method: callable,
                   linearization_points: jnp.ndarray,
                   ):
    """ Computes the parallel Kalman filter routine given a linearization
     and returns a series of filtered_states TODO:reference

    Parameters
    ----------
    initial_state: MVNParams
        prior belief on the initial state distribution
    observations: (n, K) array
        array of n observations of dimension K
    transition_function: callable or tuple of callable
        transition function of the state space model
    transition_covariance: (D, D) array
        transition covariance for each time step
    observation_function: callable or tuple of callable
        observation function of the state space model
    observation_covariance: (K, K) array
        observation error covariances for each time step
    linearization_method: callable
        one of taylor or sigma_points linearization method
    linearization_points: (n, D) array
        points at which to compute the jacobians.

    Returns
    -------
    filtered_states: MVNParams
        list of filtered states

    """
    n_observations = observations.shape[0]

    @vmap
    def make_params(obs, i, x_k_1, x_k):
        return make_associative_filtering_params(observation_function, observation_covariance,
                                                 transition_function, transition_covariance, obs,
                                                 i, initial_state.mean,
                                                 initial_state.cov, x_k_1, x_k, linearization_method)

    x_k_1_s = jnp.concatenate((initial_state.mean.reshape(1, -1), linearization_points[:-1]), 0)
    As, bs, Cs, etas, Js = make_params(observations, jnp.arange(n_observations), x_k_1_s, linearization_points)
    _, filtered_means, filtered_covariances, _, _ = lax.associative_scan(filtering_operator, (As, bs, Cs, etas, Js))

    return vmap(MVNParams)(filtered_means, filtered_covariances)
