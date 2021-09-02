from typing import Callable

import jax.numpy as jnp, scipy
import jax.scipy.linalg as jlinalg
from jax import lax, vmap, jacfwd

from parsmooth.utils import MVNParams
from .operators_sqrt import filtering_operator


def linearization_param(transition_function, observation_function, R, Q, x_hat, m0):
    
    
    jac_observation_function = jacfwd(observation_function, 0)
    jac_transition_function = jacfwd(transition_function, 0)
    
    x_hat_m0 = jnp.concatenate([m0.reshape(1,-1),x_hat], axis=0)
    x_hat_m0 = jnp.expand_dims(x_hat_m0, axis=1)
    x_2step = jnp.concatenate([x_hat_m0[:-1],x_hat_m0[1:]],axis=1)
  
    def body(_, x_2step):
        
        x_k_1 = x_2step[0]
        x_k = x_2step[1]
        F = jac_transition_function(x_k_1)   
        c = transition_function(x_k_1) - F @ x_k_1
        Qp = scipy.linalg.cholesky(Q, lower=True)
        
        H = jac_observation_function(x_k)   
        d = observation_function(x_k) - H @ x_k
        Rp = scipy.linalg.cholesky(R, lower=True)
        
        return None, (F, c, Qp, H, d, Rp)
    
    _, linear_param = lax.scan(body, None, x_2step)
    
    return linear_param


def Tria(A):
    tria_A = jlinalg.qr(A.T, mode = 'economic')[1].T
    return tria_A


def make_associative_filtering_params(F, c, W, H, d, V, y, m0, N0, i):      ##Note N0
    
    # W, V, and N0 are the square root of Q, R, and P0.
    
    predicate = i == 0

    def _first(_):
        return _make_associative_filtering_params_first(F, c, W, H, d, V, m0, N0, y)

    def _generic(_):
        return _make_associative_filtering_params_generic(F, c, W, H, d, V, y)

    return lax.cond(predicate,
                    _first,  # take initial
                    _generic,  # take generic
                    None)


def _make_associative_filtering_params_first(F, c, W, H, d, V, m0, N0, y):

    
    # W, V, and N0 are the square root of Q, R, and P0.
    
    nx = W.shape[0]
    ny = V.shape[0]
    
    m1 = F @ m0 + c
    N1_ = Tria(jnp.concatenate((F @ N0, W), axis = 1))
    Psi_ = jnp.block([[H @ N1_, V], [N1_, jnp.zeros((N1_.shape[0], V.shape[1]))]])
    Tria_Psi_ = Tria(Psi_)
    Psi11_ = Tria_Psi_[:ny , :ny]
    Psi21_ = Tria_Psi_[ny: ny + nx , :ny]
    Psi22_ = Tria_Psi_[ny: ny + nx , ny:]
    Y1 = Psi11_
    K1 = jlinalg.solve(Psi11_.T, Psi21_.T).T     # sym_pos?

    A = jnp.zeros(F.shape)
    b = m1 + K1 @ (y - H @ m1 - d)
    U = Psi22_

    Z1 = jlinalg.solve(Y1, H @ F).T
    eta = jlinalg.solve(Y1.T, Z1.T).T @ (y - H @ c - d)
    Z = jnp.block([Z1,jnp.zeros((nx,nx-ny))])

    return A, b, U, eta, Z


def _make_associative_filtering_params_generic(F, c, W, H, d, V, y):
    
    nx = W.shape[0]
    ny = V.shape[0]
    
    Psi = jnp.block([ [ H @ W, V ] , [ W , jnp.zeros((nx,ny)) ] ])
    Tria_Psi = Tria(Psi)
    Psi11 = Tria_Psi[:ny , :ny]
    Psi21 = Tria_Psi[ny:ny + nx , :ny]
    Psi22 = Tria_Psi[ny:ny + nx , ny:]
    Y = Psi11
    K = jlinalg.solve(Psi11.T, Psi21.T).T
    A = F - K @ H @ F
    b = c + K @ (y - H @ c - d)
    U = Psi22

    Z1 = jlinalg.solve(Y, H @ F).T
    eta = jlinalg.solve(Y.T, Z1.T).T @ (y - H @ c - d)

    Z = jnp.block([Z1,jnp.zeros((nx,nx-ny))])

    return A, b, U, eta, Z


def filter_routine(initial_state: MVNParams,
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
    initial_state: MVNParams
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
    filtered_states: MVNParams
        list of filtered states

    """
    n_observations = observations.shape[0]
    x_dim = initial_state.mean.shape[0]
    dtype = initial_state.mean.dtype
    if linearization_points is None:
        linearization_points = jnp.zeros((n_observations, x_dim), dtype=dtype)
        
    linear_param = linearization_param(transition_function, observation_function, observation_covariance, 
                                       transition_covariance, linearization_points, initial_state.mean)

    @vmap
    def make_params(obs, i, F, c, W, H, d, V):
        return make_associative_filtering_params(F, c, W, H, d, V, obs,
                                                 initial_state.mean,
                                                 initial_state.cov, i)

    As, bs, Us, etas, Zs = make_params(observations, jnp.arange(n_observations), *linear_param)

    _, filtered_means, filtered_covariances, _, _ = lax.associative_scan(filtering_operator, (As, bs, Us, etas, Zs))

    return vmap(MVNParams)(filtered_means, filtered_covariances), linear_param
