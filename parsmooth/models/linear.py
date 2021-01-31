from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit

__all__ = ["make_parameters", "get_data"]


def _transition_function(x, A):
    """ Deterministic transition function used in the state space model

    Parameters
    ----------
    x: array_like
        The current state
    A: array_like
        transition matrix

    Returns
    -------
    out: array_like
        The transitioned state
    """
    return jnp.dot(A, x)


def _observation_function(x, H):
    """
    Returns the observed angles as function of the state and the sensors locations

    Parameters
    ----------
    x: array_like
        The current state
    H: array_like
        observation matrix

    Returns
    -------
    y: array_like
        The observed data
    """
    return jnp.dot(H, x)


def make_parameters(r, q):
    A = 0.5 * jnp.eye(2)
    Q = q * jnp.eye(2)
    R = r * jnp.eye(1)
    H = jnp.array([[1., 0.5]])

    observation_function = jit(partial(_observation_function, H=H))
    transition_function = jit(partial(_transition_function, A=A))

    return A, H, Q, R, observation_function, transition_function


def get_data(x0, A, H, R, Q, T, random_state=None):
    """

    Parameters
    ----------
    x0: array_like
        true initial state
    A: array_like
        transition matrix
    H: array_like
        transition matrix
    R: array_like
        observation model covariance
    Q: array_like
        noise covariance
    s1: array_like
        The location of the first sensor
    s2: array_like
        The location of the second sensor
    random_state: np.random.RandomState or int, optional
        numpy random state

    Returns
    -------
    ts: array_like
        array of time steps
    true_states: array_like
        array of true states
    observations: array_like
        array of observations
    """
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    R_shape = R.shape[0]
    Q_shape = Q.shape[0]
    normals = random_state.randn(T, Q_shape + R_shape).astype(np.float32)
    chol_R = np.linalg.cholesky(R)
    chol_Q = np.linalg.cholesky(Q)

    x = np.copy(x0).astype(np.float32)
    observations = np.empty((T, R_shape), dtype=np.float32)
    true_states = np.empty((T, Q_shape), dtype=np.float32)

    for i in range(T):
        x = A @ x + chol_Q @ normals[i, :Q_shape]
        y = H @ x + chol_R @ normals[i, Q_shape:]
        true_states[i] = x
        observations[i] = y

    ts = np.linspace(1, T, T).astype(np.float32)

    return ts, true_states, observations
