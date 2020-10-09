from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy.linalg as linalg
from jax import lax, jit

__all__ = ["make_parameters", "get_data"]


def _transition_function(x, dt):
    """ Deterministic transition function used in the state space model

    Parameters
    ----------
    x: array_like
        The current state
    dt: float
        Time step between observations

    Returns
    -------
    out: array_like
        The transitioned state
    """
    w = x[-1]
    predicate = jnp.abs(w) < 1e-6

    coswt = jnp.cos(w * dt)
    sinwt = jnp.sin(w * dt)

    def true_fun(_):
        return coswt, 0., sinwt, dt

    def false_fun(_):
        coswto = coswt - 1
        return coswt, coswto / w, sinwt, sinwt / w

    coswt, coswtopw, sinwt, sinwtpw = lax.cond(predicate, true_fun, false_fun, None)

    F = jnp.array([[1, 0, sinwtpw, -coswtopw, 0],
                   [0, 1, coswtopw, sinwtpw, 0],
                   [0, 0, coswt, sinwt, 0],
                   [0, 0, -sinwt, coswt, 0],
                   [0, 0, 0, 0, 1]])
    return F @ x


def _observation_function(x, s1, s2):
    """
    Returns the observed angles as function of the state and the sensors locations

    Parameters
    ----------
    x: array_like
        The current state
    s1: array_like
        The first sensor location
    s2: array_like
        The second sensor location

    Returns
    -------
    y: array_like
        The observed angles, the first component is the angle w.r.t. the first sensor, the second w.r.t the second.
    """
    return jnp.array([jnp.arctan2(x[1] - s1[1], x[0] - s1[0]),
                      jnp.arctan2(x[1] - s2[1], x[0] - s2[0])])


def make_parameters(qc, qw, r, dt, s1, s2):
    """ Discretizes the model with continuous transition noise qc, for step-size dt.
    The model is described in "Multitarget-multisensor tracking: principles and techniques" by
    Bar-Shalom, Yaakov and Li, Xiao-Rong

    Parameters
    ----------
    qc: float
        Transition covariance of the continuous SSM
    qw: float
        Transition covariance of the continuous SSM
    r: float
        Observation error standard deviation
    dt: float
        Discretization time step
    s1: array_like
        The location of the first sensor
    s2: array_like
        The location of the second sensor

    Returns
    -------
    Q: array_like
        The transition covariance matrix for the discrete SSM
    R: array_like
        The observation covariance matrix
    observation_function: callable
        The observation function
    transition_function: callable
        The transition function
    """

    Q = jnp.array([[qc * dt ** 3 / 3, 0, qc * dt ** 2 / 2, 0, 0],
                   [0, qc * dt ** 3 / 3, 0, qc * dt ** 2 / 2, 0],
                   [qc * dt ** 2 / 2, 0, qc * dt, 0, 0],
                   [0, qc * dt ** 2 / 2, 0, qc * dt, 0],
                   [0, 0, 0, 0, dt * qw]])

    R = r ** 2 * jnp.eye(2)

    observation_function = jit(partial(_observation_function, s1=s1, s2=s2))
    transition_function = jit(partial(_transition_function, dt=dt))

    return Q, R, observation_function, transition_function


@nb.njit
def _get_data(x, dt, a_s, s1, s2, r, normals, observations, true_states):
    for i, a in enumerate(a_s):
        with nb.objmode(x='float32[::1]'):
            F = np.array([[0, 0, 1, 0],
                          [0, 0, 0, 1],
                          [0, 0, 0, a],
                          [0, 0, -a, 0]], dtype=np.float32)
            x = linalg.expm(F * dt) @ x
        y1 = np.arctan2(x[1] - s1[1], x[0] - s1[0]) + r * normals[i, 0]
        y2 = np.arctan2(x[1] - s2[1], x[0] - s2[0]) + r * normals[i, 1]

        observations[i] = [y1, y2]
        observations[i] = [y1, y2]
        true_states[i] = np.concatenate((x, np.array([a])))
    # return true_states, observations


def get_data(x0, dt, r, T, s1, s2, random_state=None):
    """

    Parameters
    ----------
    x0: array_like
        true initial state
    dt: float
        time step for observations
    r: float
        observation model standard deviation
    T: int
        number of time steps
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
    a_s = 1 + 10 * dt * np.cumsum(random_state.randn(T))
    a_s = a_s.astype(np.float32)
    s1 = np.asarray(s1, dtype=np.float32)
    s2 = np.asarray(s2, dtype=np.float32)

    x = np.copy(x0).astype(np.float32)
    observations = np.empty((T, 2), dtype=np.float32)
    true_states = np.empty((T, 5), dtype=np.float32)
    ts = np.linspace(dt, (T + 1) * dt, T).astype(np.float32)

    normals = random_state.randn(T, 2).astype(np.float32)

    _get_data(x, dt, a_s, s1, s2, r, normals, observations, true_states)
    return ts, true_states, observations


def plot_bearings(states, labels, s1, s2, figsize=(10, 10), quiver=False):
    """

    Parameters
    ----------
    states: list of array_like
        list of states to plot
    labels: list of str
        list of lables for the states
    s1: array_like
        first sensor
    s2: array_like
        second sensor
    figsize: tuple of int
        figure size in inches
    quiver: bool
        show the velocity field

    Returns
    -------

    """
    fig, ax = plt.subplots(figsize=figsize)

    if not isinstance(states, list):
        states = [states]

    if not isinstance(labels, list):
        labels = [labels]

    for label, state in zip(labels, states):
        ax.plot(*state[:, :2].T, linestyle='--', label=label)
        if quiver:
            ax.quiver(*state[::10].T, units='xy', scale=4, width=0.01)
    ax.scatter(*s1, marker="o", s=200, label="Sensor 1", color='k')
    ax.scatter(*s2, marker="x", s=200, label="Sensor 2", color='k')

    ax.legend(loc="lower left")
