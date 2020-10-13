from collections import namedtuple
from typing import Tuple, Callable

import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
import jax.scipy.stats as jstats
import numpy as np
from jax import lax
from jax.lax import cond

from ..utils import MVNormalParameters, make_matrices_parameters

SigmaPoints = namedtuple(
    'SigmaPoints', ['points', 'wm', 'wc']
)


def cubature_weights(n_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Computes the weights associated with the spherical cubature method.
    The number of sigma-points is 2 * n_dim

    Parameters
    ----------
    n_dim: int
        Dimensionality of the problem

    Returns
    -------
    wm: np.ndarray
        Weights means
    wc: np.ndarray
        Weights covariances
    xi: np.ndarray
        Orthogonal vectors
    """
    wm = np.ones(shape=(2 * n_dim,)) / (2 * n_dim)
    wc = wm
    xi = np.concatenate([np.eye(n_dim), -np.eye(n_dim)], axis=1) * np.sqrt(n_dim)

    return wm, wc, xi


def get_sigma_points(mv_normal_parameters: MVNormalParameters) -> SigmaPoints:
    """ Computes the sigma-points for a given mv normal distribution
    The number of sigma-points is 2*n_dim

    Parameters
    ----------
    mv_normal_parameters: MVNormalParameters
        Mean and Covariance of the distribution

    Returns
    -------
    out: SigmaPoints
        sigma points for the spherical cubature transform

    """
    mean = mv_normal_parameters.mean
    n_dim = mean.shape[0]

    wm, wc, xi = cubature_weights(n_dim)

    sigma_points = jnp.tile(mean.reshape(-1, 1), (1, wm.shape[0])) \
                   + jnp.dot(jnp.linalg.cholesky(mv_normal_parameters.cov), xi)

    return SigmaPoints(sigma_points, wm, wc)


def get_mv_normal_parameters(sigma_points: SigmaPoints, noise: np.ndarray) -> MVNormalParameters:
    """ Computes the MV Normal distribution parameters associated with the sigma points

    Parameters
    ----------
    sigma_points: SigmaPoints
        shape of sigma_points.points is (n_dim, 2*n_dim)
    noise: (n_dim, n_dim) array
        additive noise covariance matrix, if any
    Returns
    -------
    out: MVNormalParameters
        Mean and covariance of RV of dimension K computed from sigma-points
    """
    mean = jnp.dot(sigma_points.points, sigma_points.wm)
    diff = sigma_points.points - mean.reshape(-1, 1)
    cov = jnp.dot(sigma_points.wc.reshape(1, -1) * diff, diff.T) + noise
    return MVNormalParameters(mean, cov=cov)


def _transform(sigma_points: SigmaPoints,
               noise: SigmaPoints or MVNormalParameters,
               f: Callable) -> Tuple[SigmaPoints, MVNormalParameters]:
    """ Apply a function on the sigma points.

    Parameters
    ----------
    sigma_points: SigmaPoints
        sigma points to be transformed by f
    noise: SigmaPoints or MVNormalParameters
        if is_additive is True, then this is SigmaPoints, else it's assumed to be a MVNormalParameters
    f: Callable
        :math: `f(x)`

    Returns
    -------
    propagated_sigma_points: SigmaPoints
        sigma points after transformation
    propagated_mvn_parameters: MVNormalParameters
        approximate mnv from the propagated sigma points
    """

    propagated_points = f(sigma_points.points)
    propagated_sigma_points = SigmaPoints(propagated_points, sigma_points.wm, sigma_points.wc)

    propagated_mvn_parameters = get_mv_normal_parameters(propagated_sigma_points, noise.cov)

    return propagated_sigma_points, propagated_mvn_parameters


def predict(transition_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
            transition_covariance: jnp.ndarray,
            previous_sigma_points: SigmaPoints
            ) -> MVNormalParameters:
    """ Computes the cubature Kalman filter linearization of :math:`x_{t+1} = f(x_t, \mathcal{N}(0, \Sigma))`

    Parameters
    ----------
    transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
        transition function of the state space model
    transition_covariance: (D,D) array
        covariance :math:`\Sigma` of the noise fed to transition_function
    previous_sigma_points: SigmaPoints
        previous sigma points for the filter x
    Returns
    -------
    mvn_parameters: MVNormalParameters
        Propagated approximate Normal distribution
    """
    cov_shape = transition_covariance.shape[0]
    zero = jnp.zeros(cov_shape, dtype=transition_covariance.dtype)
    return _transform(previous_sigma_points,
                      MVNormalParameters(zero, transition_covariance),
                      transition_function)[1]


def update(observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
           observation_covariance: jnp.ndarray,
           predicted_points: SigmaPoints,
           predicted_parameters: MVNormalParameters,
           observation: jnp.ndarray) -> Tuple[float, MVNormalParameters]:
    """ Computes the extended kalman filter linearization of :math:`x_t \mid y_t`

    Parameters
    ----------
    observation_function: callable :math:`h(x_t,\epsilon_t)\mapsto y_t`
        observation function of the state space model
    observation_covariance: (K,K) array
        observation_error :math:`\Sigma` fed to observation_function
    predicted_points: SigmaPoints
        predicted sigma points of the filter :math:`x`
    predicted_parameters: MVNormalParameters
        predicted approximate mv normal parameters of the filter :math:`x`
    observation: (K) array
        Observation :math:`y`
    Returns
    -------
    loglikelihood: float
        Log-likelihood increment for observation
    updated_mvn_parameters: MVNormalParameters
        filtered state
    """

    cov_shape = observation_covariance.shape[0]
    zero = jnp.zeros(cov_shape, dtype=observation_covariance.dtype)
    obs_sigma_points, obs_mvn_parameters = _transform(predicted_points,
                                                      MVNormalParameters(zero,
                                                                         observation_covariance),
                                                      observation_function)

    loglikelihood = jstats.multivariate_normal.logpdf(observation,
                                                      obs_mvn_parameters.mean,
                                                      obs_mvn_parameters.cov)

    cross_covariance = jnp.dot(
        (predicted_points.points - predicted_parameters.mean.reshape(-1, 1)) * predicted_points.wm.reshape(1, -1),
        obs_sigma_points.points.T - obs_mvn_parameters.mean.reshape(1, -1))

    gain = jlinalg.solve(obs_mvn_parameters.cov, cross_covariance.T, sym_pos=True).T
    mean = predicted_parameters.mean + jnp.dot(gain, observation - obs_mvn_parameters.mean)
    cov = predicted_parameters.cov - jnp.dot(gain, cross_covariance.T)
    return loglikelihood, MVNormalParameters(mean, cov)


def filter_routine(initial_state: MVNormalParameters,
                   observations: jnp.ndarray,
                   transition_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   transition_covariances: jnp.ndarray,
                   observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   observation_covariances: jnp.ndarray) -> Tuple[float, MVNormalParameters]:
    """ Computes the predict-update routine of the cubature Kalman Filter equations and returns a series of filtered_states

    Parameters
    ----------
    initial_state: MVNormalParameters
        prior belief on the initial state distribution
    observations: (n, K) array
        array of n observations of dimension K
    transition_function: callable :math:`f(x_t, \epsilon_t) \mapsto x_{t-1}`
        transition function of the state space model
    transition_covariances: (D, D) or (1, D, D) or (n, D, D) array
        transition covariances for each time step, if passed only one, it is repeated n times
    observation_function: callable :math:`h(x_t,\epsilon_t)\mapsto y_t`
        observation function of the state space model
    observation_covariances: (K, K) or (1, K, K) or (n, K, K) array
        observation error covariances for each time step, if passed only one, it is repeated n times

    Returns
    -------
    loglikelihood: float
        Marginal loglikelihood of the observations given the parameters
    filtered_states: MVNormalParameters
        list of filtered states
    """

    n_observations = observations.shape[0]

    transition_covariances, observation_covariances = list(map(
        lambda z: make_matrices_parameters(z, n_observations),
        [transition_covariances,
         observation_covariances]))

    def body(carry, inputs):
        loglikelihood, state = carry
        observation, transition_covariance, observation_covariance = inputs

        sigma_points = get_sigma_points(state)

        loglikelihood_increment, updated_state = update(observation_function, observation_covariance, sigma_points,
                                                        state, observation)
        updated_sigma_points = get_sigma_points(updated_state)

        predicted_state = predict(transition_function, transition_covariance,
                                  updated_sigma_points)
        return (loglikelihood + loglikelihood_increment, predicted_state), updated_state

    (loglikelihood, _), filtered_states = lax.scan(body,
                                                   (0., initial_state),
                                                   [observations,
                                                    transition_covariances,
                                                    observation_covariances],
                                                   length=n_observations, unroll=10)

    return loglikelihood, filtered_states


def smooth(transition_function: Callable[[jnp.ndarray], jnp.ndarray],
           transition_covariance: jnp.array,
           filtered_state: MVNormalParameters,
           previous_smoothed: MVNormalParameters) -> MVNormalParameters:
    """
    One step cubature kalman smoother

    Parameters
    ----------
    transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
        transition function of the state space model
    transition_covariance: (D,D) array
        covariance :math:`\Sigma` of the noise fed to transition_function
    filtered_state: MVNormalParameters
        mean and cov computed by Kalman Filtering
    previous_smoothed: MVNormalParameters,
        smoothed state of the previous step
    Returns
    -------
    smoothed_state: MVNormalParameters
        smoothed state
    """

    filtered_sigma_points = get_sigma_points(filtered_state)

    cov_shape = transition_covariance.shape[0]
    zero = jnp.zeros(cov_shape, dtype=transition_covariance.dtype)

    predicted_points, predicted_mvn = _transform(filtered_sigma_points.points,
                                                 MVNormalParameters(zero, transition_covariance),
                                                 transition_function)

    cross_covariance = jnp.dot(
        (predicted_points.points - predicted_mvn.mean.reshape(-1, 1)) * predicted_points.wm.reshape(1, -1),
        filtered_sigma_points.points.T - filtered_state.mean.reshape(1, -1))

    gain = jlinalg.solve(predicted_mvn.cov, cross_covariance.T, sym_pos=True).T

    mean_diff = previous_smoothed.mean - filtered_state.mean
    cov_diff = previous_smoothed.cov - filtered_state.cov

    mean = filtered_state.mean + jnp.dot(gain, mean_diff)
    cov = filtered_state.cov + jnp.dot(gain, jnp.dot(cov_diff, gain.T))
    smoothed_state = MVNormalParameters(mean, cov)
    return smoothed_state


def smoother_routine(transition_function: Callable[[jnp.ndarray], jnp.ndarray],
                     transition_covariances: jnp.ndarray,
                     filtered_states: MVNormalParameters,
                     ) -> MVNormalParameters:
    """ Computes the cubature Rauch-Tung-Striebel smoother routine and returns a series of smoothed_states

    Parameters
    ----------
    filtered_states: MVNormalParameters
        Filtered states obtained from Kalman Filter
    transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
        transition function of the state space model
    transition_covariances: (D, D) or (1, D, D) or (n, D, D) array
        transition covariances for each time step, if passed only one, it is repeated n times

    Returns
    -------
    smoothed_states: MVNormalParameters
        list of smoothed states
    """
    n_observations = filtered_states.mean.shape[0]

    transition_covariances = make_matrices_parameters(transition_covariances, n_observations)

    def body(carry, list_inputs):
        j, state_ = carry

        def first_step(operand):
            state, _inputs, i = operand
            return (i + 1, state), state

        def otherwise(operand):
            state, inputs, i = operand
            filtered, transition_covariance = inputs
            smoothed_state = smooth(transition_function, transition_covariance, filtered, state)
            return (i + 1, smoothed_state), smoothed_state

        return cond(j > 0, otherwise, first_step, operand=(state_, list_inputs, j))

    last_state = MVNormalParameters(filtered_states.mean[-1], filtered_states.cov[-1])
    _, smoothed_states = lax.scan(body,
                                  (0, last_state),
                                  [filtered_states, transition_covariances],
                                  reverse=True)

    return smoothed_states
