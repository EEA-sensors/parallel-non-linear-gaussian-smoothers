from typing import Tuple, Callable

import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax
from jax.lax import cond

from ..cubature_common import transform, SigmaPoints, get_sigma_points
from ..utils import MVNormalParameters, make_matrices_parameters


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
    return transform(previous_sigma_points,
                     MVNormalParameters(zero, transition_covariance),
                     transition_function)[1]


def update(observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
           observation_covariance: jnp.ndarray,
           predicted_points: SigmaPoints,
           predicted_parameters: MVNormalParameters,
           observation: jnp.ndarray) -> MVNormalParameters:
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
    updated_mvn_parameters: MVNormalParameters
        filtered state
    """

    cov_shape = observation_covariance.shape[0]
    zero = jnp.zeros(cov_shape, dtype=observation_covariance.dtype)
    obs_sigma_points, obs_mvn_parameters = transform(predicted_points,
                                                     MVNormalParameters(zero,
                                                                        observation_covariance),
                                                     observation_function)

    cross_covariance = jnp.dot(
        (predicted_points.points - predicted_parameters.mean.reshape(1, -1)).T * predicted_points.wm.reshape(1, -1),
        obs_sigma_points.points - obs_mvn_parameters.mean.reshape(1, -1))

    gain = jlinalg.solve(obs_mvn_parameters.cov, cross_covariance.T, sym_pos=True).T
    mean = predicted_parameters.mean + jnp.dot(gain, observation - obs_mvn_parameters.mean)
    cov = predicted_parameters.cov - jnp.dot(gain, cross_covariance.T)
    return MVNormalParameters(mean, cov)


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

    def body(state, inputs):
        observation, transition_covariance, observation_covariance = inputs

        sigma_points = get_sigma_points(state)
        predicted_state = predict(transition_function, transition_covariance,
                                  sigma_points)

        predicted_state_sigma_points = get_sigma_points(predicted_state)
        updated_state = update(observation_function, observation_covariance, predicted_state_sigma_points,
                               predicted_state, observation)

        return updated_state, updated_state

    _, filtered_states = lax.scan(body,
                                  initial_state,
                                  [observations,
                                   transition_covariances,
                                   observation_covariances],
                                  length=n_observations)

    return filtered_states


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

    predicted_points, predicted_mvn = _transform(filtered_sigma_points,
                                                 MVNormalParameters(zero, transition_covariance),
                                                 transition_function)

    cross_covariance = jnp.dot(
        (predicted_points.points - predicted_mvn.mean.reshape(1, -1)).T * predicted_points.wm.reshape(1, -1),
        filtered_sigma_points.points - filtered_state.mean.reshape(1, -1))

    gain = jlinalg.solve(predicted_mvn.cov, cross_covariance.T, sym_pos=True).T

    mean_diff = previous_smoothed.mean - predicted_mvn.mean
    cov_diff = previous_smoothed.cov - predicted_mvn.cov

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
