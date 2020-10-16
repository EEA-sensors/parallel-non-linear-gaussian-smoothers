from typing import Tuple, Callable

import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax
from jax.lax import cond

from ..cubature_common import SigmaPoints, get_sigma_points
from ..utils import MVNormalParameters, make_matrices_parameters


def _mean(points):
    return jnp.dot(points.wm, points.points)


def _covariance(points_1, mean_1, points_2, mean_2):
    one = (points_1.points - mean_1.reshape(1, -1)).T * points_1.wc.reshape(1, -1)
    two = points_2.points - mean_2.reshape(1, -1)
    return jnp.dot(one, two)


def predict(transition_function: Callable,
            transition_covariance: jnp.ndarray,
            previous_state: MVNormalParameters,
            linearization_state: MVNormalParameters,
            return_linearized_transition: bool = False
            ) -> MVNormalParameters:
    """ Computes the cubature Kalman filter linearization of :math:`x_{t+1} = f(x_t, \mathcal{N}(0, \Sigma))`

    Parameters
    ----------
    transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
        transition function of the state space model
    transition_covariance: (D,D) array
        covariance :math:`\Sigma` of the noise fed to transition_function
    previous_state: MVNormalParameters
        previous state for the filter x
    linearization_state: MVNormalParameters
        state for the linearization of the prediction
    return_linearized_transition: bool, optional
        Returns the linearized transition matrix A

    Returns
    -------
    mvn_parameters: MVNormalParameters
        Propagated approximate Normal distribution

    A: array_like
        returned if return_linearized_transition is True
    """
    if linearization_state is None:
        linearization_state = previous_state

    sigma_points = get_sigma_points(linearization_state)
    propagated_points = transition_function(sigma_points.points)
    propagated_sigma_points = SigmaPoints(propagated_points, sigma_points.wm, sigma_points.wc)

    propagated_sigma_points_mean = _mean(propagated_sigma_points)
    propagated_sigma_points_cov = _covariance(propagated_sigma_points, propagated_sigma_points_mean,
                                              propagated_sigma_points, propagated_sigma_points_mean)
    cross_covariance = _covariance(sigma_points, linearization_state.mean, propagated_sigma_points,
                                   propagated_sigma_points_mean)

    A = jlinalg.solve(linearization_state.cov, cross_covariance, sym_pos=True).T  # Linearized transition function
    b = propagated_sigma_points_mean - jnp.dot(A, linearization_state.mean)  # Linearized offset

    mean = A @ previous_state.mean + b
    cov = transition_covariance + propagated_sigma_points_cov + A @ (previous_state.cov - linearization_state.cov) @ A.T
    if return_linearized_transition:
        return MVNormalParameters(mean, cov), A
    return MVNormalParameters(mean, cov)


def update(observation_function: Callable,
           observation_covariance: jnp.ndarray,
           predicted_state: MVNormalParameters,
           observation: jnp.ndarray,
           linearization_state: MVNormalParameters) -> MVNormalParameters:
    """ Computes the extended kalman filter linearization of :math:`x_t \mid y_t`

    Parameters
    ----------
    observation_function: callable :math:`h(x_t,\epsilon_t)\mapsto y_t`
        observation function of the state space model
    observation_covariance: (K,K) array
        observation_error :math:`\Sigma` fed to observation_function
    predicted_state: MVNormalParameters
        predicted approximate mv normal parameters of the filter :math:`x`
    observation: (K) array
        Observation :math:`y`
    linearization_state: MVNormalParameters
        state for the linearization of the update

    Returns
    -------
    updated_mvn_parameters: MVNormalParameters
        filtered state
    """
    if linearization_state is None:
        linearization_state = predicted_state
    sigma_points = get_sigma_points(linearization_state)
    obs_points = observation_function(sigma_points.points)
    obs_sigma_points = SigmaPoints(obs_points, sigma_points.wm, sigma_points.wc)

    obs_sigma_points_mean = _mean(obs_sigma_points)
    obs_sigma_points_cov = _covariance(obs_sigma_points, obs_sigma_points_mean,
                                       obs_sigma_points, obs_sigma_points_mean)
    cross_covariance = _covariance(sigma_points, linearization_state.mean, obs_sigma_points,
                                   obs_sigma_points_mean)

    H = jlinalg.solve(linearization_state.cov, cross_covariance, sym_pos=True).T  # linearized observation function
    d = obs_sigma_points_mean - jnp.dot(H, linearization_state.mean)  # linearized observation offset

    residual_cov = H @ (predicted_state.cov - linearization_state.cov) @ H.T + \
                   observation_covariance + obs_sigma_points_cov

    gain = jlinalg.solve(residual_cov, H @ predicted_state.cov).T

    predicted_observation = H @ predicted_state.mean + d

    mean = predicted_state.mean + gain @ (observation - predicted_observation)
    cov = predicted_state.cov - gain @ residual_cov @ gain.T

    return MVNormalParameters(mean, cov)


def filter_routine(initial_state: MVNormalParameters,
                   observations: jnp.ndarray,
                   transition_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   transition_covariances: jnp.ndarray,
                   observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   observation_covariances: jnp.ndarray,
                   linearization_states: MVNormalParameters = None) -> Tuple[float, MVNormalParameters]:
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
    linearization_states: MVNormalParameters, optional
        states for the cubature linearization

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
        state, previous_linearization_state = carry
        observation, transition_covariance, observation_covariance, linearization_state = inputs
        predicted_state = predict(transition_function, transition_covariance, state, previous_linearization_state)
        updated_state = update(observation_function, observation_covariance, predicted_state, observation,
                               linearization_state)

        return (updated_state, linearization_state), updated_state

    initial_linearization_state = initial_state if linearization_states is not None else None
    _, filtered_states = lax.scan(body,
                                  (initial_state, initial_linearization_state),
                                  [observations,
                                   transition_covariances,
                                   observation_covariances,
                                   linearization_states],
                                  length=n_observations)

    return filtered_states


def smooth(transition_function: Callable[[jnp.ndarray], jnp.ndarray],
           transition_covariance: jnp.array,
           filtered_state: MVNormalParameters,
           previous_smoothed: MVNormalParameters,
           linearization_state: MVNormalParameters) -> MVNormalParameters:
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
    previous_smoothed: MVNormalParameters
        smoothed state of the previous step
    linearization_state: MVNormalParameters
        state for the cubature linearization

    Returns
    -------
    smoothed_state: MVNormalParameters
        smoothed state
    """
    predicted_state, A = predict(transition_function, transition_covariance, filtered_state, linearization_state, True)
    smoothing_gain = jnp.linalg.solve(predicted_state.cov, A @ filtered_state.cov).T
    mean = filtered_state.mean + smoothing_gain @ (previous_smoothed.mean - predicted_state.mean)
    cov = filtered_state.cov + smoothing_gain @ (previous_smoothed.cov - predicted_state.cov) @ smoothing_gain.T
    return MVNormalParameters(mean, cov)


def smoother_routine(transition_function: Callable[[jnp.ndarray], jnp.ndarray],
                     transition_covariances: jnp.ndarray,
                     filtered_states: MVNormalParameters,
                     linearization_states: MVNormalParameters = None
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
    linearization_states: MVNormalParameters, optional
        states for the cubature linearization

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
            filtered, transition_covariance, linearization_state = inputs
            smoothed_state = smooth(transition_function, transition_covariance, filtered, state, linearization_state)
            return (i + 1, smoothed_state), smoothed_state

        return cond(j > 0, otherwise, first_step, operand=(state_, list_inputs, j))

    last_state = MVNormalParameters(filtered_states.mean[-1], filtered_states.cov[-1])
    _, smoothed_states = lax.scan(body,
                                  (0, last_state),
                                  [filtered_states, transition_covariances, linearization_states],
                                  reverse=True)

    return smoothed_states
