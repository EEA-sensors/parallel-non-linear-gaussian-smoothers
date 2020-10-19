from typing import Callable, Tuple

import jax.numpy as jnp
import jax.scipy.linalg as jlag
from jax import lax, jacfwd
from jax.lax import cond

from ..utils import MVNormalParameters, make_matrices_parameters

__all__ = ["filter_routine", "smoother_routine"]


def predict(transition_function: Callable[[jnp.ndarray], jnp.ndarray],
            transition_covariance: jnp.ndarray,
            prior: MVNormalParameters,
            linearization_point: jnp.ndarray) -> MVNormalParameters:
    """ Computes the extended kalman filter linearization of :math:`x_{t+1} = f(x_t, \mathcal{N}(0, \Sigma))`

    Parameters
    ----------
    transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
        transition function of the state space model
    transition_covariance: (D,D) array
        covariance :math:`\Sigma` of the noise fed to transition_function
    prior: MVNormalParameters
        prior state of the filter x
    linearization_point: jnp.ndarray
        Where to compute the Jacobian

    Returns
    -------
    out: MVNormalParameters
        Predicted state
    """
    if linearization_point is None:
        linearization_point = prior.mean
    jac_x = jacfwd(transition_function, 0)(linearization_point)
    cov = jnp.dot(jac_x, jnp.dot(prior.cov, jac_x.T)) + transition_covariance
    mean = transition_function(linearization_point)
    mean = mean + jnp.dot(jac_x, prior.mean - linearization_point)
    return MVNormalParameters(mean, cov)


def update(observation_function: Callable[[jnp.ndarray], jnp.ndarray],
           observation_covariance: jnp.ndarray,
           predicted: MVNormalParameters,
           observation: jnp.ndarray,
           linearization_point: jnp.ndarray) -> Tuple[float, MVNormalParameters]:
    """ Computes the extended kalman filter linearization of :math:`x_t \mid y_t`

    Parameters
    ----------
    observation_function: callable :math:`h(x_t,\epsilon_t)\mapsto y_t`
        observation function of the state space model
    observation_covariance: (K,K) array
        observation_error :math:`\Sigma` fed to observation_function
    predicted: MVNormalParameters
        predicted state of the filter :math:`x`
    observation: (K) array
        Observation :math:`y`
    linearization_point: jnp.ndarray
        Where to compute the Jacobian

    Returns
    -------
    loglikelihood: float
        Log-likelihood increment for observation
    updated_state: MVNormalParameters
        filtered state
    """
    if linearization_point is None:
        linearization_point = predicted.mean
    jac_x = jacfwd(observation_function, 0)(linearization_point)

    obs_mean = observation_function(linearization_point) + jnp.dot(jac_x, predicted.mean - linearization_point)

    residual = observation - obs_mean
    residual_covariance = jnp.dot(jac_x, jnp.dot(predicted.cov, jac_x.T))
    residual_covariance = residual_covariance + observation_covariance

    gain = jnp.dot(predicted.cov, jlag.solve(residual_covariance, jac_x, sym_pos=True).T)

    mean = predicted.mean + jnp.dot(gain, residual)
    cov = predicted.cov - jnp.dot(gain, jnp.dot(residual_covariance, gain.T))
    updated_state = MVNormalParameters(mean, cov)
    return updated_state


def filter_routine(initial_state: MVNormalParameters,
                   observations: jnp.ndarray,
                   transition_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   transition_covariances: jnp.ndarray,
                   observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   observation_covariances: jnp.ndarray,
                   linearization_points: jnp.ndarray = None) -> Tuple[float, MVNormalParameters]:
    """ Computes the predict-update routine of the Kalman Filter equations and returns a series of filtered_states

    Parameters
    ----------
    initial_state: MVNormalParameters
        prior belief on the initial state distribution
    observations: (n, K) array
        array of n observations of dimension K
    transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
        transition function of the state space model
    transition_covariances: (D, D) or (1, D, D) or (n, D, D) array
        transition covariances for each time step, if passed only one, it is repeated n times
    observation_function: callable :math:`h(x_t,\epsilon_t)\mapsto y_t`
        observation function of the state space model
    observation_covariances: (K, K) or (1, K, K) or (n, K, K) array
        observation error covariances for each time step, if passed only one, it is repeated n times
    linearization_points: (n, D) array, optional
        points at which to compute the jacobians.

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
        state, prev_linearization_point = carry
        observation, transition_covariance, observation_covariance, linearization_point = inputs
        predicted_state = predict(transition_function, transition_covariance, state, prev_linearization_point)
        updated_state = update(observation_function, observation_covariance, predicted_state,
                               observation, linearization_point)
        return (updated_state, linearization_point), updated_state

    initial_linearization_point = initial_state.mean if linearization_points is not None else None
    _, filtered_states = lax.scan(body,
                                  (initial_state, initial_linearization_point),
                                  [observations,
                                   transition_covariances,
                                   observation_covariances,
                                   linearization_points],
                                  length=n_observations)

    return filtered_states


def smooth(transition_function: Callable[[jnp.ndarray], jnp.ndarray],
           transition_covariance: jnp.array,
           filtered_state: MVNormalParameters,
           previous_smoothed: MVNormalParameters,
           linearization_point: jnp.ndarray) -> MVNormalParameters:
    """
    One step extended kalman smoother

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
    linearization_point: jnp.ndarray
        Where to compute the Jacobian

    Returns
    -------
    smoothed_state: MVNormalParameters
        smoothed state
    """

    jac_x = jacfwd(transition_function, 0)(linearization_point)

    mean = transition_function(linearization_point) + jnp.dot(jac_x, filtered_state.mean - linearization_point)
    mean_diff = previous_smoothed.mean - mean

    cov = jnp.dot(jac_x, jnp.dot(filtered_state.cov, jac_x.T)) + transition_covariance
    cov_diff = previous_smoothed.cov - cov

    gain = jnp.dot(filtered_state.cov, jlag.solve(cov, jac_x, sym_pos=True).T)

    mean = filtered_state.mean + jnp.dot(gain, mean_diff)
    cov = filtered_state.cov + jnp.dot(gain, jnp.dot(cov_diff, gain.T))
    return MVNormalParameters(mean, cov)


def smoother_routine(transition_function: Callable[[jnp.ndarray], jnp.ndarray],
                     transition_covariances: jnp.ndarray,
                     filtered_states: MVNormalParameters,
                     linearization_points: jnp.ndarray = None
                     ) -> MVNormalParameters:
    """ Computes the extended Rauch-Tung-Striebel (a.k.a extended Kalman) smoother routine and returns a series of smoothed_states

    Parameters
    ----------
    filtered_states: MVNormalParameters
        Filtered states obtained from Kalman Filter
    transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
        transition function of the state space model
    transition_covariances: (D, D) or (1, D, D) or (n, D, D) array
        transition covariances for each time step, if passed only one, it is repeated n times
    linearization_points: (n, D) array, optional
        points at which to compute the jacobians.

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
            filtered, transition_covariance, linearization_point = inputs
            if linearization_point is None:
                linearization_point = filtered.mean
            smoothed_state = smooth(transition_function, transition_covariance, filtered, state, linearization_point)
            return (i + 1, smoothed_state), smoothed_state

        return cond(j > 0, otherwise, first_step, operand=(state_, list_inputs, j))

    last_state = MVNormalParameters(filtered_states.mean[-1], filtered_states.cov[-1])
    _, smoothed_states = lax.scan(body,
                                  (0, last_state),
                                  [filtered_states, transition_covariances, linearization_points],
                                  reverse=True)

    return smoothed_states


def iterated_smoother_routine(initial_state: MVNormalParameters,
                              observations: jnp.ndarray,
                              transition_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                              transition_covariances: jnp.ndarray,
                              observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                              observation_covariances: jnp.ndarray,
                              initial_linearization_points: jnp.ndarray = None,
                              n_iter: int = 100):
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
    transition_covariances: (D, D) or (1, D, D) or (n, D, D) array
        transition covariances for each time step, if passed only one, it is repeated n times
    observation_function: callable :math:`h(x_t,\epsilon_t)\mapsto y_t`
        observation function of the state space model
    observation_covariances: (K, K) or (1, K, K) or (n, K, K) array
        observation error covariances for each time step, if passed only one, it is repeated n times
    initial_linearization_points: (N, D) array , optional
        points at which to compute the jacobians durning the first pass.
    n_iter: int
        number of times the filter-smoother routine is computed

    Returns
    -------
    iterated_smoothed_trajectories: MVNormalParameters
        The result of the smoothing routine

    """
    n_observations = observations.shape[0]
    x_dim = initial_state.mean.shape[0]
    dtype = initial_state.mean.dtype

    transition_covariances, observation_covariances = list(map(
        lambda z: make_matrices_parameters(z, n_observations),
        [transition_covariances,
         observation_covariances]))

    def body(linearization_points, _):
        if linearization_points is not None:
            linearization_points = linearization_points.mean
        filtered_states = filter_routine(initial_state, observations, transition_function, transition_covariances,
                                         observation_function, observation_covariances, linearization_points)
        return smoother_routine(transition_function, transition_covariances, filtered_states,
                                linearization_points), None

    if initial_linearization_points is None:
        initial_linearization_states = body(None, None)[0]
    else:
        initial_linearization_states = MVNormalParameters(initial_linearization_points, jnp.empty((n_observations,
                                                                                                   x_dim,
                                                                                                   x_dim), dtype))

    iterated_smoothed_trajectories, _ = lax.scan(body, initial_linearization_states, jnp.arange(n_iter))
    return iterated_smoothed_trajectories
