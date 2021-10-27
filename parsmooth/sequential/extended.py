from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlag
from jax import lax, jacfwd
from jax.scipy.stats import multivariate_normal

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
    return MVNormalParameters(mean, 0.5 * (cov + cov.T))


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
    updated_state = MVNormalParameters(mean, 0.5 * (cov + cov.T))

    loglikelihood = multivariate_normal.logpdf(residual, jnp.zeros_like(residual), residual_covariance)
    return loglikelihood, updated_state


def filter_routine(initial_state: MVNormalParameters,
                   observations: jnp.ndarray,
                   transition_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   transition_covariances: jnp.ndarray,
                   observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   observation_covariances: jnp.ndarray,
                   linearization_points: jnp.ndarray = None,
                   propagate_first: bool = True) -> Tuple[float, MVNormalParameters]:
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
    propagate_first: bool, optional
        Is the first step a transition or an update? i.e. False if the initial time step has
        an associated observation. Default is True.

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

    def prop_first_body(carry, inputs):
        running_ell, state, prev_linearization_point = carry
        observation, transition_covariance, observation_covariance, linearization_point = inputs
        predicted_state = predict(transition_function, transition_covariance, state, prev_linearization_point)
        loglikelihood, updated_state = update(observation_function, observation_covariance, predicted_state,
                                              observation, linearization_point)

        return (running_ell + loglikelihood, updated_state, linearization_point), updated_state

    def update_first_body(carry, inputs):
        running_ell, state, _ = carry
        observation, transition_covariance, observation_covariance, linearization_point = inputs
        loglikelihood, updated_state = update(observation_function, observation_covariance, state,
                                              observation, linearization_point)
        predicted_state = predict(transition_function, transition_covariance, updated_state, linearization_point)
        return (running_ell + loglikelihood, predicted_state, linearization_point), updated_state

    body = prop_first_body if propagate_first else update_first_body

    if linearization_points is not None:
        initial_linearization_point = linearization_points[0] if linearization_points is not None else None
        linearization_points = linearization_points[1:] if propagate_first else linearization_points
    else:
        initial_linearization_point = linearization_points = None

    (ell, *_), filtered_states = lax.scan(body,
                                          (0., initial_state, initial_linearization_point),
                                          [observations,
                                           transition_covariances,
                                           observation_covariances,
                                           linearization_points],
                                          length=n_observations)

    if propagate_first:
        filtered_states = jax.tree_map(lambda y, z: jnp.concatenate([y[None, ...], z], 0), initial_state,
                                       filtered_states)

    return ell, filtered_states


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
                     linearization_points: jnp.ndarray = None,
                     propagate_first: bool = True,
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
    propagate_first: bool, optional
        Is the first step a transition or an update? i.e. False if the initial time step has
        an associated observation. Default is True.

    Returns
    -------
    smoothed_states: MVNormalParameters
        list of smoothed states
    """
    n_observations = filtered_states.mean.shape[0]
    if propagate_first:
        transition_covariances = make_matrices_parameters(transition_covariances, n_observations - 1)
    else:
        transition_covariances = make_matrices_parameters(transition_covariances, n_observations)
        transition_covariances = transition_covariances[:-1]

    def body(state, inputs):
        filtered, transition_covariance, linearization_point = inputs
        if linearization_point is None:
            linearization_point = filtered.mean
        smoothed_state = smooth(transition_function, transition_covariance, filtered, state, linearization_point)
        return smoothed_state, smoothed_state

    last_state = MVNormalParameters(filtered_states.mean[-1], filtered_states.cov[-1])
    filtered_states, linearization_points = jax.tree_map(lambda x: x[:-1],
                                                         [filtered_states, linearization_points])
    _, smoothed_states = lax.scan(body,
                                  last_state,
                                  [filtered_states, transition_covariances, linearization_points],
                                  reverse=True)

    smoothed_states = jax.tree_map(lambda y, z: jnp.concatenate([y, z[None, ...]], 0), smoothed_states,
                                   last_state)

    return smoothed_states


def iterated_smoother_routine(initial_state: MVNormalParameters,
                              observations: jnp.ndarray,
                              transition_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                              transition_covariances: jnp.ndarray,
                              observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                              observation_covariances: jnp.ndarray,
                              initial_linearization_points: jnp.ndarray = None,
                              n_iter: int = 100,
                              propagate_first: bool = True):
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
    initial_linearization_points: jnp.ndarray , optional
        points at which to compute the jacobians durning the first pass.
    n_iter: int
        number of times the filter-smoother routine is computed
    propagate_first: bool, optional
        Is the first step a transition or an update? i.e. False if the initial time step has
        an associated observation. Default is True.

    Returns
    -------
    iterated_smoothed_trajectories: MVNormalParameters
        The result of the smoothing routine

    """
    n_observations = observations.shape[0]

    transition_covariances, observation_covariances = list(map(
        lambda z: make_matrices_parameters(z, n_observations),
        [transition_covariances,
         observation_covariances]))

    def body(curr_smoother, _):
        if curr_smoother is not None:
            linearization_points = curr_smoother.mean if isinstance(curr_smoother,
                                                                    MVNormalParameters) else curr_smoother
        else:
            linearization_points = None

        _, filtered_states = filter_routine(initial_state, observations, transition_function, transition_covariances,
                                            observation_function, observation_covariances, linearization_points,
                                            propagate_first)
        return smoother_routine(transition_function, transition_covariances, filtered_states,
                                linearization_points, propagate_first), None

    if initial_linearization_points is None:
        initial_linearization_points = body(None, None)[0]

    iterated_smoothed_trajectories, _ = lax.scan(body, initial_linearization_points, jnp.arange(n_iter))
    return iterated_smoothed_trajectories
