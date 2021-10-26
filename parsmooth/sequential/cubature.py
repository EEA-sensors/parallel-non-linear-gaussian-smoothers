from typing import Tuple, Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax
from jax.scipy.stats import multivariate_normal

from ..cubature_common import SigmaPoints, get_sigma_points, get_mv_normal_parameters, covariance_sigma_points
from ..utils import MVNormalParameters, make_matrices_parameters


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

    F: array_like
        returned if return_linearized_transition is True
    """
    if linearization_state is None:
        linearization_state = previous_state

    sigma_points = get_sigma_points(linearization_state)
    propagated_points = transition_function(sigma_points.points)
    propagated_sigma_points = SigmaPoints(propagated_points,
                                          sigma_points.wm,
                                          sigma_points.wc)

    propagated_state = get_mv_normal_parameters(propagated_sigma_points)
    cross_covariance = covariance_sigma_points(sigma_points, linearization_state.mean, propagated_sigma_points,
                                               propagated_state.mean)

    F = jlinalg.solve(linearization_state.cov, cross_covariance, sym_pos=True).T  # Linearized transition function
    b = propagated_state.mean - jnp.dot(F, linearization_state.mean)  # Linearized offset

    mean = F @ previous_state.mean + b
    cov = transition_covariance + propagated_state.cov + F @ (previous_state.cov - linearization_state.cov) @ F.T
    if return_linearized_transition:
        return MVNormalParameters(mean, cov), F
    return MVNormalParameters(mean, 0.5 * (cov + cov.T))


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

    obs_state = get_mv_normal_parameters(obs_sigma_points)
    cross_covariance = covariance_sigma_points(sigma_points, linearization_state.mean, obs_sigma_points,
                                               obs_state.mean)

    H = jlinalg.solve(linearization_state.cov, cross_covariance, sym_pos=True).T  # linearized observation function

    d = obs_state.mean - jnp.dot(H, linearization_state.mean)  # linearized observation offset

    residual_cov = H @ (predicted_state.cov - linearization_state.cov) @ H.T + \
                   observation_covariance + obs_state.cov

    gain = jlinalg.solve(residual_cov, H @ predicted_state.cov).T

    predicted_observation = H @ predicted_state.mean + d

    residual = observation - predicted_observation
    mean = predicted_state.mean + gain @ residual
    cov = predicted_state.cov - gain @ residual_cov @ gain.T
    loglikelihood = multivariate_normal.logpdf(residual, jnp.zeros_like(residual), residual_cov)

    return loglikelihood, MVNormalParameters(mean, 0.5 * (cov + cov.T))


def filter_routine(initial_state: MVNormalParameters,
                   observations: jnp.ndarray,
                   transition_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   transition_covariances: jnp.ndarray,
                   observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   observation_covariances: jnp.ndarray,
                   linearization_states: MVNormalParameters = None,
                   propagate_first: bool = True) -> Tuple[float, MVNormalParameters]:
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
        running_ell, state, prev_linearization_state = carry
        observation, transition_covariance, observation_covariance, linearization_state = inputs
        predicted_state = predict(transition_function, transition_covariance, state, prev_linearization_state)
        loglikelihood, updated_state = update(observation_function, observation_covariance, predicted_state,
                                              observation, linearization_state)

        return (running_ell + loglikelihood, updated_state, linearization_state), updated_state

    def update_first_body(carry, inputs):
        running_ell, state, _ = carry
        observation, transition_covariance, observation_covariance, linearization_point = inputs
        loglikelihood, updated_state = update(observation_function, observation_covariance, state,
                                              observation, linearization_point)
        predicted_state = predict(transition_function, transition_covariance, updated_state, linearization_point)
        return (running_ell + loglikelihood, predicted_state, linearization_point), updated_state

    body = prop_first_body if propagate_first else update_first_body

    initial_linearization_state = jax.tree_map(lambda z: z[0], linearization_states)
    if propagate_first:
        linearization_states = jax.tree_map(lambda z: z[1:], linearization_states)

    (ell, *_), filtered_states = lax.scan(body,
                                          (0., initial_state, initial_linearization_state),
                                          [observations,
                                           transition_covariances,
                                           observation_covariances,
                                           linearization_states],
                                          length=n_observations)

    if propagate_first:
        filtered_states = jax.tree_map(lambda y, z: jnp.concatenate([y[None, ...], z], 0), initial_state,
                                       filtered_states)
    return ell, filtered_states


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
    predicted_state, F = predict(transition_function, transition_covariance, filtered_state, linearization_state, True)
    smoothing_gain = jnp.linalg.solve(predicted_state.cov, F @ filtered_state.cov).T
    mean = filtered_state.mean + smoothing_gain @ (previous_smoothed.mean - predicted_state.mean)
    cov = filtered_state.cov + smoothing_gain @ (previous_smoothed.cov - predicted_state.cov) @ smoothing_gain.T
    return MVNormalParameters(mean, 0.5 * (cov + cov.T))


def smoother_routine(transition_function: Callable[[jnp.ndarray], jnp.ndarray],
                     transition_covariances: jnp.ndarray,
                     filtered_states: MVNormalParameters,
                     linearization_states: MVNormalParameters = None,
                     propagate_first: bool = True
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
        filtered, transition_covariance, linearization_state = inputs
        if linearization_state is None:
            linearization_state = filtered
        smoothed_state = smooth(transition_function, transition_covariance, filtered, state, linearization_state)
        return smoothed_state, smoothed_state

    last_state = MVNormalParameters(filtered_states.mean[-1], filtered_states.cov[-1])
    filtered_states, linearization_states = jax.tree_map(lambda x: x[:-1],
                                                         [filtered_states, linearization_states])
    _, smoothed_states = lax.scan(body,
                                  last_state,
                                  [filtered_states, transition_covariances, linearization_states],
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
                              initial_linearization_states: jnp.ndarray = None,
                              n_iter: int = 100,
                              propagate_first: bool = False) -> MVNormalParameters:
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
    initial_linearization_states: MVNormalParameters, optional
        states for linearization of the first pass.
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

    def body(linearization_points, _):
        _, filtered_states = filter_routine(initial_state, observations, transition_function, transition_covariances,
                                            observation_function, observation_covariances, linearization_points,
                                            propagate_first)
        return smoother_routine(transition_function, transition_covariances, filtered_states,
                                linearization_points, propagate_first), None

    if initial_linearization_states is None:
        initial_linearization_states = body(None, None)[0]

    iterated_smoothed_trajectories, _ = lax.scan(body, initial_linearization_states, jnp.arange(n_iter))
    return iterated_smoothed_trajectories
