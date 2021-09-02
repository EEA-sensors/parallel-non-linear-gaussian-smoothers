from typing import Callable, Tuple

import jax.numpy as jnp
import jax.scipy.linalg as jlag
from jax import lax
from jax.lax import cond

from parsmooth.utils import MVNParams, make_matrices_parameters

__all__ = ["filter_routine", "smoother_routine"]


def _linearize(linearization_state, prev_state, chol, fun, method):
    if linearization_state is None:
        linearization_state = prev_state
    else:
        if linearization_state.cov is None:
            linearization_state = MVNParams(linearization_state.mean, None, prev_state.chol)
    F, c, chol_Q = method(fun, linearization_state, chol, sqrt=True)
    return F, c, Q + F @ (prev_state.cov - linearization_state.cov) @ F.T


def predict(transition_function: Callable or Tuple[Callable, 2],
            transition_chol: jnp.ndarray or None,
            prior: MVNParams,
            linearization_state: MVNParams or None,
            linearization_method: Callable) -> MVNParams:
    r""" Computes the extended kalman filter linearization of :math:`x_{t+1} = f(x_t, \mathcal{N}(0, \Sigma))`

    Parameters
    ----------
    transition_function: callable
        transition function of the state space model
    transition_chol: (D,D) array
        SQRT covariance :math:`\Sigma` of the noise fed to transition_function
    prior: MVNParams
        prior state of the filter x
    linearization_state: MVNParams
        Where to compute the Jacobian
    linearization_method: callable
        The linearization method
    Returns
    -------
    out: MVNParams
        Predicted state
    """

    F, c, chol_Q = _linearize(linearization_state, prior, transition_chol, transition_function,
                              linearization_method)

    mean = c + jnp.dot(F, prior.mean)

    return MVNParams(mean, None, chol_Q)


def update(observation_function: Callable or Tuple[Callable, 2],
           observation_covariance: jnp.ndarray or None,
           predicted: MVNParams,
           observation: jnp.ndarray,
           linearization_state: MVNParams or None,
           linearization_method: Callable) -> Tuple[float, MVNParams]:
    r""" Computes the extended kalman filter linearization of :math:`x_t \mid y_t`

    Parameters
    ----------
    observation_function: callable :math:`h(x_t,\epsilon_t)\mapsto y_t`
        observation function of the state space model
    observation_covariance: (K,K) array
        observation_error :math:`\Sigma` fed to observation_function
    predicted: MVNParams
        predicted state of the filter :math:`x`
    observation: (K) array
        Observation :math:`y`
    linearization_state: MVNParams
        Where to compute the Jacobian
    linearization_method: callable
        The linearization method

    Returns
    -------
    loglikelihood: float
        Log-likelihood increment for observation
    updated_state: MVNParams
        filtered state
    """

    H, d, R = _linearize(linearization_state, predicted, observation_covariance, observation_function,
                         linearization_method)
    obs_mean = d + jnp.dot(H, predicted.mean)

    residual = observation - obs_mean
    gain = jnp.dot(predicted.cov, jlag.solve(R, H, sym_pos=True).T)

    mean = predicted.mean + jnp.dot(gain, residual)
    cov = predicted.cov - jnp.dot(gain, jnp.dot(R, gain.T))
    updated_state = MVNParams(mean, cov)
    return updated_state


def filter_routine(initial_state: MVNParams,
                   observations: jnp.ndarray,
                   transition_function: Callable or Tuple[Callable, 2],
                   transition_covariances: jnp.ndarray,
                   observation_function: Callable or Tuple[Callable, 2],
                   observation_covariances: jnp.ndarray,
                   linearization_method: Callable,
                   linearization_points: MVNParams = None) -> MVNParams:
    r""" Computes the linearized predict-update routine of the Kalman Filter equations and returns a series of filtered_states

    Parameters
    ----------
    initial_state: MVNParams
        prior belief on the initial state distribution
    observations: (n, K) ndarray
        array of n observations of dimension K
    transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
        transition function of the state space model
    transition_covariances: (D, D) or (1, D, D) or (n, D, D) array
        transition covariances for each time step, if passed only one, it is repeated n times
    observation_function: callable :math:`h(x_t,\epsilon_t) \mapsto y_t`
        observation function of the state space model
    observation_covariances: (K, K) or (1, K, K) or (n, K, K) array
        observation error covariances for each time step, if passed only one, it is repeated n times
    linearization_method: callable
        The linearization method
    linearization_points: (n, D) MVNormalParameters, optional
        points at which to compute the Jacobians.

    Returns
    -------
    filtered_states: MVNParams
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
        predicted_state = predict(transition_function, transition_covariance, state, prev_linearization_point,
                                  linearization_method)
        updated_state = update(observation_function, observation_covariance, predicted_state,
                               observation, linearization_point, linearization_method)
        return (updated_state, linearization_point), updated_state

    if linearization_points is not None:
        initial_linearization_point = MVNParams(linearization_points.mean[0], linearization_points.cov[0])
    else:
        initial_linearization_point = None

    _, filtered_states = lax.scan(body,
                                  (initial_state, initial_linearization_point),
                                  (observations,
                                   transition_covariances,
                                   observation_covariances,
                                   linearization_points),
                                  length=n_observations)

    return MVNParams(filtered_states[0], filtered_states[1])


def smooth(transition_function: Callable or Tuple[Callable, 2],
           transition_covariances: jnp.ndarray,
           filtered_state: MVNParams,
           previous_smoothed: MVNParams,
           linearization_method: Callable,
           linearization_state: MVNParams or None) -> MVNParams:
    r"""One step extended kalman smoother

        Parameters
        ----------
        transition_function: callable
             transition function of the state space model
        transition_covariances: (D,D) array
            covariance :math:`\Sigma` of the noise fed to transition_function
        filtered_state: MVNParams
            mean and cov computed by Kalman Filtering
        previous_smoothed: MVNormalParameters,
            smoothed state of the previous step
        linearization_method: Callable
            The linearization method
        linearization_state: MVNParams
            Where to compute the Jacobian

        Returns
        -------
        smoothed_state: MVNParams
            smoothed state
        """
    F, c, Q = _linearize(linearization_state, filtered_state, transition_covariances, transition_function,
                         linearization_method)
    mean_diff = previous_smoothed.mean - (c + jnp.dot(F, filtered_state.mean))
    cov_diff = previous_smoothed.cov - Q

    gain = jnp.dot(filtered_state.cov, jlag.solve(Q, F, sym_pos=True).T)

    mean = filtered_state.mean + jnp.dot(gain, mean_diff)
    cov = filtered_state.cov + jnp.dot(gain, jnp.dot(cov_diff, gain.T))

    return MVNParams(mean, cov)


def smoother_routine(transition_function: Callable[[jnp.ndarray], jnp.ndarray],
                     transition_covariances: jnp.ndarray,
                     filtered_states: MVNParams,
                     linearization_method: Callable,
                     linearization_points: MVNParams = None) -> MVNParams:
    """ Computes the extended Rauch-Tung-Striebel (a.k.a extended Kalman) smoother routine and returns a series of smoothed_states

    Parameters
    ----------
    filtered_states: MVNParams
        Filtered states obtained from Kalman Filter
    transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
        transition function of the state space model
    transition_covariances: (D, D) or (1, D, D) or (n, D, D) array
        transition covariances for each time step, if passed only one, it is repeated n times
    linearization_method: Callable
        The linearization method
    linearization_points: (n, D) MVNormalParameters, optional
        points at which to compute the jacobians.

    Returns
    -------
    smoothed_states: MVNParams
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
            filtered, transition_covariance, linearization_states = inputs
            smoothed_state = smooth(transition_function, transition_covariance, filtered, state,
                                    linearization_method, linearization_states)
            return (i + 1, smoothed_state), smoothed_state

        return cond(j > 0, otherwise, first_step, operand=(state_, list_inputs, j))

    last_state = MVNParams(filtered_states.mean[-1], filtered_states.cov[-1])
    _, smoothed_states = lax.scan(body,
                                  (0, last_state),
                                  [filtered_states, transition_covariances, linearization_points],
                                  reverse=True)

    return MVNParams(smoothed_states[0], smoothed_states[1])


def iterated_smoother_routine(initial_state: MVNParams,
                              observations: jnp.ndarray,
                              transition_function: Callable or Tuple[Callable, 2],
                              transition_covariances: jnp.ndarray,
                              observation_function: Callable or Tuple[Callable, 2],
                              observation_covariances: jnp.ndarray,
                              linearization_method: Callable,
                              initial_linearization_states: MVNParams = None,
                              n_iter: int = 100):
    """
    Computes the Gauss-Newton iterated extended Kalman smoother

    Parameters
    ----------
    initial_state: MVNParams
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
    linearization_method: callable
        method to linearize
    initial_linearization_states: MVNormalParameters , optional
        points at which to linearize during the first pass.
        If None, these will follow the standard linearization of sequential EKC, CKF
    n_iter: int
        number of times the filter-smoother routine is computed

    Returns
    -------
    iterated_smoothed_trajectories: MVNParams
        The result of the smoothing routine

    """
    n_observations = observations.shape[0]

    transition_covariances, observation_covariances = list(map(
        lambda z: make_matrices_parameters(z, n_observations),
        [transition_covariances,
         observation_covariances]))

    def body(linearization_states, _):
        filtered_states = filter_routine(initial_state, observations, transition_function, transition_covariances,
                                         observation_function, observation_covariances, linearization_method,
                                         linearization_states)
        return smoother_routine(transition_function, transition_covariances, filtered_states, linearization_method,
                                linearization_states), None

    if initial_linearization_states is None:
        initial_linearization_states = body(None, None)

    iterated_smoothed_trajectories, _ = lax.scan(body, initial_linearization_states, jnp.arange(n_iter))
    return MVNParams(iterated_smoothed_trajectories[0], iterated_smoothed_trajectories[1])
