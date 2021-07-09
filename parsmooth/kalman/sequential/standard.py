from typing import Callable, Tuple

import jax.numpy as jnp
import jax.scipy.linalg as jlag
from jax import lax
from jax.lax import cond

from parsmooth.utils import MVNormalParameters, make_matrices_parameters

__all__ = ["filter_routine", "smoother_routine"]


def _linearize(linearization_state, prev_state, cov, fun, method):
    if linearization_state is None:
        linearization_state = prev_state
    else:
        if linearization_state.cov is None:
            linearization_state = MVNormalParameters(linearization_state.mean, prev_state.cov)
    F, c, Q = method(fun, linearization_state, cov, sqrt=False)
    return F, c, Q + F @ (prev_state.cov - linearization_state.cov) @ F.T


def predict(transition_function: Callable or Tuple[Callable, 2],
            transition_covariance: jnp.ndarray or None,
            prior: MVNormalParameters,
            linearization_state: MVNormalParameters or None,
            linearization_method: Callable) -> MVNormalParameters:
    """ Computes the extended kalman filter linearization of :math:`x_{t+1} = f(x_t, \mathcal{N}(0, \Sigma))`

    Parameters
    ----------
    transition_function: callable
        transition function of the state space model
    transition_covariance: (D,D) array
        covariance :math:`\Sigma` of the noise fed to transition_function
    prior: MVNormalParameters
        prior state of the filter x
    linearization_state: MVNormalParameters
        Where to compute the Jacobian
    linearization_method: callable
        The linearization method
    Returns
    -------
    out: MVNormalParameters
        Predicted state
    """

    F, c, Q = _linearize(linearization_state, prior, transition_covariance, transition_function,
                         linearization_method)

    mean = c + jnp.dot(F, prior.mean)

    return MVNormalParameters(mean, Q)


def update(observation_function: Callable or Tuple[Callable, 2],
           observation_covariance: jnp.ndarray or None,
           predicted: MVNormalParameters,
           observation: jnp.ndarray,
           linearization_state: MVNormalParameters or None,
           linearization_method: Callable) -> Tuple[float, MVNormalParameters]:
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
    linearization_state: MVNormalParameters
        Where to compute the Jacobian
    linearization_method: callable
        The linearization method

    Returns
    -------
    loglikelihood: float
        Log-likelihood increment for observation
    updated_state: MVNormalParameters
        filtered state
    """

    H, d, R = _linearize(linearization_state, predicted, observation_covariance, observation_function,
                         linearization_method)
    obs_mean = d + jnp.dot(H, predicted.mean)

    residual = observation - obs_mean
    gain = jnp.dot(predicted.cov, jlag.solve(R, H, sym_pos=True).T)

    ##residual_covariance = jnp.dot(H, jnp.dot(predicted.cov, H.T)) + R
    ##gain = jnp.dot(predicted.cov, jlag.solve(residual_covariance, H, sym_pos=True).T)

    mean = predicted.mean + jnp.dot(gain, residual)
    cov = predicted.cov - jnp.dot(gain, jnp.dot(R, gain.T))
    ##cov = predicted.cov - jnp.dot(gain, jnp.dot(residual_covariance, gain.T))
    updated_state = MVNormalParameters(mean, cov)
    return updated_state


def filter_routine(initial_state: MVNormalParameters,
                   observations: jnp.ndarray,
                   transition_function: Callable or Tuple[Callable, 2],
                   transition_covariances: jnp.ndarray,
                   observation_function: Callable or Tuple[Callable, 2],
                   observation_covariances: jnp.ndarray,
                   linearization_method: Callable,
                   linearization_points: jnp.ndarray = None) -> MVNormalParameters:
    """ Computes the linearized predict-update routine of the Kalman Filter equations and returns a series of filtered_states

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
    linearization_method: callable
        The linearization method
    linearization_points: (n, D) array, optional
        points at which to compute the jacobians.

    Returns
    -------
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
        predicted_state = predict(transition_function, transition_covariance, state, prev_linearization_point,
                                  linearization_method)
        updated_state = update(observation_function, observation_covariance, predicted_state,
                               observation, linearization_point, linearization_method)
        return (updated_state, linearization_point), updated_state

    initial_linearization_point = initial_state if linearization_points is not None else None

    _, filtered_states = lax.scan(body,
                                  (initial_state, initial_linearization_point),
                                  (observations,
                                   transition_covariances,
                                   observation_covariances,
                                   linearization_points),
                                  length=n_observations)

    return MVNormalParameters(filtered_states[0], filtered_states[1])

def smooth(transition_function: Callable or Tuple[Callable, 2],
           transition_covariances: jnp.ndarray,
           filtered_state: MVNormalParameters,
           previous_smoothed: MVNormalParameters,
           linearization_method: Callable,
           linearization_state: MVNormalParameters or None) -> MVNormalParameters:
    """
        One step extended kalman smoother

        Parameters
        ----------
        transition_function: callable
             transition function of the state space model
        transition_covariances: (D,D) array
            covariance :math:`\Sigma` of the noise fed to transition_function
        filtered_state: MVNormalParameters
            mean and cov computed by Kalman Filtering
        previous_smoothed: MVNormalParameters,
            smoothed state of the previous step
        linearization_method: Callable
            The linearization method
        linearization_state: MVNormalParameters
            Where to compute the Jacobian

        Returns
        -------
        smoothed_state: MVNormalParameters
            smoothed state
        """
    F, c, Q = _linearize(linearization_state, filtered_state, transition_covariances, transition_function,
                         linearization_method)
    mean_diff = previous_smoothed.mean - (c + jnp.dot(F, filtered_state.mean))
    cov_diff = previous_smoothed.cov - Q
    # cov = Q + jnp.dot(F, jnp.dot(filtered_state.cov, F.T)
    # cov_diff = previous_smoothed.cov - cov  # ?

    gain = jnp.dot(filtered_state.cov, jlag.solve(Q, F, sym_pos=True).T)
    # gain = jnp.dot(filtered_state.cov, jlag.solve(cov, F, sym_pos=True).T)

    mean = filtered_state.mean + jnp.dot(gain, mean_diff)
    cov = filtered_state.cov + jnp.dot(gain, jnp.dot(cov_diff, gain.T))

    return MVNormalParameters(mean, cov)

def smoother_routine(transition_function: Callable[[jnp.ndarray], jnp.ndarray],
                     transition_covariances: jnp.ndarray,
                     filtered_states: MVNormalParameters,
                     linearization_method: Callable,
                     linearization_points: jnp.ndarray = None) -> MVNormalParameters:
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
            filtered, transition_covariance, linearization_states = inputs
            smoothed_state = smooth(transition_function, transition_covariance, filtered, state,
                                    linearization_method, linearization_states)
            return (i + 1, smoothed_state), smoothed_state

        return cond(j > 0, otherwise, first_step, operand=(state_, list_inputs, j))

    last_state = MVNormalParameters(filtered_states.mean[-1], filtered_states.cov[-1])
    _, smoothed_states = lax.scan(body,
                                  (0, last_state),
                                  [filtered_states, transition_covariances, linearization_points],
                                  reverse=True)

    return MVNormalParameters(smoothed_states[0], smoothed_states[1])


def iterated_smoother_routine(initial_state: MVNormalParameters,
                              observations: jnp.ndarray,
                              transition_function: Callable or Tuple[Callable, 2],
                              transition_covariances: jnp.ndarray,
                              observation_function: Callable or Tuple[Callable, 2],
                              observation_covariances: jnp.ndarray,
                              linearization_method: Callable,
                              initial_linearization_states: MVNormalParameters = None,
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
    linearization_method: callable
        method to linearize
    initial_linearization_states: MVNormalParameters , optional
        points at which to linearize during the first pass.
        If None, these will follow the standard linearization of sequential EKC, CKF
    n_iter: int
        number of times the filter-smoother routine is computed

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

    def body(linearization_states, _):
        filtered_states = filter_routine(initial_state, observations, transition_function, transition_covariances,
                                         observation_function, observation_covariances, linearization_method,
                                         linearization_states)
        return smoother_routine(transition_function, transition_covariances, filtered_states, linearization_method,
                                linearization_states), None

    if initial_linearization_states is None:
        initial_linearization_states = body(None, None)

    iterated_smoothed_trajectories, _ = lax.scan(body, initial_linearization_states, jnp.arange(n_iter))
    return MVNormalParameters(iterated_smoothed_trajectories[0], iterated_smoothed_trajectories[1])





# Previously

# def smooth(transition_function: Callable[[jnp.ndarray], jnp.ndarray],
#            transition_covariance: jnp.array,
#            filtered_state: MVNormalParameters,
#            previous_smoothed: MVNormalParameters,
#            linearization_method: Callable,
#            linearization_state: MVNormalParameters) -> MVNormalParameters:
#     """
#     One step extended kalman smoother
#
#     Parameters
#     ----------
#     transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
#         transition function of the state space model
#     transition_covariance: (D,D) array
#         covariance :math:`\Sigma` of the noise fed to transition_function
#     filtered_state: MVNormalParameters
#         mean and cov computed by Kalman Filtering
#     previous_smoothed: MVNormalParameters,
#         smoothed state of the previous step
#     linearization_state: MVNormalParameters
#         Where to linearize
#
#     Returns
#     -------
#     smoothed_state: MVNormalParameters
#         smoothed state
#     """
#     F, c, Q = _linearize(linearization_state, filtered_state, transition_covariance, transition_function,
#                          linearization_method)
#
#     mean_diff = c + jnp.dot(F, filtered_state.mean) - previous_smoothed.mean
#
#     cov_diff = previous_smoothed.cov - Q
#
#     gain = jnp.dot(filtered_state.cov, jlag.solve(Q, F, sym_pos=True).T)
#
#     mean = filtered_state.mean + jnp.dot(gain, mean_diff)
#     cov = filtered_state.cov + jnp.dot(gain, jnp.dot(cov_diff, gain.T))
#     return MVNormalParameters(mean, cov)

#
# def smoother_routine(transition_function: Callable[[jnp.ndarray], jnp.ndarray],
#                      transition_covariances: jnp.ndarray,
#                      filtered_states: MVNormalParameters,
#                      linearization_method: Callable,
#                      linearization_states: MVNormalParameters) -> MVNormalParameters:
#     """ Computes the extended Rauch-Tung-Striebel (a.k.a extended Kalman) smoother routine and returns a series of smoothed_states
#
#     Parameters
#     ----------
#     filtered_states: MVNormalParameters
#         Filtered states obtained from Kalman Filter
#     transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
#         transition function of the state space model
#     transition_covariances: (D, D) or (1, D, D) or (n, D, D) array
#         transition covariances for each time step, if passed only one, it is repeated n times
#     linearization_points: (n, D) array, optional
#         points at which to compute the jacobians.
#
#     Returns
#     -------
#     smoothed_states: MVNormalParameters
#         list of smoothed states
#     """
#     n_observations = filtered_states.mean.shape[0]
#
#     transition_covariances = make_matrices_parameters(transition_covariances, n_observations)
#
#     def body(carry, list_inputs):
#         j, state_ = carry
#
#         def first_step(operand):
#             state, _inputs, i = operand
#             return (i + 1, state), state
#
#         def otherwise(operand):
#             state, inputs, i = operand
#             filtered, transition_covariance, linearization_states = inputs
#             smoothed_state = smooth(transition_function, transition_covariance, filtered, state,
#                                     linearization_method, linearization_states)
#             return (i + 1, smoothed_state), smoothed_state
#
#         return cond(j > 0, otherwise, first_step, operand=(state_, list_inputs, j))
#
#     last_state = MVNormalParameters(filtered_states.mean[-1], filtered_states.cov[-1])
#     _, smoothed_states = lax.scan(body,
#                                   (0, last_state),
#                                   [filtered_states, transition_covariances, linearization_states],
#                                   reverse=True)
#
#     return MVNormalParameters(smoothed_states[0], smoothed_states[1])
#
#
# def iterated_smoother_routine(initial_state: MVNormalParameters,
#                               observations: jnp.ndarray,
#                               transition_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
#                               transition_covariances: jnp.ndarray,
#                               observation_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
#                               observation_covariances: jnp.ndarray,
#                               linearization_method: Callable,
#                               initial_linearization_states: MVNormalParameters = None,
#                               n_iter: int = 100):
#     """
#     Computes the Gauss-Newton iterated extended Kalman smoother
#
#     Parameters
#     ----------
#     initial_state: MVNormalParameters
#         prior belief on the initial state distribution
#     observations: (n, K) array
#         array of n observations of dimension K
#     transition_function: callable :math:`f(x_t,\epsilon_t)\mapsto x_{t-1}`
#         transition function of the state space model
#     transition_covariances: (D, D) or (1, D, D) or (n, D, D) array
#         transition covariances for each time step, if passed only one, it is repeated n times
#     observation_function: callable :math:`h(x_t,\epsilon_t)\mapsto y_t`
#         observation function of the state space model
#     observation_covariances: (K, K) or (1, K, K) or (n, K, K) array
#         observation error covariances for each time step, if passed only one, it is repeated n times
#     linearization_method: callable
#         method to linearize
#     initial_linearization_states: MVNormalParameters , optional
#         points at which to linearize during the first pass.
#         If None, these will follow the standard linearization of sequential EKC, CKF
#     n_iter: int
#         number of times the filter-smoother routine is computed
#
#     Returns
#     -------
#     iterated_smoothed_trajectories: MVNormalParameters
#         The result of the smoothing routine
#
#     """
#     n_observations = observations.shape[0]
#
#     transition_covariances, observation_covariances = list(map(
#         lambda z: make_matrices_parameters(z, n_observations),
#         [transition_covariances,
#          observation_covariances]))
#
#     def body(linearization_states, _):
#         filtered_states = filter_routine(initial_state, observations, transition_function, transition_covariances,
#                                          observation_function, observation_covariances, linearization_method,
#                                          linearization_states)
#         return smoother_routine(transition_function, transition_covariances, filtered_states, linearization_method,
#                                 linearization_states), None
#
#     if initial_linearization_states is None:
#         initial_linearization_states = body(None, None)
#
#     iterated_smoothed_trajectories, _ = lax.scan(body, initial_linearization_states, jnp.arange(n_iter))
#     return MVNormalParameters(iterated_smoothed_trajectories[0], iterated_smoothed_trajectories[1])