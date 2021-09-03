from typing import Callable, Tuple

import jax.numpy as jnp
import jax.scipy.linalg as jlag
from jax import lax, jacfwd
from jax.lax import cond


from parsmooth.utils import MVNormalParameters, make_matrices_parameters

__all__ = ["filter_routine", "smoother_routine"]

    
def _linearize(transition_function, observation_function, R, Q, x_hat, m0):
    

    jac_observation_function = jacfwd(observation_function, 0)
    jac_transition_function = jacfwd(transion_function, 0)
    
    x_hat_m0 = jnp.concatenate([m0.mean.reshape(1,-1),x_hat.mean], axis=0)
    x_hat_m0 = jnp.expand_dims(x_hat_m0, axis=1)
    x_2step = jnp.concatenate([x_hat_m0[:-1],x_hat_m0[1:]],axis=1)
  
    def body(_, x_2step):
        
        x_k_1 = x_2step[0]
        x_k = x_2step[1]
        F = jac_transition_function(x_k_1)   
        c = transition_function(x_k_1) - F @ x_k_1
        Qp = jlag.cholesky(Q, lower=True)
        
        H = jac_observation_function(x_k)   
        d = observation_function(x_k) - H @ x_k
        Rp = jlag.cholesky(R, lower=True)
        
        return None, (F, c, Qp, H, d, Rp)
    
    _, linear_param = lax.scan(body, None, x_2step)
    
    return linear_param    


def Tria(A):
    tria_A = jlag.qr(A.T, mode = 'economic')[1].T
    return tria_A



def predict(prior: MVNormalParameters,
            F: jnp.ndarray,
            c: jnp.ndarray,
            W: jnp.ndarray) -> MVNormalParameters:
    r""" Computes the extended kalman filter linearization of :math:`x_{t+1} = f(x_t, \mathcal{N}(0, \Sigma))`
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
    
    mean = c + jnp.dot(F, prior.mean)
    N_ = jlag.cholesky(prior.cov, lower = True)
    N = Tria(jnp.concatenate((F @ N_, W), axis = 1))

    return MVNormalParameters(mean, N)

def update(predicted: MVNormalParameters,
           observation: jnp.ndarray,
           H: jnp.ndarray,
           d: jnp.ndarray,
           V: jnp.ndarray) -> Tuple[float, MVNormalParameters]:
    r""" Computes the extended kalman filter linearization of :math:`x_t \mid y_t`
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
    
    nx = predicted.cov.shape[0]
    ny = V.shape[1]
    Psi = jnp.block([ [H @ predicted.cov, V], [predicted.cov, jnp.zeros((nx, ny)) ] ])
    Tria_Psi = Tria(Psi)
    Psi11 = Tria_Psi[:ny , :ny]
    Psi21 = Tria_Psi[ny: ny + nx , :ny]
    Psi22 = Tria_Psi[ny: ny + nx , ny:]
    Y = Psi11
    gain = jlag.solve(Psi11.T, Psi21.T).T    
    
    
    obs_mean = d + jnp.dot(H, predicted.mean)
    residual = observation - obs_mean
    mean = predicted.mean + jnp.dot(gain, residual)
    cov = Psi22
    
    updated_state = MVNormalParameters(mean, cov)
    
    return updated_state

def filter_routine(initial_state: MVNormalParameters,
                   observations: jnp.ndarray,
                   Fs: jnp.ndarray,
                   cs: jnp.ndarray,
                   Ws: jnp.ndarray,
                   Hs: jnp.ndarray,
                   ds: jnp.ndarray,
                   Vs: jnp.ndarray) -> MVNormalParameters:
    r""" Computes the linearized predict-update routine of the Kalman Filter equations and returns a series of filtered_states
    Parameters
    ----------
    initial_state: MVNormalParameters
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
    filtered_states: MVNormalParameters
        list of filtered states
    """
    n_observations = observations.shape[0]
    
    def body(carry, inputs):
        state = carry
        observation, F, c, W, H, d, V = inputs
          
        
        predicted_state = predict(state, F, c, W)
        updated_state = update(predicted_state, observation, H, d, V)
        return updated_state, updated_state


    _, filtered_states = lax.scan(body,
                                  initial_state,
                                  (observations,
                                   Fs,
                                   cs,
                                   Ws,
                                   Hs,
                                   ds,
                                   Vs),
                                  length=n_observations)
    


    return MVNormalParameters(filtered_states[0], filtered_states[1])



def smooth(filtered_state: MVNormalParameters,
           previous_smoothed: MVNormalParameters,
           F: jnp.ndarray,
           c: jnp.ndarray,
           W: jnp.ndarray) -> MVNormalParameters:
    r"""One step extended kalman smoother
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
    
    nx = F.shape[0]
    N = filtered_state.cov
    Phi = jnp.block([[F @ N, W], [N, jnp.zeros((N.shape[0], W.shape[1]))]])
    Tria_Phi = Tria(Phi)
    Phi11 = Tria_Phi[:nx , :nx]
    Phi21 = Tria_Phi[nx: nx + N.shape[0] , :nx]
    Phi22 = Tria_Phi[nx: nx + N.shape[0] , nx:] 
    
    gain = jlag.solve(Phi11.T, Phi21.T, sym_pos=True).T
    mean_diff = previous_smoothed.mean - (c + jnp.dot(F, filtered_state.mean))
    
    mean = filtered_state.mean + jnp.dot(gain, mean_diff)
    cov = Tria(jnp.concatenate((Phi22, gain @ previous_smoothed.cov), axis = 1))
    
    return MVNormalParameters(mean, cov)

def smoother_routine(filtered_states: MVNormalParameters,
                     Fs,
                     cs,
                     Ws,
                     Hs,
                     ds,
                     Vs) -> MVNormalParameters:
    """ Computes the extended Rauch-Tung-Striebel (a.k.a extended Kalman) smoother routine and returns a series of smoothed_states
    Parameters
    ----------
    filtered_states: MVNormalParameters
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
    smoothed_states: MVNormalParameters
        list of smoothed states
    """
    n_observations = filtered_states.mean.shape[0]

    def body(carry, list_inputs):
        j, state_ = carry

        def first_step(operand):
            state, _inputs, i = operand
            return (i + 1, state), state

        def otherwise(operand):
            state, inputs, i = operand
            filtered, F, c, W = inputs
            smoothed_state = smooth(filtered, state, F, c, W)
            return (i + 1, smoothed_state), smoothed_state

        return cond(j > 0, otherwise, first_step, operand=(state_, list_inputs, j))

    last_state = MVNormalParameters(filtered_states.mean[-1], filtered_states.cov[-1])
    _, smoothed_states = lax.scan(body,
                                  (0, last_state),
                                  [filtered_states, Fs, cs, Ws],
                                  reverse=True)

    return MVNormalParameters(smoothed_states[0], smoothed_states[1])


def iterated_smoother_routine(initial_state: MVNormalParameters,
                              observations: jnp.ndarray,
                              transition_function: Callable[[jnp.ndarray], jnp.ndarray],
                              transition_covariance: jnp.ndarray,
                              observation_function: Callable[[jnp.ndarray], jnp.ndarray],
                              observation_covariance: jnp.ndarray,
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
        [transition_covariance,
         observation_covariance]))

    def body(linearization_states, _):
        
        linear_param = _linearize(transition_function, observation_function, observation_covariance, transition_covariance, linearization_states, initial_state)
        
        filtered_states = filter_routine(initial_state, observations, *linear_param)
        
        return smoother_routine(filtered_states, *linear_param), None

    if initial_linearization_states is None:
        initial_linearization_states = body(None, None)

    iterated_smoothed_trajectories, _ = lax.scan(body, initial_linearization_states, jnp.arange(n_iter))
    return MVNormalParameters(iterated_smoothed_trajectories[0], iterated_smoothed_trajectories[1])









