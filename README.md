# Parallel Filtering and Smoothing

Temporal Parallelization of non-linear Filtering and Smoothing <cite>[2-3]</cite> implemented using JAX methods.


What is it?
-----------

This is an implementation of parallelized Bayesian Filters and Smoothers with CPU/GPU/TPU support coded using using [JAX](https://github.com/google/jax) primitives, in particular associative scan.

Supported features
------------------

* Extended Kalman Filtering and Smoothing
* Cubature Kalman Filtering and Smoothing
* Iterated versions of the above

Installation
------------
- With GPU CUDA 11.0 support
  - Using pip
    Run `pip install https://github.com/EEA-sensors/parallelEKF.git -f  https://storage.googleapis.com/jax-releases/jax_releases.html`
  - By cloning
    Clone https://github.com/EEA-sensors/parallelEKF.git
    Run `pip install -r requirements.txt -f  https://storage.googleapis.com/jax-releases/jax_releases.html`
    Run `python setup.py [install|develop]` depending on the level of installation you want
 - Without GPU support
  - By cloning
    Clone https://github.com/EEA-sensors/parallelEKF.git
    Run `python setup.py [install|develop] --no-deps` depending on the level of installation you want
    Manually install the dependencies `jax` and `jaxlib`, and for examples only `matplotlib`, `numba`, `tqdm`

Example
-------

```python
from pekf.parallel import ieks
from pekf.utils import MVNormalParameters

initial_guess = MVNormalParameters(...)
data = ...
Q = ...  # transition noise covariance matrix
R = ...  # observation error covariance matrix

def transition_function(x):
  ...
  return next_x
  
def observation_function(x):
  ...
  return obs
  
iterated_smoothed_trajectories = ieks(initial_guess, 
                                      data, 
                                      transition_function, 
                                      Q, 
                                      observation_function, 
                                      R, 
                                      n_iter=100)  # runs the parallel IEKS 100 times.

```

For more examples, see the [notebooks](https://github.com/EEA-sensors/parallelEKF/tree/master/notebooks)

References
----------

[1] S. Särkkä. *Bayesian Filtering and Smoothing.*  In: Cambridge University Press 2013.
[2] S Särkkä and A. F. García-Fernández. *Temporal Parallelization of Bayesian Smoothers.* In: IEEE Transactions on Automatic Control 2020.
