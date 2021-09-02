# Parallel Iterated Extended and Sigma-Point Kalman Smoothers

Companion code in JAX for the paper Parallel Iterated Extended and Sigma-Point Kalman Smoothers [2].

What is it?
-----------

This is an implementation of parallelized Extended and Sigma-Points Bayesian Filters and Smoothers with CPU/GPU/TPU support coded using using [JAX](https://github.com/google/jax) primitives, in particular associative scan.

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
from parsmooth.parallel import ieks
from parsmooth.utils import MVNParams

initial_guess = MVNParams(...)
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

For more examples, see the [notebooks](https://github.com/EEA-sensors/parallelEKF/tree/master/notebooks).

Acknowlegments
--------------
This JAX-based code was created by [Adrien Corenflos](https://adriencorenflos.github.io/) to implement the original idea by [Fatemeh Yaghoobi](https://fatameh-yaghoobi.github.io/) [2] who provided the initial code for the parallelized extended Kalman filter in pure Python. The sequential cubature filtering code was adapted from some original code by [Zheng Zhao](https://users.aalto.fi/~zhaoz1/).

References
----------

[1] S. Särkkä and A. F. García-Fernández. *Temporal Parallelization of Bayesian Smoothers.* In: IEEE Transactions on Automatic Control 2020.

[2] F. Yaghoobi and A. Corenflos and S. Hassan and S. Särkkä. *Parallel Iterated Extended and Sigma-Points Kalman Smoothers.* To appear in Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
