import jax.numpy as jnp
from jax import jacfwd


def linearize(f, linearization_point, noise_cov, sqrt=False):
    if sqrt:
        return _sqrt_linearize(f, linearization_point, noise_cov)
    else:
        return _standard_linearize(f, linearization_point, noise_cov)


def _sqrt_linearize(f, linearization_point, noise_cov):
    pass


def _standard_linearize(f, linearization_point, noise_cov):
    if isinstance(f, tuple):
        mean_f, cov_f = f
        Fx = jacfwd(mean_f, 0)(linearization_point)
        bias = mean_f(linearization_point) - Fx @ linearization_point
        Q = cov_f(linearization_point)
    elif callable(f):
        d = noise_cov.shape[0]
        zero = jnp.zeros((d,), dtype=noise_cov.dtype)
        Fx, Fq = jacfwd(f, (0, 1))(linearization_point, zero)
        Q = Fq @ noise_cov @ Fq.T
        bias = f(linearization_point, zero) - Fx @ linearization_point
    else:
        raise ValueError("f is either a tuple of univariate functions, or a bivariate function")
    return Fx, bias, Q
