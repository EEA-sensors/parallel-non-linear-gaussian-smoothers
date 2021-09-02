from functools import partial

import jax.numpy as jnp
import numpy as np
import numpy.testing as np_test
import pytest

from parsmooth.linearization.taylor import linearize
from parsmooth.utils import MVNParams


def linear_function(x, q, a, b):
    return b @ x + a @ q


def linear_mean_function(x, b):
    return b @ x


def linear_cov_function(_, q, a):
    return a @ q @ a.T


def sine_function(x, q):
    return jnp.sin(x) + jnp.sin(q)


def sine_mean_function(x):
    return jnp.sin(x)


def sine_cov_function(_):
    return jnp.array([[0.5 * (1 - jnp.exp(-2))]])


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42, 666])
def test_standard_extended_linearization(d, seed):
    np.random.seed(seed)
    a = np.random.randn(d, d)
    b = np.random.randn(d, d)
    zeros = jnp.zeros((d,))
    cov = jnp.eye(d)
    state = MVNParams(zeros, cov)
    F, c, Q = linearize(partial(linear_function, a=a, b=b),
                        state, cov, sqrt=False)
    np_test.assert_allclose(F, b)
    np_test.assert_allclose(c, zeros)
    np_test.assert_allclose(Q, a @ cov @ a.T + b @ cov @ b.T)


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42, 666])
def test_mean_cov_standard_extended_linearization(d, seed):
    zeros = jnp.zeros((d,))
    cov = jnp.eye(d)
    np.random.seed(seed)
    a = np.random.randn(d, d)
    b = np.random.randn(d, d)
    state = MVNParams(zeros, cov)
    funs = (partial(linear_mean_function, b=b), partial(linear_cov_function, q=cov, a=a))
    F, c, Q = linearize(funs, state, cov, sqrt=False)
    np_test.assert_allclose(F, b)
    np_test.assert_allclose(c, zeros)
    np_test.assert_allclose(Q, a @ cov @ a.T)


def test_sine_extended_linearization():
    zeros = jnp.zeros((1,))
    cov = jnp.eye(1)
    funs = (sine_mean_function, sine_cov_function)
    state = MVNParams(zeros, cov)
    F1, c1, Q1 = linearize(funs, state, cov, sqrt=False)
    F2, c2, Q2 = linearize(sine_function, state, cov, sqrt=False)
    np_test.assert_allclose(F1, F2)
    np_test.assert_allclose(c1, c2)
    np_test.assert_allclose(Q1, 0.5 * (1 - np.exp(-2)))
    np_test.assert_allclose(Q2, 1.)
