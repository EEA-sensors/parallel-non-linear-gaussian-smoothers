from functools import partial

import numpy as np
import numpy.testing as np_test
import pytest
from jax.config import config

from parsmooth.kalman.sequential.standard import filter_routine
from parsmooth.linearization.taylor import linearize as extended_linearize
from parsmooth.utils import MVNormalParameters


@pytest.fixture(scope="module")
def jax_setup():
    config.update("jax_enable_x64", True)


def linear_transition_function(x, q, a):
    return a @ x + q


def linear_observation_function(x, r, c):
    return c @ x + r


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42, 666])
def test_one_step_linear_extended(dim_x, dim_y, seed):
    np.random.seed(seed)
    y = np.random.randn(dim_y)

    F = np.random.randn(dim_x, dim_x)

    H = np.random.randn(dim_y, dim_x)

    chol_Q = np.random.rand(dim_x, dim_x)
    chol_Q[np.triu_indices(dim_x, k=1)] = 0.
    Q = chol_Q @ chol_Q.T

    chol_R = np.random.rand(dim_y, dim_y)
    chol_R[np.triu_indices(dim_y, k=1)] = 0.
    R = chol_R @ chol_R.T

    transition_fun = partial(linear_transition_function, a=F)
    observation_fun = partial(linear_observation_function, c=H)

    m0 = np.random.randn(dim_x)
    P0_chol = np.random.rand(dim_x, dim_x)
    P0_chol[np.triu_indices(dim_x, k=1)] = 0.
    P0 = P0_chol @ P0_chol.T
    initial_state = MVNormalParameters(m0, P0)
    filtered_states_None_linearization = filter_routine(initial_state, y.reshape(1, -1),
                                                        transition_fun, Q,
                                                        observation_fun, R, extended_linearize,
                                                        None)

    random_mean = np.random.randn(1, dim_x)
    random_chol_cov = np.random.rand(1, dim_x, dim_x)
    random_chol_cov[:, np.triu_indices(dim_x, k=1)] = 0.
    random_cov = np.matmul(random_chol_cov, np.transpose(random_chol_cov, [0, 2, 1]))
    random_linearization = MVNormalParameters(random_mean, random_cov)
    filtered_states_random_linearization = filter_routine(initial_state, y.reshape(1, -1),
                                                          transition_fun, Q,
                                                          observation_fun, R,
                                                          extended_linearize,
                                                          random_linearization)

    np_test.assert_allclose(filtered_states_None_linearization.mean, filtered_states_random_linearization.mean,
                            atol=1e-5, rtol=1e-5)

    np_test.assert_allclose(filtered_states_None_linearization.cov, filtered_states_random_linearization.cov,
                            atol=1e-5, rtol=1e-5)

    P = F @ P0 @ F.T + Q
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    expected_mean = F @ m0 + K @ (y - H @ F @ m0)
    expected_cov = P - K @ H @ P

    np_test.assert_allclose(filtered_states_None_linearization.mean[0], expected_mean,
                            atol=1e-5, rtol=1e-5)
    np_test.assert_allclose(filtered_states_None_linearization.cov[0], expected_cov,
                            atol=1e-5, rtol=1e-5)


@pytest.mark.skip
def test_one_step_linear_cubature(dim_x, dim_y, seed):
    pass


@pytest.mark.skip(reason="Needs implementing")
def test_one_step_extended_cubature(dim_x, dim_y, seed):
    pass


@pytest.mark.skip(reason="Needs implementing")
def test_one_step_sine_cubature(dim_x, dim_y, seed):
    pass


@pytest.mark.skip(reason="Needs implementing")
def test_todo():
    pass


@pytest.mark.skip(reason="Needs implementing")
def test_end_to_end_extended():
    pass


@pytest.mark.skip(reason="Needs implementing")
def test_end_to_end_cubature():
    pass
