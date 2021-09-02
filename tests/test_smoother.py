from functools import partial

import numpy as np
import numpy.testing as np_test
import pytest
from jax.config import config

from parsmooth.kalman.sequential.standard import smoother_routine
from parsmooth.linearization.taylor import linearize as extended_linearize
from parsmooth.linearization.sigma_points import linearize as sigma_points_linearize
from parsmooth.utils import MVNParams


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
@pytest.mark.parametrize("linearization_method", [extended_linearize])
def test_one_step_linear(dim_x, dim_y, seed, linearization_method):
    np.random.seed(seed)

    F = np.random.randn(dim_x, dim_x)

    chol_Q = np.random.rand(dim_x, dim_x)
    chol_Q[np.triu_indices(dim_x, k=1)] = 0.
    Q = chol_Q @ chol_Q.T

    transition_fun = partial(linear_transition_function, a=F)

    filtered_mean = np.random.randn(2, dim_x)
    filtered_chol_cov = np.random.rand(2, dim_x, dim_x)
    filtered_chol_cov[:, np.triu_indices(dim_x, k=1)] = 0.
    filtered_cov = np.matmul(filtered_chol_cov, np.transpose(filtered_chol_cov, [0, 2, 1]))

    filtered_result = MVNParams(filtered_mean, filtered_cov)

    smoother_states_None_linearization = smoother_routine(transition_fun,
                                                          Q,
                                                          filtered_result,
                                                          linearization_method,
                                                          None)

    random_mean = np.random.randn(2, dim_x)
    random_chol_cov = np.random.rand(2, dim_x, dim_x)
    random_chol_cov[:, np.triu_indices(dim_x, k=1)] = 0.
    random_cov = np.matmul(random_chol_cov, np.transpose(random_chol_cov, [0, 2, 1]))
    random_linearization = MVNParams(random_mean, random_cov)

    smoother_states_random_linearization = smoother_routine(transition_fun,
                                                            Q,
                                                            filtered_result,
                                                            linearization_method,
                                                            random_linearization)

    np_test.assert_allclose(smoother_states_None_linearization.mean, smoother_states_random_linearization.mean,
                            atol=1e-5, rtol=1e-5)

    np_test.assert_allclose(smoother_states_None_linearization.cov, smoother_states_random_linearization.cov,
                            atol=1e-5, rtol=1e-5)

    prev_smoothed_mean = filtered_result.mean[1]
    prev_smoothed_cov = filtered_result.cov[1]
    m_k = filtered_result.mean[0]
    P_k = filtered_result.cov[0]
    m_ = F @ m_k
    P_ = F @ P_k @ F.T + Q
    G = P_k @ F.T @ np.linalg.inv(P_)
    expected_mean = m_k + G @ (prev_smoothed_mean - m_)
    expected_cov = P_k + G @ (prev_smoothed_cov - P_) @ G.T

    np_test.assert_allclose(smoother_states_None_linearization.cov[0], expected_cov,
                            atol=1e-5, rtol=1e-5)
    np_test.assert_allclose(smoother_states_None_linearization.mean[0], expected_mean,
                            atol=1e-5, rtol=1e-5)
