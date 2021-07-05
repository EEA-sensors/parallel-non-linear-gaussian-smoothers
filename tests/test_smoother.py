from functools import partial

import numpy as np
import numpy.testing as np_test
import pytest
from jax.config import config

from parsmooth.kalman.sequential.standard import smoother_routine
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

    F = np.random.randn(dim_x, dim_x)

    chol_Q = np.random.rand(dim_x, dim_x)
    chol_Q[np.triu_indices(dim_x, k=1)] = 0.
    Q = chol_Q @ chol_Q.T

    transition_fun = partial(linear_transition_function, a=F)

    filtered_result = MVNormalParameters(np.random.randn(1, dim_x), np.random.randn(1, dim_x, dim_x))


    smoother_states_None_linearization = smoother_routine(transition_fun,
                                                          Q,
                                                          filtered_result,
                                                          extended_linearize,
                                                          None)

    random_mean = np.random.randn(1, dim_x)
    random_cov = np.random.randn(1, dim_x, dim_x)
    random_linearization = MVNormalParameters(random_mean, random_cov)
    smoother_states_random_linearization = smoother_routine(transition_fun,
                                                            Q,
                                                            filtered_result,
                                                            extended_linearize,
                                                            random_linearization)

    np_test.assert_allclose(smoother_states_None_linearization.mean, smoother_states_random_linearization.mean,
                            atol=1e-5, rtol=1e-5)

    np_test.assert_allclose(smoother_states_None_linearization.cov, smoother_states_random_linearization.cov,
                            atol=1e-5, rtol=1e-5)

    m_k = filtered_result.mean
    P_k = filtered_result.cov
    m_ = F @ m_k
    P_ = F @ P_k @ F.T + Q
    G = P_k @ F.T @ np.linalg.inv(P_)
    expected_mean = m_k + G @ (m_k - m_)
    expected_cov = P_k + G @ (P_k - P_) @ G.T

    np_test.assert_allclose(filtered_states_None_linearization.mean[0], expected_mean,
                            atol=1e-5, rtol=1e-5)
    np_test.assert_allclose(filtered_states_None_linearization.cov[0], expected_cov,
                            atol=1e-5, rtol=1e-5)


