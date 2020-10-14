from collections import namedtuple
from typing import Tuple, Callable

import jax.numpy as jnp
import numpy as np

from ..utils import MVNormalParameters

SigmaPoints = namedtuple(
    'SigmaPoints', ['points', 'wm', 'wc']
)


def cubature_weights(n_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Computes the weights associated with the spherical cubature method.
    The number of sigma-points is 2 * n_dim

    Parameters
    ----------
    n_dim: int
        Dimensionality of the problem

    Returns
    -------
    wm: np.ndarray
        Weights means
    wc: np.ndarray
        Weights covariances
    xi: np.ndarray
        Orthogonal vectors
    """
    wm = np.ones(shape=(2 * n_dim,)) / (2 * n_dim)
    wc = wm
    xi = np.concatenate([np.eye(n_dim), -np.eye(n_dim)], axis=0) * np.sqrt(n_dim)

    return wm, wc, xi


def get_sigma_points(mv_normal_parameters: MVNormalParameters) -> SigmaPoints:
    """ Computes the sigma-points for a given mv normal distribution
    The number of sigma-points is 2*n_dim

    Parameters
    ----------
    mv_normal_parameters: MVNormalParameters
        Mean and Covariance of the distribution

    Returns
    -------
    out: SigmaPoints
        sigma points for the spherical cubature transform

    """
    mean = mv_normal_parameters.mean
    n_dim = mean.shape[0]

    wm, wc, xi = cubature_weights(n_dim)

    sigma_points = jnp.repeat(mean.reshape(1, -1), wm.shape[0], axis=0) \
                   + jnp.dot(jnp.linalg.cholesky(mv_normal_parameters.cov), xi.T).T

    return SigmaPoints(sigma_points, wm, wc)


def get_mv_normal_parameters(sigma_points: SigmaPoints, noise: np.ndarray) -> MVNormalParameters:
    """ Computes the MV Normal distribution parameters associated with the sigma points

    Parameters
    ----------
    sigma_points: SigmaPoints
        shape of sigma_points.points is (n_dim, 2*n_dim)
    noise: (n_dim, n_dim) array
        additive noise covariance matrix, if any
    Returns
    -------
    out: MVNormalParameters
        Mean and covariance of RV of dimension K computed from sigma-points
    """
    mean = jnp.dot(sigma_points.wm, sigma_points.points)
    diff = sigma_points.points - mean.reshape(1, -1)
    cov = jnp.dot(sigma_points.wc.reshape(1, -1) * diff.T, diff) + noise
    return MVNormalParameters(mean, cov=cov)


def transform(sigma_points: SigmaPoints,
              noise: SigmaPoints or MVNormalParameters,
              f: Callable) -> Tuple[SigmaPoints, MVNormalParameters]:
    """ Apply a function on the sigma points.

    Parameters
    ----------
    sigma_points: SigmaPoints
        sigma points to be transformed by f
    noise: SigmaPoints or MVNormalParameters
        if is_additive is True, then this is SigmaPoints, else it's assumed to be a MVNormalParameters
    f: Callable
        :math: `f(x)`

    Returns
    -------
    propagated_sigma_points: SigmaPoints
        sigma points after transformation
    propagated_mvn_parameters: MVNormalParameters
        approximate mnv from the propagated sigma points
    """

    propagated_points = f(sigma_points.points)
    propagated_sigma_points = SigmaPoints(propagated_points, sigma_points.wm, sigma_points.wc)

    propagated_mvn_parameters = get_mv_normal_parameters(propagated_sigma_points, noise.cov)

    return propagated_sigma_points, propagated_mvn_parameters
