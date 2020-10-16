from collections import namedtuple
from typing import Tuple

import jax.numpy as jnp
import numpy as np

from ..utils import MVNormalParameters

SigmaPoints = namedtuple(
    'SigmaPoints', ['points', 'wm', 'wc']
)


def mean_sigma_points(points):
    """
    Computes the mean of sigma points

    Parameters
    ----------
    points: SigmaPoints
        The sigma points

    Returns
    -------
    mean: array_like
        the mean of the sigma points
    """
    return jnp.dot(points.wm, points.points)


def covariance_sigma_points(points_1, mean_1, points_2, mean_2):
    """
    Computes the covariance between two sets of sigma points

    Parameters
    ----------
    points_1: SigmaPoints
        first set of sigma points
    mean_1: array_like
        assumed mean of the first set of points
    points_2: SigmaPoints
        second set of sigma points
    points_1: SigmaPoints
        assumed mean of the second set of points

    Returns
    -------
    cov: array_like
        the covariance of the two sets
    """
    one = (points_1.points - mean_1.reshape(1, -1)).T * points_1.wc.reshape(1, -1)
    two = points_2.points - mean_2.reshape(1, -1)
    return jnp.dot(one, two)


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


def get_mv_normal_parameters(sigma_points: SigmaPoints) -> MVNormalParameters:
    """ Computes the MV Normal distribution parameters associated with the sigma points

    Parameters
    ----------
    sigma_points: SigmaPoints
        shape of sigma_points.points is (n_dim, 2*n_dim)
    Returns
    -------
    out: MVNormalParameters
        Mean and covariance of RV of dimension K computed from sigma-points
    """
    m = mean_sigma_points(sigma_points)
    cov = covariance_sigma_points(sigma_points, m, sigma_points, m)
    return MVNormalParameters(m, cov=cov)
