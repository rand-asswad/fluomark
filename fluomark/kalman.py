"""Continuous-discrete Kalman filtering for a local-linear-trend model.

The state is ``x = [f, df/dt]`` and evolves according to

    df = (df/dt) dt,        d(df/dt) = kappa dW.

Measurements are ``y_k = f(t_k) + v_k``, with independent Gaussian noise
of variance ``measurement_variance``. The filter is causal: each estimate
at ``t_k`` uses measurements up to and including ``t_k``.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class KalmanResult:
    """Filtering results at measurement times.

    Attributes
    ----------
    states : ndarray, shape (N, 2)
        Posterior estimates of ``[f, df/dt]``.
    covariances : ndarray, shape (N, 2, 2)
        Corresponding posterior covariance matrices.
    log_likelihood : float
        Sum of the Gaussian innovation log densities, excluding the first
        measurement, which initializes the state.
    """

    states: np.ndarray
    covariances: np.ndarray
    log_likelihood: float


def transition_matrices(delta_t, kappa):
    """Return the exact state transition and process covariance over ``delta_t``.

    The process covariance is the integrated white-acceleration covariance
    for the state ``[f, df/dt]`` with diffusion intensity ``kappa**2``.
    """
    if delta_t <= 0 or kappa <= 0:
        raise ValueError("delta_t and kappa must be positive.")

    Phi = np.array([[1.0, delta_t], [0.0, 1.0]])
    Q = kappa**2 * np.array([
        [delta_t**3 / 3.0, delta_t**2 / 2.0],
        [delta_t**2 / 2.0, delta_t],
    ])
    return Phi, Q


def kalman_filter(times, measurements, kappa, measurement_variance):
    """Filter scalar measurements with the local-linear-trend model.

    Parameters
    ----------
    times : array_like, shape (N,)
        Strictly increasing measurement times.
    measurements : array_like, shape (N,)
        Observations of the first state component, ``f``.
    kappa : float
        Diffusion amplitude of the derivative state.
    measurement_variance : float
        Observation noise variance, assumed known and constant.

    Returns
    -------
    KalmanResult
        Posterior state estimates, covariances, and innovation log-likelihood.

    Notes
    -----
    Initialization is ``x_0 = [y_0, 0]`` and ``P_0 = diag(R, 1)``.
    The measurement at ``t_0`` is used only for initialization. Covariance
    updates use Joseph's form for numerical robustness.
    """
    times = np.asarray(times, dtype=float)
    measurements = np.asarray(measurements, dtype=float)
    if times.ndim != 1 or measurements.ndim != 1 or len(times) == 0:
        raise ValueError("times and measurements must be nonempty 1D arrays.")
    if len(times) != len(measurements):
        raise ValueError("times and measurements must have the same length.")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(measurements)):
        raise ValueError("times and measurements must contain finite values.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing.")
    if kappa <= 0 or not np.isfinite(kappa):
        raise ValueError("kappa must be positive and finite.")
    if measurement_variance <= 0 or not np.isfinite(measurement_variance):
        raise ValueError("measurement_variance must be positive and finite.")

    n = len(times)
    states = np.empty((n, 2))
    covariances = np.empty((n, 2, 2))
    x = np.array([measurements[0], 0.0])
    P = np.diag([measurement_variance, 1.0])
    states[0], covariances[0] = x, P
    log_likelihood = 0.0

    for k in range(1, n):
        Phi, Q = transition_matrices(times[k] - times[k - 1], kappa)
        x_pred = Phi @ x
        P_pred = Phi @ P @ Phi.T + Q

        innovation = measurements[k] - x_pred[0]
        S = P_pred[0, 0] + measurement_variance
        if S <= 0 or not np.isfinite(S):
            raise FloatingPointError("Invalid Kalman innovation variance.")

        K = P_pred[:, 0] / S
        x = x_pred + K * innovation
        A = np.eye(2)
        A[:, 0] -= K
        P = A @ P_pred @ A.T + measurement_variance * np.outer(K, K)
        states[k], covariances[k] = x, P
        log_likelihood -= 0.5 * (np.log(2.0 * np.pi * S) + innovation**2 / S)

    return KalmanResult(states, covariances, log_likelihood)


def kalman_log_likelihood(kappa, times, measurements, measurement_variance):
    """Return the Gaussian innovation log-likelihood for a given ``kappa``.

    The first observation is excluded because it initializes the filter.
    Nonpositive ``kappa`` values have log-likelihood ``-inf``.
    """
    if kappa <= 0 or not np.isfinite(kappa):
        return -np.inf
    return kalman_filter(times, measurements, kappa,
                         measurement_variance).log_likelihood


def calibrate_kappa(times, measurements, measurement_variance):
    """Estimate ``kappa`` by maximizing the innovation log-likelihood.

    Optimization is over ``log(kappa)`` within ``[-15, 5]``, using the bounded
    scalar optimizer. Returns ``(kappa_hat, maximum_log_likelihood)``.
    """
    def objective(log_kappa):
        return -kalman_log_likelihood(np.exp(log_kappa), times, measurements,
                                      measurement_variance)

    result = minimize_scalar(objective, bounds=(-15.0, 5.0), method="bounded",
                             options={"xatol": 1e-5})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Maximum-likelihood calibration failed.")
    return np.exp(result.x), -result.fun
