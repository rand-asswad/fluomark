import numpy as np


def monod(s, max_growth, half_saturation):
    """Monod growth function.

    Parameters
    ----------
    s : array-like
        Substrate concentration(s).
    max_growth : float
        Maximum growth rate.
    half_saturation : float
        Half-saturation constant.

    Returns
    -------
    array-like
        Growth rate(s) corresponding to the substrate concentration(s).
    """
    func = lambda s: max_growth * s / (half_saturation + s)
    if np.isscalar(s):
        return func(s) if s > 0 else 0
    s = np.asarray(s)
    return np.where(s > 0, func(s), 0)

def monod_inverse(growth, max_growth, half_saturation):
    """Inverse of the Monod growth function.

    Parameters
    ----------
    growth : array-like
        Growth rate(s).
    max_growth : float
        Maximum growth rate.
    half_saturation : float
        Half-saturation constant.

    Returns
    -------
    array-like
        Substrate concentration(s) corresponding to the growth rate(s).
    """
    func = lambda g: half_saturation * g / (max_growth - g)
    if np.isscalar(growth):
        if growth <= 0:
            return 0
        elif growth >= max_growth:
            return np.inf
        else:
            return func(growth)
    growth = np.asarray(growth)
    return np.where(growth <= 0, 0, np.where(growth >= max_growth, np.inf, func(growth)))

def monod_derivative(s, max_growth, half_saturation):
    """Derivative of the Monod growth function with respect to substrate concentration.

    Parameters
    ----------
    s : array-like
        Substrate concentration(s).
    max_growth : float
        Maximum growth rate.
    half_saturation : float
        Half-saturation constant.

    Returns
    -------
    array-like
        Derivative(s) of the growth rate with respect to substrate concentration.
    """
    func = lambda s: max_growth * half_saturation / (half_saturation + s)**2
    if np.isscalar(s):
        return func(s) if s > 0 else 0
    s = np.asarray(s)
    return np.where(s > 0, func(s), 0)
