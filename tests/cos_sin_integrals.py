import numpy as np

def cos_integral(k,d):
    """k - float, d - list of length 2 of floats."""

    return 1.0/(k**2.0 * d[0] * d[1]) * (np.sin(k * d[0]) *np.sin(k * d[1]) - (1.0 - np.cos(k * d[0])) * (1.0-np.cos(k * d[1])))

def sin_integral(k,d):
    """As above"""

    return 1.0/(k**2.0 * d[0] * d[1]) * ((1.0 - np.cos(k * d[0]))*np.sin(k * d[1]) +  np.sin(k * d[0])*(1.0-np.cos(k * d[1])))
