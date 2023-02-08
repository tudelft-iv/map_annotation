import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline


def parametric_spline(path_xy, kind):
    path_t = np.linspace(0, 1, path_xy.size // 2)
    spline = interp1d(path_t, path_xy.T, kind=kind)
    return spline, path_t


def strictly_increasing(L):
    return all(x < y for x, y in zip(L, L[1:]))


def strictly_decreasing(L):
    return all(x > y for x, y in zip(L, L[1:]))


def non_increasing(L):
    return all(x >= y for x, y in zip(L, L[1:]))


def non_decreasing(L):
    return all(x <= y for x, y in zip(L, L[1:]))


def monotonic(L):
    return non_increasing(L) or non_decreasing(L)
