from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BSpline, splev, splprep


def tangent_length(spline: BSpline, u: float):
    x_deriv, y_deriv = splev(u, spline, 1)
    return np.sqrt(x_deriv ** 2 + y_deriv ** 2)


def spline_length(spline: BSpline):
    length, err = quad(
        lambda u: tangent_length(spline, u), 0, 1, epsabs=1e-5, epsrel=1e-5, limit=1000
    )
    return length


def discretize_spline(
    spline: BSpline, resolution_meters: float
) -> List[Tuple[int, int]]:
    length = spline_length(spline)

    num_pts = length // resolution_meters
    remainder = length % resolution_meters
    remainder_frac = remainder / length

    t = np.linspace(0, 1 - remainder_frac, int(num_pts) + 1)

    # add spline endpoint
    t = np.hstack((t, [1.0]))
    # t = np.linspace(0, 1, 40)

    spline_pts = np.stack(spline(t), axis=0)
    # dists = [np.linalg.norm(p2 - p1) for p1, p2 in zip(spline_pts[:-1], spline_pts[1:])]
    # print(dists)
    return spline_pts


def make_parametric_path(points):
    dists = np.cumsum(
        [np.linalg.norm(end - start) for start, end in zip(points[:-1], points[1:])]
    )
    path_u = dists / dists[-1]
    path_u = np.insert(path_u, 0, 0)

    return path_u


def get_unique_nodes(nodes):
    points, idx = np.unique(nodes, axis=0, return_index=True)
    return points[np.argsort(idx), :]


def get_spline_parameters(nodes, order=3, **kwargs):
    """Get B-spline parameters"""
    nodes = get_unique_nodes(nodes)
    if len(nodes) <= 1:
        raise ValueError(
            "There must be at least two unique nodes to interpolate between"
        )

    if len(nodes) <= order:
        # there must be at least order+1 nodes to fit a spline,
        # so linearly interpolate to add some additional nodes
        nodes = interpolate(nodes, n_points=order + 1, order=1)

    # use distance along "curve" as parameter
    path_u = make_parametric_path(nodes)

    tck, _ = splprep(nodes.T, u=path_u, k=order, **kwargs)
    return tck


def interpolate(nodes, n_points=100, order=3, **kwargs):
    tck = get_spline_parameters(nodes, order=order, **kwargs)
    t = np.linspace(0, 1, n_points)
    r = splev(t, tck)
    coords = np.array(list(zip(*r)))
    return coords
