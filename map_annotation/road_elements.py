import math
from enum import Enum

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, interp1d
from shapely.geometry import LineString

from map_annotation.interpolation import (discretize_spline,
                                          get_spline_parameters,
                                          get_unique_nodes, interpolate,
                                          make_parametric_path)


class RoadElement:
    RoadType = Enum(
        "RoadType",
        ["RESIDENTIAL", "CAR", "BIKE_LANE", "CYCLE_STREET", "TRAM_BUS_LANE", "HIGHWAY"],
    )
    AllowedAgent = Enum(
        "AllowedAgent", ["PEDESTRIAN", "CYCLIST", "CAR", "BUS", "TRAM", "OTHER"]
    )
    RoadType_AllowedAgents_map = {
        RoadType.RESIDENTIAL: [
            AllowedAgent.PEDESTRIAN,
            AllowedAgent.CYCLIST,
            AllowedAgent.CAR,
            AllowedAgent.OTHER,
        ],
        RoadType.CAR: [
            AllowedAgent.CAR,
            AllowedAgent.BUS,
            AllowedAgent.OTHER,
        ],
        RoadType.BIKE_LANE: [
            AllowedAgent.CYCLIST,
        ],
        RoadType.CYCLE_STREET: [
            AllowedAgent.PEDESTRIAN,
            AllowedAgent.CYCLIST,
            AllowedAgent.CAR,
            AllowedAgent.OTHER,
        ],
        RoadType.TRAM_BUS_LANE: [
            AllowedAgent.BUS,
            AllowedAgent.TRAM,
            AllowedAgent.OTHER,
        ],
        RoadType.HIGHWAY: [
            AllowedAgent.CAR,
            AllowedAgent.BUS,
            AllowedAgent.OTHER,
        ],
    }

    def __init__(self, element_id, road_type=None, allowed_agents=None):
        self.id = element_id
        self.road_type = road_type
        self.allowed_agents = allowed_agents


class RoadElementCollection:
    def __init__(self):
        # self.geod = pyproj.Geod(ellps="WGS84")  # Convert coordinates to meters
        self.init_elements()

    def init_elements(self):
        self.elements = {}
        self.element_ids = []

    def load(self, filepath):
        file_ = gpd.read_file(filepath)
        df = gpd.GeoDataFrame.explode(file_, index_parts=False)
        return self.from_df(df)

    def from_df(self, df: pd.DataFrame):
        raise NotImplementedError

    def to_df(self):
        raise NotImplementedError

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= len(self.element_ids):
            raise StopIteration
        element_id = self.element_ids[self._idx]
        self._idx += 1
        return self.elements[element_id]

    def __getitem__(self, element_id):
        return self.elements[element_id]

    def __len__(self):
        return len(self.elements)


class RoadLine(RoadElement):
    BoundaryType = Enum("BoundaryType", ["SOLID", "DASHED"])
    LineType = Enum(
        "LineType", ["CENTERLINE", "BOUNDARY", "CONNECTOR", "START_LINE", "END_LINE"]
    )

    def __init__(
        self,
        boundary_id: int,
        nodes: np.array,
        boundary_type: BoundaryType = None,
        road_type: RoadElement.RoadType = None,
        line_type: LineType = None,
        allowed_agents: list[RoadElement.AllowedAgent] = None,
    ):
        super().__init__(boundary_id, road_type, allowed_agents)
        self.nodes = nodes
        self.boundary_type = boundary_type
        self.line_type = line_type

    def __len__(self):
        return len(self.nodes)

    @property
    def geometry(self):
        return LineString(self.nodes)

    def get_spline_parameters(self, nodes=None, order=3, **kwargs):
        nodes = self.nodes if nodes is None else nodes
        return get_spline_parameters(nodes, order=order, **kwargs)

    def interpolate(self, n_points=100, order=3, **kwargs):
        return interpolate(self.nodes, n_points=n_points, order=order, **kwargs)

    def interp1d(self, n_points=100):
        nodes = get_unique_nodes(self.nodes)
        if len(nodes) <= 1:
            raise ValueError(
                "There must be at least two unique nodes to interpolate between"
            )

        path_u = make_parametric_path(nodes)
        interpolator = interp1d(path_u, nodes.T, kind="linear")

        path_t = np.linspace(0, 1, n_points)
        nodes_intp = interpolator(path_t).T
        return nodes_intp

    def discretize(self, resolution, order=3, **kwargs):
        # resolution is in m
        assert resolution > 0, "Resolution must be greater than zero"
        t, c, k = self.get_spline_parameters(order=order, **kwargs)

        spline = BSpline(t, np.array(c).T, k)
        xy = discretize_spline(spline, resolution)

        theta = [
            np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) for p1, p2 in zip(xy[:-1], xy[1:])
        ]
        theta.insert(0, 0.0)  # add yaw of 0 to first pose
        theta = np.array(theta)

        poses = np.hstack((xy, theta[:, None]))
        return poses
