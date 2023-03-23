import math
from enum import Enum

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import utm
from scipy.interpolate import interp1d

from map_annotation.transforms import CoordTransformer


class RoadElement:
    RoadType = Enum("RoadType", ["URBAN", "CAR", "BIKE", "BUS"])
    AllowedAgent = Enum(
        "AllowedAgent", ["PEDESTRIAN", "CYCLIST", "CAR", "BUS", "TRAM", "OTHER"]
    )
    _transformer = CoordTransformer()

    def __init__(self, element_id, road_type=None, allowed_agents=None):
        self.id = element_id
        self.road_type = road_type
        self.allowed_agents = allowed_agents

        # instantiating this is slow, so this is a workaround
        self.transformer = RoadElement._transformer


class RoadElementCollection:
    def __init__(self):
        # TODO why this coordinate system? --> psuedo-mercator projection
        self.geod = pyproj.Geod(ellps="WGS84")  # Convert coordinates to meters

        self.transformer = CoordTransformer()

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
    LineType = Enum("LineType", ["CENTERLINE", "BOUNDARY", "CONNECTOR"])

    def __init__(
        self,
        boundary_id,
        nodes,
        boundary_type: BoundaryType = None,
        road_type: RoadElement.RoadType = None,
        line_type: LineType = None,
        allowed_agents: list[RoadElement.AllowedAgent] = None,
    ):
        # which frame should 'nodes' be in?
        super().__init__(boundary_id, road_type, allowed_agents)
        self.nodes = nodes
        self.boundary_type = boundary_type
        self.line_type = line_type

    def __len__(self):
        return len(self.nodes)

    def get_attr_in_frame(self, attr, frame):
        if attr != "nodes":
            raise ValueError(f"Only implemented for 'nodes'. Value passed: '{attr}'.")

        attr_str = f"{attr}"
        if frame == "global":
            pass
        else:
            attr_str = attr_str + "_" + frame

        return getattr(self, attr_str)

    @property
    def nodes_utm(self):
        # nodes are in (lon, lat) format, utm expectes (lat, lon)
        lonlat = self.nodes_lonlat
        nodes_utm = utm.from_latlon(lonlat[:, 1], lonlat[:, 0])
        self.utm_zone = nodes_utm[2:]
        return np.stack(nodes_utm[:2], axis=-1)

    @property
    def nodes_lonlat(self):
        # nodes_global = list(
        #    self.transformer.t_global_nl(self.nodes[:, 0], self.nodes[:, 1])
        # )
        nodes_global = self.transformer.t_global_nl(self.nodes[:, 0], self.nodes[:, 1])
        nodes_global = np.vstack(nodes_global).T
        return nodes_global

    def _get_nodes_in_frame(self, frame):
        if frame == "nl":
            return self.nodes
        elif frame == "lonlat":
            return self.nodes_lonlat
        elif frame == "utm":
            return self.nodes_utm
        else:
            raise ValueError(f'Frame "{frame}" not valid.')

    def interpolate(self, frame="utm", n_points=100, kind="linear"):
        nodes = self._get_nodes_in_frame(frame)

        path_t = np.linspace(0, 1, nodes.size // 2)

        r = nodes.T
        spline = interp1d(path_t, r, kind=kind)
        t = np.linspace(np.min(path_t), np.max(path_t), n_points)
        r = spline(t)

        return r.T

    def discretize(self, resolution, interp_kind="cubic"):
        assert resolution > 0, "Resolution must be non-negative"
        n_points = math.ceil(len(self) / resolution) + 1
        assert (
            n_points > 1
        ), "there must be more than one point to discretize the RoadLine - try lowering the resolution"

        # interpolate with number of points and calculate yaws
        xy = self.interpolate(n_points=n_points, kind=interp_kind)
        theta = [np.arctan2(p2.y - p1.y, p2.x, p1.x) for p1, p2 in zip(xy[:-1], xy[1:])]
        theta.insert(0, 0.0)  # add yaw of 0 to first pose
        theta = np.array(theta)

        poses = np.hstack(xy[-1, 1], theta[-1, 1])
        return poses
