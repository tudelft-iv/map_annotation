# from fiona import bounds
import pyproj
import utm

import numpy as np
import geopandas as gpd
from map_annotation.transforms import CoordTransformer

from shapely.geometry import Polygon
import shapely


class Polygons:
    """
    Polygon class to handle all required polygon map operations
    """

    def __init__(self):
        """
        Initialize Polygon geometry
        """
        self.polygon = None
        self.geod = pyproj.Geod(ellps="WGS84")  # Convert coordinates to meters

    def from_df(self, df):
        self.polygons = {}
        self.element_ids = list(set(df["element_id"].astype("int64").values))

        for element_id in self.element_ids:
            geometry_data = df[df["element_id"] == element_id]

            # nodes = np.array((geometry_data['geometry'][0]).exterior.coords)
            road_type = geometry_data["road_type"]
            allowed_agents = geometry_data["allowed_agents"]
            geometry = geometry_data["geometry"]

            self.polygon = Polygon(element_id, road_type, allowed_agents, geometry)
            self.polygons[element_id] = self.polygon

        return self

    def __getitem__(self, idx):
        return self.polygons[idx]

    def get_polygons_in_box(self, global_pose, map_extent, frame="utm"):
        if frame == "lon-lat":
            pass
        polygons = self.polygons

        self._get_polygons_in_box(polygons, global_pose, map_extent)

    def _get_polygons_in_box(self, polygons, global_pose, map_extent):

        box = np.array(
            [
                [global_pose[0] - map_extent, global_pose[0] + map_extent],
                [global_pose[1] - map_extent, global_pose[1] + map_extent],
            ]
        )

        x_min = box[0, 0]
        x_max = box[0, 1]
        y_min = box[1, 0]
        y_max = box[1, 1]

        polygons_in_box = []
        for element_id in polygons:
            polygon = polygons[element_id]
            for node in polygon.nodes:
                if element_id in polygons_in_box:
                    continue
                if (x_min <= node[0] < x_max) & (y_min <= node[1] < y_max):
                    polygons_in_box.append(element_id)

        polygons = [self.polygons[polygon] for polygon in polygons_in_box]

        return polygons


class Bounds:
    def __init__(self, element_id, road_type, nodes):
        self.element_id = element_id
        self.road_type = road_type
        self.nodes = nodes

    @property
    def nodes_utm(self):

        lon, lat = self.nodes_lonlat
        nodes_utm = utm.from_latlon(lat, lon)
        self.utm_zone = nodes_utm[2:]

        return np.stack(nodes_utm[:2], axis=-1)

    @property
    def nodes_lonlat(self):
        # print(self.nodes)
        trans = CoordTransformer()
        nodes_global = list(trans.t_global_nl(self.nodes[:, 0], self.nodes[:, 1]))
        # print(nodes_global)
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
        # print(self.nodes)
        path_t = np.linspace(0, 1, nodes.size // 2)

        path_x = nodes[:, 0]
        path_y = nodes[:, 1]
        # print(path_x.shape)
        # print(path_y.shape)
        r = nodes.T
        # print(r.shape)
        # print(path_t.shape)
        spline = interp1d(path_t, r, kind=kind)

        t = np.linspace(np.min(path_t), np.max(path_t), n_points)
        r = spline(t)

        return r.T


class Polygon:
    def __init__(self, element_id, road_type, allowed_agents, geometry):
        self.id = element_id
        self.type = road_type
        self.allowed_agents = allowed_agents
        self.geometry = geometry
        self._bounds = None

    @property
    def bounds(self):
        if self._bounds is None:
            self._bounds = self._calculate_bounds()

        return self._bounds

    def _calculate_bounds(self):
        poly = self.geometry.tolist()
        bounds_x, bounds_y = np.array(poly[0].exterior.coords.xy)
        nodes = np.stack((bounds_x, bounds_y), axis=-1)

        bounds = Bounds(self.id, self.type, nodes)

        return bounds

