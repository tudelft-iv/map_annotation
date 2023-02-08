import pyproj
import utm
import math

import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt

from transforms import CoordTransformer
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiPoint
import shapely.geometry as shapely


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
            type = geometry_data["type"]

            self.polygon = Polygon(
                element_id, road_type, allowed_agents, geometry, type
            )
            self.polygons[element_id] = self.polygon

        return self

    def __getitem__(self, idx):
        return self.polygons[idx]

    def get_frame_location(self, target_agent_id, global_pose, map_extent, yaw_angle):
        """
        Determine bounding box around a target_agent with the region of interest of a given scene.
        """
        x, y, yaw_agent = global_pose[0], global_pose[1], global_pose[2]

        theta = yaw_angle

        top_left = [x + map_extent[0], y + map_extent[3]]
        bottom_left = [x + map_extent[0], y + map_extent[2]]
        bottom_right = [x + map_extent[1], y + map_extent[2]]
        top_right = [x + map_extent[1], y + map_extent[3]]

        rectangle = [top_left, top_right, bottom_right, bottom_left]

        if target_agent_id != "0":
            rectangle_rotated = [
                self.rotate_point(point, global_pose[:2], -theta) for point in rectangle
            ]
            rectangle_rotated = [
                self.rotate_point(point, global_pose[:2], yaw_agent - theta)
                for point in rectangle_rotated
            ]
            # rectangle_rotated = [self.rotate_point(point, global_pose[:2], np.pi/2) for point in rectangle_rotated]
        else:
            rectangle_rotated = [
                self.rotate_point(point, global_pose[:2], -theta) for point in rectangle
            ]

        x_min, y_min = np.min(rectangle_rotated, axis=0)
        x_max, y_max = np.max(rectangle_rotated, axis=0)

        bounding_box = np.array([[x_min, x_max], [y_min, y_max]])

        box = shapely.Polygon(rectangle_rotated)

        return bounding_box, box

    def rotate_point(self, point, origin, angle):

        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

        return qx, qy

    def get_polygons_in_box(
        self, polygons, target_agent_id, global_pose, map_extent, yaw_angle, frame="utm"
    ):
        if frame == "lon-lat":
            pass

        return self._get_polygons_in_box(
            polygons, target_agent_id, global_pose, map_extent, yaw_angle
        )

    def _get_polygons_in_box(
        self, polygons, target_agent_id, global_pose, map_extent, yaw_angle
    ):
        _, box = self.get_frame_location(
            target_agent_id, global_pose, map_extent, yaw_angle
        )

        # x_min = box[0,0]
        # x_max = box[0,1]
        # y_min = box[1,0]
        # y_max = box[1,1]

        polygons_in_box = []
        for id in polygons.element_ids:
            polygon = polygons[id]
            polygon = shapely.Polygon(polygon.bounds.nodes_utm)
            if polygon.intersects(box):
                polygons_in_box.append(id)

            # for node in polygon.bounds.nodes_utm:
            #     if id in polygons_in_box:
            #         continue
            #     if (x_min <= node[0] < x_max) & (y_min <= node[1] < y_max):
            #         polygons_in_box.append(id)

        # for element_id in polygons_in_box:
        #     plt.plot(polygons[element_id].bounds.nodes_utm[:,0], polygons[element_id].bounds.nodes_utm[:,1])

        return polygons_in_box


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
    def __init__(self, element_id, road_type, allowed_agents, geometry, type):
        self.id = element_id
        self.type = road_type
        self.allowed_agents = allowed_agents
        self.geometry = geometry
        self.type = type
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


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    # data['type'] = 'intersection'
    # data2['type'] = 'crosswalk'
    # #print(data)

    # polygons = data.append(data2)

    # polygons.to_file('polygons.gpkg', driver='GPKG', layer='polygons')

    # intersections = Polygons().from_df(data)
    # crosswalks = Polygons().from_df(data2)

    # print(intersections.element_ids)
    # print(crosswalks.element_ids)

    # print(polygons.element_ids)
    # Global coordinates of the prius at frame location within lanes file
    global_pose = 593201.79861197, 5763099.17704595
    # print(nodes)
