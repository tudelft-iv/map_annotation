import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pyproj
import math

import utm
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiPoint
from shapely.ops import nearest_points, unary_union

from map_annotation.polygons import Polygons
from map_annotation.lanes import Lanes
from map_annotation.transforms import CoordTransformer


class LaneConnectors:
    """
    Lanes class to handle all required lane operations
    """

    def __init__(self):
        """
        Initialize lane geometry
        """
        self.connectors = None

        # Convert coordinates to meters
        self.geod = pyproj.Geod(ellps="WGS84")

    def from_df(self, df):
        """
        Retrieve lane information from labelled data.
        """
        self.connectors = {}
        self.connector_ids = list(set(df["connector_id"]))
        # Retrieve lane data
        for connector_id in self.connector_ids:
            connector_data = df[df["connector_id"] == connector_id]
            # centerline = connector_data[connector_data['geometry']].squeeze().todict()
            intersection_id = connector_data["intersection_id"]

            connector = RoadLine(
                np.array(connector_data["geometry"].tolist()[0].coords)[:, :2]
            )

            self.connector = Connector(connector_id, intersection_id, connector)
            self.connectors[connector_id] = self.connector

        return self

    def __getitem__(self, idx):
        return self.connectors[idx]

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

        box = Polygon(rectangle_rotated)

        return bounding_box, box

    def rotate_point(self, point, origin, angle):

        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

        return qx, qy

    def get_connectors_in_box(
        self,
        connectors,
        target_agent_id,
        global_pose,
        map_extent,
        yaw_angle,
        frame="utm",
    ):
        """
        Select the lanes that are within the specified bounding box.
        """
        if frame == "lon-lat":
            pass
        connectors = self.connectors

        return self._get_connectors_in_box(
            connectors, target_agent_id, global_pose, map_extent, yaw_angle
        )

    def _get_connectors_in_box(
        self, connectors, target_agent_id, global_pose, map_extent, yaw_angle
    ):
        """
        Select the lanes that are within the specified bounding box.
        """

        _, box = self.get_frame_location(
            target_agent_id, global_pose, map_extent, yaw_angle
        )

        # x_min = box[0,0]
        # x_max = box[0,1]
        # y_min = box[1,0]
        # y_max = box[1,1]

        # Checks for lanes that geometrically match the region of the frame of interest
        connectors_in_box = []

        for connector_id in connectors:
            connector = self.connectors[connector_id]
            line = connector.centerline

            if line.intersects(box):
                connectors_in_box.append(connector_id)

            # for node in lane.centerline.nodes_utm:
            #     if lane_id in lanes_in_box:
            #         pass
            #     elif (x_min <= node[0] < x_max) & (y_min <= node[1] < y_max):
            #        lanes_in_box.append(lane_id)

        return connectors_in_box

    def discretize_connectors(self, connector_ids, polyline_resolution):

        pose_lists = []

        for id in connector_ids:
            connector = self.connectors[id]
            lane = connector.centerline
            discretized_lane = self.discretize_lane(
                id, lane, polyline_resolution, lanebool=False
            )
            pose_lists.append(discretized_lane)

        return pose_lists

    def discretize_lane(self, id, lane, polyline_resolution, lanebool):
        pose_list = []

        if lanebool:
            line = LineString(lane.centerline.nodes_utm)
        else:
            line = lane

        poses = self.discretize(line, polyline_resolution)

        for pose in poses:
            pose_list.append(pose)

        return {id: pose_list}

    def discretize(self, line, polyline_resolution):
        path_length = self.get_path_length(line)

        discretization = []

        n_points = int(max(math.ceil(path_length / polyline_resolution) + 1.5, 2))
        resolution = path_length / (n_points - 1)

        start_pose = line.coords[0]

        for step in range(n_points):
            step_along_path = step * resolution

            new_point = line.interpolate(step_along_path)

            if len(discretization) != 0:
                theta = np.arctan2(
                    (new_point.y - start_pose[1]), (new_point.x - start_pose[0])
                )
            else:
                theta = 0

            new_pose = (new_point.x, new_point.y, theta)
            discretization.append(new_pose)

            start_pose = new_pose

        # x, y, theta = zip(*discretization)
        # plt.scatter(x, y)
        # plt.plot(x,y)
        # plt.scatter(discretization[-1][0], discretization[-1][1])
        # plt.plot(*line.coords.xy)
        # plt.show()

        # print(discretization)

        return discretization

    def get_path_length(self, line):
        return line.length


class RoadLine:
    def __init__(self, nodes):
        self.nodes = nodes

    @property
    def nodes_utm(self):
        # nodes are in (lon, lat) format, so need to be reversed
        lon, lat = self.nodes_lonlat
        nodes_utm = utm.from_latlon(lat, lon)
        self.utm_zone = nodes_utm[2:]
        return np.stack(nodes_utm[:2], axis=-1)

    @property
    def nodes_lonlat(self):
        trans = CoordTransformer()
        nodes_global = list(trans.t_global_nl(self.nodes[:, 0], self.nodes[:, 1]))
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
        path_x = nodes[:, 0]
        path_y = nodes[:, 1]

        r = nodes.T
        spline = interp1d(path_t, r, kind=kind)
        t = np.linspace(np.min(path_t), np.max(path_t), n_points)
        r = spline(t)

        return r.T


class Connector:
    def __init__(self, connector_id, intersection_id, centerline):
        self.id = connector_id
        self.centerline = LineString(centerline.nodes)
        self.intersection_id = intersection_id
