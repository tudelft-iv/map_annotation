import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from numpy.linalg import norm
import pyproj
import math

import utm
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiPoint
from shapely.ops import nearest_points, unary_union

from map_annotation.polygons import Polygons
from map_annotation.utils import non_decreasing, non_increasing, monotonic
from map_annotation.transforms import CoordTransformer


class Lanes:
    """
    Lanes class to handle all required lane operations
    """

    def __init__(self):
        """
        Initialize lane geometry
        """
        self.lanes = None

        # Convert coordinates to meters
        self.geod = pyproj.Geod(ellps="WGS84")

    def from_df(self, df):
        """
        Retrieve lane information from labelled data.
        """
        self.lanes = {}
        self.lane_ids = list(set(df["lane_id"].astype("int64").values))

        # Retrieve lane data
        for lane_id in self.lane_ids:
            lane_data = df[df["lane_id"] == lane_id]
            left_bound_data = lane_data[lane_data["boundary_left"]].squeeze().to_dict()
            right_bound_data = (
                lane_data[lane_data["boundary_right"]].squeeze().to_dict()
            )

            left_boundary = RoadLine(
                left_bound_data["lane_id"],
                left_bound_data["road_type"],
                np.array(left_bound_data["geometry"].coords)[:, :2],
            )
            right_boundary = RoadLine(
                right_bound_data["lane_id"],
                right_bound_data["road_type"],
                np.array(right_bound_data["geometry"].coords)[:, :2],
            )

            predecessors = right_bound_data["predecessors"]
            successors = right_bound_data["successors"]
            allowed_agents = right_bound_data["allowed_agents"]

            self.lane = Lane(
                lane_id,
                left_boundary,
                right_boundary,
                predecessors,
                successors,
                allowed_agents,
            )
            self.lanes[lane_id] = self.lane

        return self

    def __getitem__(self, idx):
        return self.lanes[idx]

    def get_frame_location(self, target_agent_id, global_pose, map_extent, yaw_angle):
        """
        Determine bounding box around a target_agent with the region of interest of a given scene.
        """
        x, y, yaw_agent = global_pose[0], global_pose[1], global_pose[2]

        theta = yaw_angle
        # angle = yaw_agent - theta

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

    def get_lanes_in_box(
        self, lanes, target_agent_id, global_pose, map_extent, yaw_angle, frame="utm"
    ):
        """
        Select the lanes that are within the specified bounding box.
        """
        if frame == "lon-lat":
            raise ValueError(f"Frame {frame} not implemented.")
        lanes = self.lanes

        return self._get_lanes_in_box(
            lanes, target_agent_id, global_pose, map_extent, yaw_angle
        )

    def _get_lanes_in_box(
        self, lanes, target_agent_id, global_pose, map_extent, yaw_angle
    ):
        """
        Select the lanes that are within the specified bounding box.
        """

        _, box = self.get_frame_location(
            target_agent_id, global_pose, map_extent, yaw_angle
        )

        # Checks for lanes that geometrically match the region of the frame of interest
        lanes_in_box = []

        for lane_id in lanes:
            lane = lanes[lane_id]
            line = LineString(lane.centerline.nodes_utm)

            if line.intersects(box):
                lanes_in_box.append(lane_id)

        return lanes_in_box

    def get_lane_connections(
        self, lane_id, polygons, global_pose, map_extent, dist_thres
    ):
        """
        Retrieve lane connections between lanes that connect with a specified lane of interest.
        """
        # Retrieve successors of a given lane
        successors = self.lanes[lane_id].successors

        # Select lanes with successors
        if successors is None or not successors:
            return None

        successors = successors.split(",")

        lane_connectors = []
        dist_threshold = dist_thres
        element_ids = polygons.get_polygons_in_box(
            polygons, global_pose, map_extent, frame="utm"
        )

        ref_lane_start = Point(self.lanes[lane_id].centerline.nodes_utm[0])
        ref_lane_end = Point(self.lanes[lane_id].centerline.nodes_utm[-1])

        for successor in successors:
            successor = int(successor)
            (
                ref_point,
                connection_points_1,
                connection_points_2,
            ) = self.determine_connnection_points(lane_id, ref_lane_end, successor)

            for element_id in element_ids:
                # Filter polygons to intersections only
                if polygons[element_id].type.tolist()[0] == "intersection":
                    # Determines whether intersection geomatches final lane node
                    ref_polygon = Polygon(polygons[element_id].bounds.nodes_utm)
                    dist = LineString(nearest_points(ref_point, ref_polygon)).length

                    # if lane end is within threshold from an intersection
                    if dist < dist_threshold:
                        connection_line = np.concatenate(
                            (connection_points_1, connection_points_2), axis=0
                        )

                        x = np.asarray([i[0] for i in connection_line])
                        y = np.asarray([i[1] for i in connection_line])

                        xt, yt = self.interpolate_lane_connector(x, y)

                        # Remove points not located within the refernce intersection
                        points = list(zip(xt, yt))
                        pop = []

                        for idx, point in enumerate(points):
                            point = Point(point)
                            # Use distance function as contain/within methods have rounding errors
                            if point.distance(ref_polygon) > 1e-3:
                                pop.append(idx)

                        pop.reverse()

                        for to_pop in pop:
                            points.pop(to_pop)

                        connector_geom = LineString(points)

                        lane_connections = {
                            "connector_id": f"{lane_id}_{successor}",
                            "intersection_id": element_id,
                            "connection_line": connector_geom,
                        }
                        lane_connectors.append(lane_connections)

        return lane_connectors

    def determine_connnection_points(self, lane_id, ref_lane_end, successor):
        connection_points_1 = self.lanes[lane_id].centerline.nodes_utm[-3:]
        connection_points_2 = self.lanes[successor].centerline.nodes_utm[:3]

        ref_point = ref_lane_end

        return ref_point, connection_points_1, connection_points_2

    def discretize_connectors(self, lane_connectors, polyline_resolution):
        pose_lists = []
        ids = []

        # print(lane_connectors)
        for list_connector in lane_connectors:
            if list_connector is None:
                pass
            else:
                for connector in list_connector:
                    id = connector["connector_id"]
                    lane = connector["connection_line"]
                    discretized_lane = self.discretize_lane(
                        id, lane, polyline_resolution, lanebool=False
                    )
                    pose_lists.append(discretized_lane)
                    ids.append(id)

        return pose_lists

    def discretize_lanes(self, lane_ids, polyline_resolution):

        pose_lists = []

        for id in lane_ids:
            lane = self.lanes[id]
            id = str(id)
            discretized_lane = self.discretize_lane(
                id, lane, polyline_resolution, lanebool=True
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

        return discretization

    def get_path_length(self, line):
        return line.length

    def interpolate_lane_connector(self, x, y):

        # Check for duplicate values
        if len(x) == len(np.unique(x)) and len(y) == len(np.unique(y)):
            pass
        else:
            x = np.unique(x)
            y = np.unique(y)

        points = np.array([x, y]).T

        # Linear length along the line:
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]

        n = np.linspace(0, 1, 75)

        interpolator = interp1d(distance, points, kind="quadratic", axis=0)
        connector_nodes = interpolator(n)

        out_x = connector_nodes.T[0]
        out_y = connector_nodes.T[1]

        return out_x, out_y

    def get_neighbouring_lanes(self, lane_id, d_threshold) -> list:
        """
        Get all neighbouring lane segments of a given lane segment
        :param lane_id: lane identifier for which to execute the operation
        :return left_neighbours: List of lane segment identifiers that are left neighbours
        :return right_neighbours: List of lane segment identifiers that are right neighbours
        """

        # TODO clean function, add functionality of boundary type
        left_line, right_line = (
            self.lanes[lane_id].left_boundary,
            self.lanes[lane_id].right_boundary,
        )

        left_line = LineString(left_line.nodes_utm)
        right_line = LineString(right_line.nodes_utm)

        potential_neighbours = []
        left_neighbours = []
        right_neighbours = []

        for i in range(1, len(self.lanes)):
            if (
                self.lanes[lane_id].left_boundary.type == 2
                or self.lanes[lane_id].right_boundary.type == 2
            ):
                if (
                    self.lanes[i].left_boundary.type == 2
                    or self.lanes[i].right_boundary.type == 2
                ):
                    if (self.lanes[i].id != lane_id) and (
                        self.lanes[i].id not in potential_neighbours
                    ):
                        potential_neighbours.append(self.lanes[i].id)

        for neighbour in potential_neighbours:
            left_line_other, right_line_other = (
                self.lanes[neighbour].left_boundary,
                self.lanes[neighbour].right_boundary,
            )

            left_line_other = LineString(left_line_other.nodes_utm)
            right_line_other = LineString(right_line_other.nodes_utm)

            if (
                self.lanes[lane_id].left_boundary.type == 2
                and self.lanes[neighbour].right_boundary.type == 2
            ):
                distance1 = LineString(
                    nearest_points(left_line, right_line_other)
                ).length

                if distance1 <= d_threshold:
                    left_neighbours.append(neighbour)

            if (
                self.lanes[lane_id].right_boundary.type == 2
                and self.lanes[neighbour].left_boundary.type == 2
            ):
                distance2 = LineString(
                    nearest_points(right_line, left_line_other)
                ).length

                if distance2 <= d_threshold:
                    right_neighbours.append(neighbour)

        return left_neighbours, right_neighbours


class RoadLine:
    def __init__(self, boundary_id, boundary_type, nodes):
        self.id = boundary_id
        self.type = boundary_type
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


class Lane:
    def __init__(
        self,
        lane_id,
        left_boundary,
        right_boundary,
        predecessors,
        successors,
        allowed_agents,
        lane_type=None,
    ):
        self.id = lane_id
        self.predecessors = predecessors
        self.successors = successors
        self.allowed_agents = allowed_agents
        self.type = lane_type

        self._left_boundary = left_boundary
        self._right_boundary = right_boundary
        self._centerline = None

    @property
    def left_boundary(self):
        return self._left_boundary

    @left_boundary.setter
    def left_boundary(self, data):
        self._left_boundary = data
        self._centerline = None

    @property
    def right_boundary(self):
        return self._right_boundary

    @right_boundary.setter
    def right_boundary(self, data):
        self._right_boundary = data
        self._centerline = None

    @property
    def centerline(self):
        if self._centerline is None:
            self._centerline = self._calculate_centerline()

        return self._centerline

    def _calculate_centerline(self):
        left_line = self.left_boundary.interpolate(frame="nl")
        right_line = self.right_boundary.interpolate(frame="nl")
        assert len(left_line) == len(
            right_line
        ), "Error! The left and right boundaries do not consist of equal points."

        ref_point = Point(right_line[0])
        # ref_point2 = Point(right_line[-1])

        if ref_point.distance(Point(left_line[0])) > ref_point.distance(
            Point(left_line[-1])
        ):
            left_line = np.flipud(left_line)

        lr_line = np.stack([left_line, right_line], axis=-1)
        midpoints = [
            [(left_coord + right_coord) / 2 for left_coord, right_coord in point]
            for point in lr_line
        ]

        centerline = RoadLine(None, "lane_centerline", np.array(midpoints))

        return centerline


def check_lane_direction(left_lane, right_lane, yaw_diff_threshold):
    start_left, end_left = np.array(left_lane[0]), np.array(left_lane[-1])
    start_right, end_right = np.array(right_lane[0]), np.array(right_lane[-1])

    left_lane_vector = end_left - start_left
    right_lane_vector = end_right - start_right

    cos_sim = (left_lane_vector @ right_lane_vector.T) / (
        norm(left_lane_vector) * norm(right_lane_vector)
    )

    if cos_sim < 0:
        return True

    return False
