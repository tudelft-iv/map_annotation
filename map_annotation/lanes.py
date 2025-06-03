import ast

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from centerline.geometry import Centerline
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, nearest_points
from tqdm import tqdm

from map_annotation.road_elements import (RoadElement, RoadElementCollection,
                                          RoadLine)
from map_annotation.utils import list_to_str, parse_str_to_list


class Lanes(RoadElementCollection):
    """
    Lanes class to handle all required lane operations
    """

    def __init__(self, verbose=False):
        """
        Initialize lane geometry
        """
        super().__init__()
        self.element_ids = None
        self.verbose = verbose

    def from_df(self, df):
        """
        Retrieve lane information from labelled data.
        """
        self.element_ids = list(set(df["element_id"].astype(str).values))

        # Retrieve lane data
        if self.verbose:
            print("Loading map annotations...")

        for idx, lane_data in df.iterrows():
            lane_id = str(lane_data["element_id"])

            road_type = RoadElement.RoadType(int(lane_data["road_type"]))
            allowed_agents = RoadElement.RoadType_AllowedAgents_map[road_type]

            lbound, rbound = ast.literal_eval(
                lane_data["boundary_left"]
            ), ast.literal_eval(lane_data["boundary_right"])

            polygon = np.array(list(lane_data["geometry"].exterior.coords))

            left_boundary = RoadLine(
                lane_id,
                np.array(lbound),
                line_type=RoadLine.LineType.BOUNDARY,
                road_type=road_type,
            )
            right_boundary = RoadLine(
                lane_id,
                np.array(rbound),
                line_type=RoadLine.LineType.BOUNDARY,
                road_type=road_type,
            )

            predecessors = parse_str_to_list(lane_data["predecessors"])
            successors = parse_str_to_list(lane_data["successors"])

            self.elements[lane_id] = Lane(
                lane_id,
                left_boundary,
                right_boundary,
                predecessors,
                successors,
                allowed_agents,
                road_type,
                polygon=polygon,
            )

        if self.verbose:
            print("Done.")

        return self

    def from_list(self, lanes):
        for lane in lanes:
            self.elements[lane.id] = lane

        return self

    def to_df(self, crs="EPSG:28992"):
        return gpd.GeoDataFrame([lane.to_dict() for lane in self], crs=crs)

    def _get_clines(self):
        clines = {
            lane_id: lane.centerline.nodes for lane_id, lane in self.elements.items()
        }
        return clines

    def get_lanes_in_circle(self, centre, radius):
        """
        Select the lanes that are within the specified circle.
        """
        clines = self._get_clines()
        circle = Point(centre).buffer(radius)
        return self._get_lanes_in_geometry(clines, circle)

    def get_lanes_in_box(self, box):
        """
        Select the lanes that are within the specified bounding box.
        :param box: array of bounds in the form of [xmin, ymin, xmax, ymax]
        """
        clines = self._get_clines()
        box = shapely.geometry.box(*box)
        return self._get_lanes_in_geometry(clines, box)

    @staticmethod
    def _get_lanes_in_geometry(lanes, geometry):
        """
        Select the lanes that are within the specified geometry.
        """
        lanes_in_box = []
        for lane_id, lane_line in lanes.items():
            if LineString(lane_line).intersects(geometry):
                # note: this tests whether any part of the line intersects with the box AREA
                lanes_in_box.append(lane_id)

        return lanes_in_box

    def get_lanes_by_ids(self, ids):
        return [self.elements[lane_id] for lane_id in ids]

    def discretize_by_ids(self, lane_ids, resolution):
        """Discretize lanes with given ids"""
        return [self.lanes[lane_id].discretize(resolution) for lane_id in lane_ids]

    def discretize(self, resolution):
        """Discretize all lanes"""
        return [lane.discretize(resolution) for lane in self.lanes]

    def get_neighbouring_lanes_by_id(self, lane_id, d_threshold) -> tuple[list, list]:
        """
        Get all neighbouring lane segments of a given lane segment
        :param lane_id: lane identifier for which to execute the operation
        :return left_neighbours: List of lane segment identifiers that are left neighbours
        :return right_neighbours: List of lane segment identifiers that are right neighbours
        """
        # TODO re-think function - what is a neighbour, and how should these be found?
        #
        # Definition: lanes are neighbouring if a lane changes is legal and possible
        # between them.
        # Conditions for neighbours:
        # - In close proximity
        # - Dashed line between them
        lane = self.lanes[lane_id]

        return self.get_neighbouring_lanes(lane, d_threshold)

    def get_neighbouring_lanes(self, lane, d_threshold) -> tuple[list, list]:
        left_line, right_line = (
            lane.left_boundary,
            lane.right_boundary,
        )

        left_line = LineString(left_line.nodes_utm)
        right_line = LineString(right_line.nodes_utm)

        potential_neighbours = []
        left_neighbours = []
        right_neighbours = []

        # TODO simplify logic and contract ifs
        # TODO narrow search to nearby lanes
        for i in range(len(self.elements)):
            if (
                lane.left_boundary.boundary_type == RoadLine.BoundaryType.DASHED
                or lane.right_boundary.boundary_type == RoadLine.BoundaryType.DASHED
            ):
                if (
                    self.elements[i].left_boundary.boundary_type
                    == RoadLine.BoundaryType.DASHED
                    or self.elements[i].right_boundary.boundary_type
                    == RoadLine.BoundaryType.DASHED
                ):
                    if (self.elements[i].id != lane.id) and (
                        self.elements[i].id not in potential_neighbours
                    ):
                        potential_neighbours.append(self.elements[i].id)

        for neighbour in potential_neighbours:
            left_line_other, right_line_other = (
                self.elements[neighbour].left_boundary,
                self.elements[neighbour].right_boundary,
            )

            left_line_other = LineString(left_line_other.nodes_utm)
            right_line_other = LineString(right_line_other.nodes_utm)

            if (
                lane.left_boundary.boundary_type == RoadLine.BoundaryType.DASHED
                and self.elements[neighbour].right_boundary.boundary_type
                == RoadLine.BoundaryType.DASHED
            ):
                distance1 = LineString(
                    nearest_points(left_line, right_line_other)
                ).length

                if distance1 <= d_threshold:
                    left_neighbours.append(neighbour)

            if (
                lane.right_boundary.boundary_type == RoadLine.BoundaryType.DASHED
                and self.elements[neighbour].left_boundary.boundary_type
                == RoadLine.BoundaryType.DASHED
            ):
                distance2 = LineString(
                    nearest_points(right_line, left_line_other)
                ).length

                if distance2 <= d_threshold:
                    right_neighbours.append(neighbour)

        return left_neighbours, right_neighbours


def keep_unique(ls):
    points, idx = np.unique(ls, axis=0, return_index=True)
    return points[np.argsort(idx), :]


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
        polygon=None,
        centerline=None,
    ):
        self.id = lane_id
        self.predecessors = predecessors
        self.successors = successors
        self.allowed_agents = allowed_agents
        self.type = lane_type

        self._left_boundary = left_boundary
        self._right_boundary = right_boundary
        self._polygon = polygon
        self._centerline = centerline
        self._start_line = None
        self._end_line = None

    @property
    def left_boundary(self):
        return self._left_boundary

    @left_boundary.setter
    def left_boundary(self, data):
        self._left_boundary = data
        self._centerline = None
        self._polygon = None
        self._start_line = None
        self._end_line = None

    @property
    def right_boundary(self):
        return self._right_boundary

    @right_boundary.setter
    def right_boundary(self, data):
        self._right_boundary = data
        self._centerline = None
        self._polygon = None
        self._start_line = None
        self._end_line = None

    @property
    def polygon(self):
        if self._polygon is None:
            self._polygon = self._get_polygon()
        return self._polygon

    @property
    def centerline(self):
        if self._centerline is None:
            self._centerline = self._calculate_centerline()

        return self._centerline

    def _calculate_centerline(self):
        left_line = self.left_boundary.interp1d(n_points=40)
        right_line = self.right_boundary.interp1d(n_points=40)
        assert len(left_line) == len(
            right_line
        ), "The left and right boundaries do not consist of equal points."

        # the left boundary is annotated in the reverse direction of the lane
        # and therefore needs to be flipped
        left_line = np.flipud(left_line)

        lr_line = np.stack([left_line, right_line], axis=-1)
        midpoints = np.array(
            [
                [(left_coord + right_coord) / 2 for left_coord, right_coord in point]
                for point in lr_line
            ]
        )

        # plt.scatter(left_line[:, 0], left_line[:, 1])
        # plt.scatter(right_line[:, 0], right_line[:, 1])
        # plt.plot(left_line[:, 0], left_line[:, 1])
        # plt.plot(right_line[:, 0], right_line[:, 1])
        # plt.plot(midpoints[:, 0], midpoints[:, 1])
        # for line in lr_line:
        #    plt.plot(line[0], line[1])
        # plt.show()

        centerline = RoadLine(
            self.id,
            np.array(midpoints),
            road_type=RoadLine.LineType.CENTERLINE,
        )

        return centerline

    @property
    def start_line(self):
        if self._start_line is None:
            self._start_line, self._end_line = self._compute_start_end_lines()
        return self._start_line

    @property
    def end_line(self):
        if self._end_line is None:
            self._start_line, self._end_line = self._compute_start_end_lines()
        return self._end_line

    def _compute_start_end_lines(self):
        # Note: the left boundary is annotated in the reverse direction of the lane
        left_line = self.left_boundary.nodes
        right_line = self.right_boundary.nodes

        # these lines follow the right-hand rule i.e. are anti-clockwise
        start_line_coords = (left_line[-1], right_line[0])
        end_line_coords = (right_line[-1], left_line[0])

        start_line = RoadLine(
            self.id,
            np.array(start_line_coords),
            road_type=RoadLine.LineType.START_LINE,
        )

        end_line = RoadLine(
            self.id,
            np.array(end_line_coords),
            road_type=RoadLine.LineType.END_LINE,
        )

        return start_line, end_line

    def discretize(self, resolution):
        return self.centerline.discretize(resolution)

    def _get_polygon(self):
        """
        Calculate lane polygon based on lane boundaries
        """
        lbound = self.left_boundary.nodes
        rbound = self.right_boundary.nodes

        # combine lane bound nodes to make polygon (final elem for closure)
        polygon = np.vstack([rbound, lbound, rbound[0]])
        return polygon

    def to_dict(self):
        lane_dict = {
            "element_id": self.id,
            "road_type": self.type.value,
            "lane_id": self.id,
            "from_object": None,
            "to_object": None,
            "connects_to": None,
            "successors": list_to_str(self.successors),
            "predecessors": list_to_str(self.predecessors),
            "boundary_left": list_to_str(list(self.left_boundary.nodes)),
            "boundary_right": list_to_str(list(self.left_boundary.nodes)),
            "geometry": Polygon(self.polygon),
        }
        return lane_dict
