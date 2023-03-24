import numpy as np
import shapely
from numpy.linalg import norm
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points
from tqdm import tqdm

from map_annotation.road_elements import RoadElementCollection, RoadLine


class Lanes(RoadElementCollection):
    """
    Lanes class to handle all required lane operations
    """

    def __init__(self):
        """
        Initialize lane geometry
        """
        super().__init__()
        self.element_ids = None

    def from_df(self, df):
        """
        Retrieve lane information from labelled data.
        """
        self.element_ids = list(set(df["lane_id"].astype("int64").values))

        # Retrieve lane data
        print("Loading map annotations...")
        mismatched_ids = {}
        for lane_id in tqdm(self.element_ids):

            lane_data = df[df["lane_id"] == lane_id]
            left_bound_data = lane_data[lane_data["boundary_left"]].squeeze().to_dict()
            right_bound_data = (
                lane_data[lane_data["boundary_right"]].squeeze().to_dict()
            )

            road_type_left = left_bound_data["road_type"]
            road_type_right = right_bound_data["road_type"]
            if road_type_left != road_type_right:
                # log lane id and choose the left type for now
                mismatched_ids[lane_id] = (road_type_left, road_type_right)

                # raise ValueError(
                #    f"Boundary road types for lane '{lane_id}' do not match. Road types: {road_type_left}, {road_type_right} (L,R)."
                # )
            road_type = left_bound_data["road_type"]

            left_boundary = RoadLine(
                left_bound_data["lane_id"],
                np.array(left_bound_data["geometry"].coords)[:, :2],
                road_type=road_type,
            )
            right_boundary = RoadLine(
                right_bound_data["lane_id"],
                np.array(right_bound_data["geometry"].coords)[:, :2],
                road_type=road_type,
            )

            predecessors = right_bound_data["predecessors"]
            successors = right_bound_data["successors"]
            allowed_agents = right_bound_data["allowed_agents"]

            self.elements[lane_id] = Lane(
                lane_id,
                left_boundary,
                right_boundary,
                predecessors,
                successors,
                allowed_agents,
                road_type,
            )

        print("Mismatched boundary types: ", mismatched_ids)
        return self

    def get_lanes_in_box(self, box, frame="utm"):
        """
        Select the lanes that are within the specified bounding box.
        :param box: array of bounds in the form of [xmin, ymin, xmax, ymax]
        :param frame: frame to compute intersection in
        """
        clines = {
            lane_id: lane.centerline.get_attr_in_frame("nodes", frame)
            for lane_id, lane in self.elements.items()
        }
        box = shapely.geometry.box(*box)
        return self._get_lanes_in_box(clines, box)

    @staticmethod
    def _get_lanes_in_box(lanes, box):
        """
        Select the lanes that are within the specified bounding box.
        """
        lanes_in_box = []
        for lane_id, lane_line in lanes.items():
            if LineString(lane_line).intersects(box):
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
            # the left boundary is annotated in the reverse direction of the lane
            # and therefore needs to be flipped
            left_line = np.flipud(left_line)

        lr_line = np.stack([left_line, right_line], axis=-1)
        midpoints = [
            [(left_coord + right_coord) / 2 for left_coord, right_coord in point]
            for point in lr_line
        ]

        centerline = RoadLine(
            self.id,
            np.array(midpoints),
            road_type=RoadLine.LineType.CENTERLINE,
        )

        return centerline

    def discretize(self, resolution):
        return self.centerline.discretize(resolution)

    def get_polygon(self, frame="utm"):
        """
        Calculate lane polygon based on lane boundaries
        """
        lbound = self.left_boundary.get_attr_in_frame("nodes", frame)
        rbound = self.right_boundary.get_attr_in_frame("nodes", frame)

        # check directions of bounds
        # they need to be in oppposite directions for this to work
        # use right boundary as guide
        ref = rbound[-1]

        start_dist = np.sum((lbound[0] - ref) ** 2)
        end_dist = np.sum((lbound[-1] - ref) ** 2)

        if end_dist < start_dist:
            # the left bound needs to be reversed to make a polygon
            lbound = lbound[::-1]

        # combine lane bound nodes to make polygon (final elem for closure)
        polygon = np.vstack([rbound, lbound, rbound[0]])
        return polygon

    def get_direction(self):
        raise NotImplementedError


def check_lane_direction(left_lane, right_lane, yaw_diff_threshold):
    start_left, end_left = np.array(left_lane[0]), np.array(left_lane[-1])
    start_right, end_right = np.array(right_lane[0]), np.array(right_lane[-1])

    left_lane_vector = end_left - start_left
    right_lane_vector = end_right - start_right

    cos_sim = (left_lane_vector @ right_lane_vector.T) / (
        norm(left_lane_vector) * norm(right_lane_vector)
    )

    return cos_sim < 0
