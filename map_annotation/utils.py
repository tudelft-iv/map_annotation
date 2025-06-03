import itertools

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import splev, splprep
from shapely.geometry import LineString, Point, Polygon

from map_annotation.interpolation import (get_unique_nodes, interpolate,
                                          make_parametric_path)


def get_min_max(center_coords, side_lengths):
    center_coords = np.asarray(center_coords)
    side_lengths = np.asarray(side_lengths)

    # Calculate the minimum and maximum coordinates of the cuboid
    min_coords = center_coords - side_lengths / 2
    max_coords = center_coords + side_lengths / 2

    return min_coords, max_coords


def create_cuboid(center_coords, side_lengths):
    min_coords, max_coords = get_min_max(center_coords, side_lengths)

    # Create the N-dimensional cuboid using meshgrid
    return list(itertools.product(*zip(min_coords, max_coords)))


# def parametric_spline(path_xy, kind):
#     path_t = np.linspace(0, 1, path_xy.size // 2)
#     spline = interp1d(path_t, path_xy.T, kind=kind)
#     return spline, path_t


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


def check_lane_direction(left_lane, right_lane, yaw_diff_threshold):
    start_left, end_left = np.array(left_lane[0]), np.array(left_lane[-1])
    start_right, end_right = np.array(right_lane[0]), np.array(right_lane[-1])

    left_lane_vector = end_left - start_left
    right_lane_vector = end_right - start_right

    cos_sim = (left_lane_vector @ right_lane_vector.T) / (
        norm(left_lane_vector) * norm(right_lane_vector)
    )

    return cos_sim < 0


def get_yaw_diff(line1, line2):
    vector1 = np.array(line1[-1] - line1[-2])
    vector2 = np.array(line2[-1] - line2[-2])

    vector1 /= norm(vector1)
    vector2 /= norm(vector2)

    assert vector1.shape == vector2.shape
    assert len(vector1.shape) == 1
    assert vector1.shape[0] in [2, 3]

    if vector1.shape[0] == 2:
        vector1 = np.append(vector1, 0)
        vector2 = np.append(vector2, 0)

    plane_normal = np.array([0, 0, 1])

    dot_product = np.dot(vector1, vector2)
    # print(vector1)
    # print(vector2)
    cross_product = np.cross(vector1, vector2)
    # print(cross_product)
    # print(dot_product)
    yaw_diff = np.arctan2(
        np.dot(cross_product, plane_normal),
        dot_product,
    )
    # print(yaw_diff)
    # exit()
    return yaw_diff


def polygon_intersection(coords1, coords2):
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)

    assert len(coords1.shape) == 2
    assert len(coords2.shape) == 2
    assert coords1.shape[1] == 2
    assert coords2.shape[1] == 2

    poly1 = Polygon(coords1)
    poly2 = Polygon(coords2)

    try:
        intersection = poly1.intersection(poly2)
        intersection_coords = list(intersection.exterior.coords)
    except Exception:
        return np.array([])

    return np.array(intersection_coords)


def remove_points_outside_polygon(points, polygon):
    # Remove points not located within the reference intersection
    pop = []

    for idx, point in enumerate(points):
        point = Point(point)
        if not polygon.contains(point):
            pop.append(idx)

    pop.reverse()

    for to_pop in pop:
        points.pop(to_pop)

    return points


def plot_connector(line1, line2, connector, polygon):
    connector = np.array(connector)
    x, y = connector[:, 0], connector[:, 1]

    fig = plt.figure()
    ax = fig.gca()
    patch = mpatches.Polygon(polygon.bounds.nodes, color=(0, 0, 1), alpha=0.3)
    ax.add_patch(patch)
    ax.plot(x, y, c="r", zorder=100)
    ax.plot(line1[:, 0], line1[:, 1])
    ax.plot(line2[:, 0], line2[:, 1])
    plt.show()


def list_to_str(lst):
    return ", ".join([str(x) for x in lst])


def parse_str_to_list(string):
    return [
        value.strip()
        for value in string.split(",")
        if not value.isspace() and value != ""
    ]


def calculate_connector_geometry(
    lane, other_lane, polygon, plot=False, **interp_kwargs
):
    num_interp_pts = 20
    order = 3

    (
        conn_pts_1,
        conn_pts_2,
    ) = determine_connection_points(lane, other_lane, polygon.bounds.geometry)

    connection_line = np.concatenate((conn_pts_1, conn_pts_2), axis=0)
    nodes = get_unique_nodes(connection_line)
    if len(nodes) <= order:
        # there must be at least order+1 nodes to fit a spline,
        # so linearly interpolate to add some additional nodes
        nodes = interpolate(nodes, n_points=order + 1, order=1)

    u = make_parametric_path(nodes)
    u_1, u_2 = u[: len(conn_pts_1)], u[-len(conn_pts_2) :]

    tck, _ = splprep(nodes.T, u=u, k=order, s=0, **interp_kwargs)
    u_interp = np.linspace(u_1[-1], u_2[0], num_interp_pts)
    points = np.stack(splev(u_interp, tck), axis=-1)

    # poly = np.array(list(polygon.bounds.geometry.coords))
    # plt.plot(poly[:, 0], poly[:, 1])
    # plt.plot(points[:, 0], points[:, 1])
    # centerline1 = lane
    # centerline2 = other_lane
    # plt.plot(centerline1[:, 0], centerline1[:, 1], c="black")
    # plt.plot(centerline2[:, 0], centerline2[:, 1], c="purple")
    # plt.show()

    if plot:
        # for debug purposes
        plot_connector(conn_pts_1, conn_pts_2, points, polygon)

    # Remove points not located within the reference intersection
    # points = remove_points_outside_polygon(points.copy(), polygon)
    return points.tolist()


def get_lane_connections(lanes, polygons, N=20):
    """
    Get all possible connections between lanes over an intersection.
    """
    lane_connectors = []
    for polygon in polygons:
        if polygon.type != "intersection":
            continue

        lanes_on_polygon = []
        for lane in lanes:
            if Polygon(lane.polygon).intersects(polygon.geometry):
                lanes_on_polygon.append(lane)

        for lane in lanes_on_polygon:
            for other_lane in lanes_on_polygon:
                lane_id = lane.id
                other_lane_id = other_lane.id
                if lane_id == other_lane_id:
                    continue

                # calculate centerline and boundaries of connector
                points_center = calculate_connector_geometry(
                    # lane.centerline.nodes, other_lane.centerline.nodes, polygon, True
                    lane.centerline.nodes,
                    other_lane.centerline.nodes,
                    # lane.centerline.discretize(resolution=1.0, s=0),
                    # other_lane.centerline.discretize(resolution=1.0, s=0),
                    polygon,
                )
                points_right = calculate_connector_geometry(
                    lane.right_boundary.nodes,
                    other_lane.right_boundary.nodes,
                    polygon,
                )
                points_left = calculate_connector_geometry(
                    lane.left_boundary.nodes,
                    other_lane.left_boundary.nodes,
                    polygon,
                )
                # points_right = []
                # points_left = []

                # flip left boundary to abide by right-hand rule for polygons
                points_left = points_left[::-1]
                connector_polygon = [*points_right, *points_left, points_right[0]]

                lane_connector = {
                    "connector_id": f"{lane_id}_{other_lane_id}",
                    "source": lane_id,
                    "dest": other_lane_id,
                    "intersection_id": polygon.id,
                    "geometry": LineString(points_center),
                    "left_boundary": list_to_str(points_left),
                    "right_boundary": list_to_str(points_right),
                    "lane_type": lane.type.value,  # TODO account for other_lane type
                    "legal": other_lane_id in lane.successors,
                    "polygon": list_to_str(connector_polygon),
                }
                lane_connectors.append(lane_connector)

    return lane_connectors


def determine_connection_points(line1, line2, polygon):
    start1, end1 = line1[[0, -1]]
    idx1 = np.argmin([polygon.distance(Point(pt)) for pt in [start1, end1]])
    reverse1 = idx1 == 0

    start2, end2 = line2[[0, -1]]
    idx2 = np.argmin([polygon.distance(Point(pt)) for pt in [start2, end2]])
    reverse2 = idx2 == 1

    if reverse1:
        line1 = line1[::-1]

    if reverse2:
        line2 = line2[::-1]

    # Select points to be used in calculating connector
    connection_points1 = line1[-3:]
    connection_points2 = line2[:3]
    # connection_points1 = line1[:]
    # connection_points2 = line2[:]

    return connection_points1, connection_points2

    # def interpolate_lane_connector(x, y, N=20):
    #     # Check for duplicate values
    #     _, idx = np.unique(x, return_index=True)
    #     idx = np.sort(idx)
    #     x = x[idx]
    #     y = y[idx]
    #
    #     # assert len(x) == len(y)
    #
    #     points = np.array([x, y]).T
    #
    #     # Position of points along the line
    #     distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    #     distances = np.insert(distances, 0, 0) / distances[-1]
    #
    #     n = np.linspace(0, 1, N)
    #
    #     if len(points) > 3:
    #         interp_kind = "cubic"
    #     else:
    #         interp_kind = "linear"
    #     interpolator = interp1d(distances, points, kind=interp_kind, axis=0)
    #     connector_nodes = interpolator(n)
    #
    #     out_x = connector_nodes.T[0]
    #     out_y = connector_nodes.T[1]
    #
    #     return out_x, out_y


if __name__ == "__main__":
    import sys

    import geopandas as gpd

    sys.path.append("..")
    from map_annotation.lanes import Lanes
    from map_annotation.polygons import Polygons

    lanes = Lanes().load("../data/processed_new/lanes.gpkg")
    polygons = Polygons().load("../data/processed_new/polygons.gpkg")
    # for polygon in polygons:
    #    print(polygon.geometry)

    # exit()

    thresh = 0.5

    connections = get_lane_connections(lanes, polygons)
    connections_df = gpd.GeoDataFrame(connections)

    print(connections_df)
