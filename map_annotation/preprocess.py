import pyproj
import os
import numpy as np
from numpy.linalg import norm
import geopandas as gpd
import pandas as pd
import fiona

from shapely.geometry import Polygon, LineString, Point
from shapely.ops import substring, nearest_points

import matplotlib.pyplot as plt

from map_annotation.lanes import Lanes
from map_annotation.polygons import Polygons


def check_lane_direction(left_lane, right_lane, yaw_diff_threshold):

    start_left, end_left = np.array(left_lane[0]), np.array(left_lane[-1])
    start_right, end_right = np.array(right_lane[0]), np.array(right_lane[-1])

    left_lane_vector = end_left - start_left
    right_lane_vector = end_right - start_right

    # lane_check = False

    # if np.sign(left_lane_vector[0]) != np.sign(right_lane_vector[0]) or np.sign(left_lane_vector[1]) != np.sign(right_lane_vector[1]):
    #     lane_check = True

    cos_sim = (left_lane_vector @ right_lane_vector.T) / (
        norm(left_lane_vector) * norm(right_lane_vector)
    )

    # Implement yaw threshold check
    # yaw_rb = 0
    # yaw_lb = 0

    # for i in range(len(right_lane) - 1):
    #     yaw_rb += np.arctan2((right_lane[i + 1, 0] - right_lane[i, 0]), (right_lane[i + 1, 1] - right_lane[i, 1]))

    # for i in range(len(left_lane) - 1):
    #     yaw_lb += np.arctan2((left_lane[i + 1, 0] - left_lane[i, 0]), (left_lane[i + 1, 1] - left_lane[i, 1]))

    # avg_yaw_rb = yaw_rb / (len(right_lane) - 1)
    # avg_yaw_lb = yaw_lb / (len(left_lane) - 1)

    # yaw_diff = np.abs(avg_yaw_lb - avg_yaw_rb)

    if cos_sim < 0:
        return True

    return False
    # else:
    #     return False


"""Pre-processing of polygons"""
# Load different polygon types
df1 = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Intersections_b.gpkg')
intersections_b = gpd.GeoDataFrame.explode(df1, index_parts=False)
df2 = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Intersections_h.gpkg')
intersections_h = gpd.GeoDataFrame.explode(df2, index_parts=False)
intersections = pd.concat([intersections_b, intersections_h], ignore_index=True)

df3 = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Crosswalks_b.gpkg')
crosswalks_b = gpd.GeoDataFrame.explode(df3, index_parts=False)
df4 = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Crosswalks_h.gpkg')
crosswalks_h = gpd.GeoDataFrame.explode(df4, index_parts=False)
crosswalks = pd.concat([crosswalks_b, crosswalks_h], ignore_index=True)

df5 = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Offroad_b.gpkg')
offroad_b = gpd.GeoDataFrame.explode(df5, index_parts=False)
df6 = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Offroad_h.gpkg')
offroad_h = gpd.GeoDataFrame.explode(df6, index_parts=False)
offroad = pd.concat([offroad_b, offroad_h], ignore_index=True)

# Assign type to polygon type
intersections["type"] = "intersection"
crosswalks["type"] = "crosswalk"
offroad["type"] = "off_road"

# Create single file with all polygons
polygons = pd.concat([intersections, crosswalks, offroad], ignore_index=True)

# Assign correct element_ids
for idx, id in enumerate(polygons["element_id"]):
    polygons["element_id"][idx] = idx + 1

# Create new file with pre-processed annotations
polygons.to_file("data/polygons_preprocessed.gpkg", driver="GPKG")

"""-------------------------------------------------------------------"""
"""Pre-processing of lanes"""
# Load lane segments
df7 = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Lanes_b.gpkg')
lanes_b = gpd.GeoDataFrame.explode(df7, index_parts=False)
df8 = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Lanes_h.gpkg')
lanes_h = gpd.GeoDataFrame.explode(df8, index_parts=False)
lanes = pd.concat([lanes_b, lanes_h], ignore_index=True)

# Assign correct lane_ids, save lane_ids changelog
lane_ids = list(lanes["lane_id"])
old_ids = list(dict.fromkeys(lane_ids))
new_ids = np.arange(1, len(old_ids) + 1, 1)

for idx, id in enumerate(lanes["lane_id"]):
    i = old_ids.index(id)
    lanes["lane_id"][idx] = new_ids[i]

# Update successor and predecessor labels with new lane_ids
for item in ["successors", "predecessors"]:
    for i, lane_ids in enumerate(lanes[item]):
        if lane_ids:
            lane_numb = lane_ids.split(",")
            updated_ids = []
            for id in lane_numb:
                id = int(id)
                idx = old_ids.index(id)
                updated_ids.append(new_ids[idx])
            lanes[item][i] = str(updated_ids)[1:-1]

# Assign correct element_ids
lanes = lanes.sort_values(by=["lane_id"], ignore_index=True)
for idx, id in enumerate(lanes["element_id"]):
    lanes["element_id"][idx] = idx + 1

# for _, lane in lanes.iterrows():
#     id = lane['lane_id']

#     options = lanes[lanes['lane_id'] == id]

#     for _, lane in options.iterrows():
#         if lane['boundary_left']:
#             left_id = lane['lane_id']
#             left_lane = np.array(lane['geometry'].coords)
#         if lane['boundary_right']:
#             right_id = lane['lane_id']
#             right_lane = np.array(lane['geometry'].coords)

#     if check_lane_direction(left_lane, right_lane, np.pi/2):
#         print(lanes[lanes['element_id'] == left_id]['geometry'])
#         lanes[lanes['element_id'] == left_id]['geometry'] = substring(lanes[lanes['element_id'] == left_id]['geometry'].tolist()[0], 1, 0, normalized=True)
#         print(lanes[lanes['element_id'] == left_id]['geometry'])
#         exit()

# ref_point = Point(left_lane[0])
# if ref_point.distance(Point(right_lane[0])) > ref_point.distance(Point(right_lane[-1])):
#     lanes[lanes['element_id'] == left_id]['geometry'] = substring(lanes[lanes['element_id'] == left_id]['geometry'].tolist()[0], 1, 0, normalized=True)
#     left_lane = np.array(lanes[lanes['element_id'] == left_id]['geometry'].tolist()[0].coords)

# ref_point = Point(left_lane[0])
# if ref_point.distance(Point(right_lane[0])) > ref_point.distance(Point(right_lane[-1])) or check_lane_direction(left_lane, right_lane, 0):
#     print('checks_failed')
#     print(id)
#     print(left_lane, right_lane)
#     exit()


# Duplicate bi-directional lanes, flip the direction and split the predecessors and successors based on the lane annotations
old = []
new = []

for idx, lane in lanes.iterrows():
    if not lane["one-way"]:
        # Select both right and left bound
        id = lane["lane_id"]
        if id not in old:
            if lane["lane_id"] == id and lane["boundary_right"]:
                right_bound_id = lane["element_id"]
                options = lanes[lanes["lane_id"] == id]
                for x, lane in options.iterrows():
                    if lane["boundary_left"]:
                        left_bound_id = lane["element_id"]
            elif lane["lane_id"] == id and lane["boundary_left"]:
                left_bound_id = lane["element_id"]
                options = lanes[lanes["lane_id"] == id]
                for x, lane in options.iterrows():
                    if lane["boundary_right"]:
                        right_bound_id = lane["element_id"]

            # Update the element_ids, lane_ids, one_way and lane directionality of the old and new elements

            right_bound = lanes[lanes["element_id"] == right_bound_id]
            left_bound = lanes[lanes["element_id"] == left_bound_id]
            left_bound_new = right_bound
            right_bound_new = left_bound

            right_bound_new["boundary_right"] = True
            right_bound_new["boundary_left"] = False
            left_bound_new["boundary_left"] = True
            left_bound_new["boundary_right"] = False

            right_bound_new["element_id"] = int(lanes["element_id"].iat[-1]) + 1
            left_bound_new["element_id"] = int(lanes["element_id"].iat[-1]) + 2
            right_bound_new["lane_id"] = int(lanes["lane_id"].iat[-1]) + 1
            left_bound_new["lane_id"] = int(lanes["lane_id"].iat[-1]) + 1

            # right_bound_new['geometry'] = LineString(right_bound_new['geometry'].tolist()[0].coords[::-1])
            # Reverse the bounds and check directionality using the new left lane/old right lane (annotated in driving direction)
            # Is called right_bound as per convention of direction in the dataset, while it actually represents the right bound of the original lane traversed in opposite direction
            left_bound_new["geometry"] = substring(
                left_bound_new["geometry"].tolist()[0], 1, 0, normalized=True
            )
            right_bound_new["geometry"] = substring(
                right_bound_new["geometry"].tolist()[0], 1, 0, normalized=True
            )

            # if check_lane_direction(np.array(left_bound_new['geometry'].tolist()[0].coords), np.array(right_bound_new['geometry'].tolist()[0].coords), np.pi/2):
            #     right_bound_new['geometry'] = substring(right_bound_new['geometry'].tolist()[0], 1, 0, normalized=True)

            ref_point = Point(left_bound_new["geometry"].tolist()[0].coords[0])
            if ref_point.distance(
                Point(right_bound_new["geometry"].tolist()[0].coords[0])
            ) > ref_point.distance(
                Point(right_bound_new["geometry"].tolist()[0].coords[-1])
            ):
                right_bound_new["geometry"] = substring(
                    right_bound_new["geometry"].tolist()[0], 1, 0, normalized=True
                )

            # ref_point = Point(left_bound_new['geometry'].tolist()[0].coords[0])
            # if ref_point.distance(Point(right_bound_new['geometry'].tolist()[0].coords[0])) > ref_point.distance(Point(right_bound_new['geometry'].tolist()[0].coords[-1])):
            #     print('checks_failed')
            #     exit()

            # Adjust one-way label
            lanes["one-way"][right_bound_id - 1] = True
            lanes["one-way"][left_bound_id - 1] = True
            right_bound_new["one-way"] = True
            left_bound_new["one-way"] = True

            # Changelog of successors and predecessors
            old.append(id)
            new.append(right_bound_new["lane_id"].tolist()[0])

            # Append new lanes to the lanes dataframe
            lanes = pd.concat(
                [lanes, right_bound_new, left_bound_new], ignore_index=True
            )

# Add all new successor and predecessor candidate labels based on the new lane copies
for item in ["successors", "predecessors"]:
    for i, lane_element in enumerate(lanes[item]):
        if lane_element:
            lane_numb = lane_element.split(",")
            updated_ids = []
            for id in lane_numb:
                id = int(id)
                updated_ids.append(id)
                if id in old:
                    idx = old.index(id)
                    new_id = new[idx]
                    updated_ids.append(new_id)
            lanes[item][i] = str(updated_ids)[1:-1]

# Adjust all successor and predecessor labels based on distance and directionality
element_ids = intersections["element_id"]

for idx, lane in lanes.iterrows():
    if lane["boundary_right"]:
        lane_id = lane["lane_id"]
        ref_start = Point(lane["geometry"].coords[0])
        ref_end = Point(lane["geometry"].coords[-1])

        predecessor_list = []
        successor_list = []

        # Check lane_id and close intersections
        for x, intersection in intersections.iterrows():
            ref_polygon = Polygon(intersection["geometry"])
            dist_succ = LineString(nearest_points(ref_end, ref_polygon)).length
            dist_pred = LineString(nearest_points(ref_start, ref_polygon)).length

            if dist_pred < 0.5:
                if lane["predecessors"]:
                    predecessors = lane["predecessors"].split(",")
                    for predecessor in predecessors:
                        predecessor_id = int(predecessor)
                        predecessor_lane = lanes[lanes["lane_id"] == predecessor_id]
                        for i, bound in predecessor_lane.iterrows():
                            if bound["boundary_right"]:
                                predecessor_start = Point(bound["geometry"].coords[0])
                                predecessor_end = Point(bound["geometry"].coords[-1])
                                # Check directionality, distance to intersection and distance to other lane end
                                if ref_polygon.distance(predecessor_end) < 0.5:
                                    if ref_start.distance(
                                        predecessor_end
                                    ) < ref_start.distance(predecessor_start):
                                        if predecessor_end.distance(
                                            ref_end
                                        ) > predecessor_end.distance(ref_start):
                                            if predecessor_id not in predecessor_list:
                                                predecessor_list.append(predecessor_id)

            if dist_succ < 0.5:
                if lane["successors"]:
                    successors = lane["successors"].split(",")
                    for successor in successors:
                        successor_id = int(successor)
                        successor_lane = lanes[lanes["lane_id"] == successor_id]
                        for i, bound in successor_lane.iterrows():
                            if bound["boundary_right"]:
                                successor_start = Point(bound["geometry"].coords[0])
                                successor_end = Point(bound["geometry"].coords[-1])
                                # Check directionality, distance to intersection and distance to other lane end
                                if ref_polygon.distance(successor_start) < 0.5:
                                    if ref_end.distance(
                                        successor_start
                                    ) < ref_end.distance(successor_end):
                                        if successor_start.distance(
                                            ref_end
                                        ) < successor_start.distance(ref_start):
                                            if successor_id not in successor_list:
                                                successor_list.append(successor_id)

    # Update the successor and predecessor labels
    for i, lane in lanes[lanes["lane_id"] == lane_id].iterrows():
        element_id = lane["element_id"]
        lanes["predecessors"][element_id - 1] = str(predecessor_list)[1:-1]
        lanes["successors"][element_id - 1] = str(successor_list)[1:-1]

# Run label consistency checks on lane annotations
for col, item in lanes.iterrows():
    if (item["boundary_right"] == True and item["boundary_left"] == True) or (
        item["boundary_right"] == False and item["boundary_left"] == False
    ):
        assert f"Error! Lane has incorrect boundary booleans."
        print(item["element_id"])

    lane_id = item["lane_id"]

    for col, item2 in lanes.iterrows():
        if item["lane_id"] == item2["lane_id"]:
            if item["element_id"] != item2["element_id"]:
                if (
                    item["boundary_left"] == item2["boundary_left"]
                    or item["boundary_right"] == item2["boundary_right"]
                ):
                    print(item["lane_id"], "has inconsistent boundaries")
                if item["one-way"] != item2["one-way"]:
                    print(item["lane_id"], "has inconsistent direction")
                if item["predecessors"] != item2["predecessors"]:
                    print(item["lane_id"], "has inconsistent predecessors")
                if item["successors"] != item2["successors"]:
                    print(item["lane_id"], "has inconsistent successors")
                if not item["geometry"]:
                    print(item["lane_id"], "has no geometry")

print("Done, all good to go!")

# Create new file with pre-processed annotations
lanes.to_file("data/lanes_preprocessed.gpkg", driver="GPKG")


"""-------------------------------------------------------------------"""
"""Pre-processing of lane connectors"""

lanes = gpd.read_file("data/lanes_preprocessed.gpkg")
lanes = gpd.GeoDataFrame.explode(lanes, index_parts=False)
lanes = Lanes().from_df(lanes)

polygons = gpd.read_file("data/polygons_preprocessed.gpkg")
polygons = gpd.GeoDataFrame.explode(polygons, index_parts=False)
polygons = Polygons().from_df(polygons)

lane_connectors = []

for lane_id in lanes.lane_ids:
    successors = lanes[lane_id].successors
    print(lane_id)

    # Select lanes with successors
    if not successors:
        pass
    elif successors is None:
        pass
    else:
        successors = successors.split(",")

        dist_threshold = 0.5
        element_ids = polygons.element_ids

        ref_lane_start = Point(lanes[lane_id].centerline.nodes_utm[0])
        ref_lane_end = Point(lanes[lane_id].centerline.nodes_utm[-1])

        for successor in successors:
            successor = int(successor)
            (
                ref_point,
                connection_points_1,
                connection_points_2,
            ) = lanes.determine_connnection_points(lane_id, ref_lane_end, successor)

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

                        xt, yt = lanes.interpolate_lane_connector(x, y)

                        # Remove points not located within the reference intersection
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

                        # x_val = [i[0] for i in points]
                        # y_val = [i[1] for i in points]

                        # plt.scatter(x_val, y_val, color='g')
                        # plt.plot(x_val, y_val, color='g')
                        # plt.scatter(xt, yt, alpha=0.2)
                        # plt.scatter(connection_points_1[:,0], connection_points_1[:,1], color='r')
                        # plt.scatter(connection_points_2[:,0], connection_points_2[:,1], color='r')
                        # plt.plot(polygons[element_id].bounds.nodes_utm[:,0], polygons[element_id].bounds.nodes_utm[:,1])
                        # plt.show()

                        # exit()
                        if connector_geom:
                            lane_connections = {
                                "connector_id": [f"{lane_id}_{successor}"],
                                "intersection_id": [element_id],
                                "connection_line": [connector_geom],
                            }
                            lane_connectors.append(lane_connections)

lane_connections = pd.DataFrame(
    columns=["connector_id", "intersection_id", "connection_line"]
)

for connector in lane_connectors:
    lane_connection = pd.DataFrame(data=connector)
    lane_connections = pd.concat([lane_connections, lane_connection], ignore_index=True)


lane_connections = gpd.GeoDataFrame(lane_connections, geometry="connection_line")
lane_connections.to_file("data/lane_connectors_preprocessed.gpkg", driver="GPKG")

print("Done")
