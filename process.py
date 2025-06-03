import argparse
import glob
import json
import os
from collections import defaultdict
from functools import partial
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import (LineString, MultiLineString, MultiPoint, Point,
                              Polygon)
from shapely.ops import nearest_points, snap
from tqdm import tqdm

from map_annotation.lanes import Lanes
from map_annotation.polygons import Polygons
from map_annotation.transforms import CoordTransformer
from map_annotation.utils import get_lane_connections, list_to_str

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", help="path to directory with raw annotations", required=True
)
parser.add_argument(
    "-o", "--output", help="path to store processed annotations", required=True
)

args = parser.parse_args()
input_dir = args.input
output_dir = args.output

Path(output_dir).mkdir(parents=True, exist_ok=True)

element_count = 0

transformer = CoordTransformer()


def make_unique_ids(num, count):
    return np.arange(count + 1, count + num + 1), count + num


def get_and_concat_dfs(input_dir, dfs_str):
    dfs_files = glob.glob(os.path.join(input_dir, dfs_str))
    dfs_dfs = [
        gpd.GeoDataFrame.explode(gpd.read_file(filepath), index_parts=False)
        for filepath in dfs_files
    ]
    dfs = pd.concat(dfs_dfs, ignore_index=True)
    return dfs


def transform_points(points, tfunc):
    return [tfunc(*coord[:2]) for coord in points]


"""Pre-processing of polygons"""
# Load different polygon types
intersections = get_and_concat_dfs(input_dir, "Intersections.gpkg")
offroad = get_and_concat_dfs(input_dir, "Offroad.gpkg")
crosswalks = get_and_concat_dfs(input_dir, "Crosswalks.gpkg")
terminal = get_and_concat_dfs(input_dir, "Terminal.gpkg")


# # fix IDs to be unique
# intersections["element_id"], element_count = make_unique_ids(
#    len(intersections), element_count
# )
# crosswalks["element_id"], element_count = make_unique_ids(
#    len(crosswalks), element_count
# )
# offroad["element_id"], element_count = make_unique_ids(len(offroad), element_count)

# Assign type to polygon type field
intersections["type"] = "intersection"
crosswalks["type"] = "crosswalk"
offroad["type"] = "offroad"
terminal["type"] = "terminal"

# Create single file with all polygons
polygons = pd.concat([intersections, crosswalks, offroad, terminal], ignore_index=True)

# transform geometries to utm frame
polygons["geometry"] = polygons["geometry"].apply(
    lambda x: Polygon(transform_points(x.exterior.coords, transformer.t_global_utm))
)
polygons["element_id"] = polygons["element_id"].astype(str)


# Create new file with pre-processed annotations
polygons.to_file(os.path.join(output_dir, "polygons.gpkg"), driver="GPKG")

"""-------------------------------------------------------------------"""
"""Pre-processing of lanes"""


def parse_object_str(object_str):
    object_mapping = {
        "i": "intersection",
        "o": "offroad",
        "c": "crosswalk",
        "l": "lane",
        "t": "terminal",
    }

    object_tag = object_str[0]
    object_type = object_mapping[object_tag]

    return object_type, object_str[1:]


# Load lane geometries
lanes_files = glob.glob(os.path.join(input_dir, "Lanes.gpkg"))
lanes_dfs = [
    gpd.GeoDataFrame.explode(gpd.read_file(filepath), index_parts=False)
    for filepath in lanes_files
]
lanes = pd.concat(lanes_dfs, ignore_index=True)
lanes["element_id"] = lanes["element_id"].astype(str)

# Load lane attributes
with open(os.path.join(input_dir, "LanesAttributes.json"), "r") as f:
    lanes_attributes = json.load(f)

lanes_attributes = pd.DataFrame.from_dict(lanes_attributes, orient="index")
# print(lanes_attributes[lanes_attributes["lane_id"] == 2])
# exit()
lanes_attributes["lane_id"] = lanes_attributes["lane_id"].astype(str)
lanes_attributes["element_id"] = lanes_attributes["lane_id"].astype(str)
# print(lanes_attributes.duplicated(["element_id"]))
# print(lanes_attributes[lanes_attributes.duplicated(["element_id"])])
# exit()
# print(lanes_attributes)
# print(lanes_attributes[lanes_attributes["element_id"] == 2])
# print(lanes_attributes[lanes_attributes["lane_id"] == 2])
# exit()

# Merge geometries and attributes on ID
# lane_ids_before = lanes["element_id"].values
lanes = lanes.merge(lanes_attributes, on="element_id")
# lane_ids_after = lanes["element_id"].values
# difference = set(lane_ids_before).symmetric_difference(set(lane_ids_after))
# print(lanes[lanes["element_id"] == 2])
# exit()

# transform lane geometry to utm frame
lanes["geometry"] = lanes["geometry"].apply(
    lambda x: Polygon(transform_points(x.exterior.coords, transformer.t_global_utm))
)


def find_touch_edge_idx(poly, other_poly):
    coords = list(poly.exterior.coords)
    for idx, (start, end) in enumerate(zip(coords[:-1], coords[1:])):
        ls = LineString((start, end))
        if ls.covered_by(other_poly):
            return idx

    return None


def ls_to_np(ls):
    return np.array(list(ls.exterior.coords))


def split_vertex_list(vlist, idx1, idx2):
    assert idx2 > idx1

    if idx2 == len(vlist):
        split1 = vlist[idx1 + 1 :]
        split2 = vlist[: idx1 + 1]
    else:
        split1 = vlist[idx1 + 1 : idx2 + 1]
        split2 = vlist[idx2 + 1 :] + vlist[: idx1 + 1]

    return split1, split2


# Create lane boundaries based on the annotations
boundaries_left = []
boundaries_right = []
for idx, lane in lanes.iterrows():
    lane_geometry = lane["geometry"]

    from_str = lane["from_object"]
    to_str = lane["to_object"]

    assert from_str is not None and len(from_str) > 0
    assert to_str is not None and len(to_str) > 0

    from_type, from_id = parse_object_str(from_str)
    to_type, to_id = parse_object_str(to_str)

    # print(polygons["element_id"], type(from_id))
    from_object = polygons[
        (polygons["type"] == from_type) & (polygons["element_id"] == from_id)
    ]
    to_object = polygons[
        (polygons["type"] == to_type) & (polygons["element_id"] == to_id)
    ]
    # print(from_object, to_object)
    # from_object = polygons[from_str]
    # to_object = polygons[to_str]

    from_geometry = from_object["geometry"].item()
    to_geometry = to_object["geometry"].item()

    from_geometry = snap(from_geometry, lane_geometry, 1e-1)
    to_geometry = snap(to_geometry, lane_geometry, 1e-1)

    if not lane_geometry.touches(from_geometry):
        print("from geometry")
        lane_coords = list(lane_geometry.exterior.coords)
        lc = np.array(lane_coords)
        from_c = np.array(list(from_geometry.exterior.coords))
        to_c = np.array(list(to_geometry.exterior.coords))
        plt.plot(lc[:, 0], lc[:, 1], c="b")
        plt.plot(from_c[:, 0], from_c[:, 1], c="black")
        plt.plot(to_c[:, 0], to_c[:, 1], c="purple")
        plt.title(lane["element_id"])

        plt.show()
        continue

    if not lane_geometry.touches(to_geometry):
        print("to geometry")

        lane_coords = list(lane_geometry.exterior.coords)
        lc = np.array(lane_coords)
        from_c = np.array(list(from_geometry.exterior.coords))
        to_c = np.array(list(to_geometry.exterior.coords))
        plt.plot(lc[:, 0], lc[:, 1], c="b")
        plt.plot(from_c[:, 0], from_c[:, 1], c="black")
        plt.plot(to_c[:, 0], to_c[:, 1], c="purple")
        plt.title(lane["element_id"])

        plt.show()
        continue

    assert lane_geometry.touches(from_geometry)
    assert lane_geometry.touches(to_geometry)

    lane_coords = list(lane_geometry.exterior.coords)

    idx1 = find_touch_edge_idx(lane_geometry, from_geometry)
    idx2 = find_touch_edge_idx(lane_geometry, to_geometry)

    pt1 = Point(lane_coords[idx1])
    pt2 = Point(lane_coords[idx2])

    if idx2 > idx1:
        # splits = split_vertex_list(lane_coords, idx1, idx2)
        boundary_right, boundary_left = split_vertex_list(lane_coords, idx1, idx2)
    else:
        # splits = split_vertex_list(lane_coords, idx2, idx1)
        boundary_left, boundary_right = split_vertex_list(lane_coords, idx2, idx1)

    for split_ in [boundary_left, boundary_right]:
        # if end of line is closer than from_geometry, reverse the direction
        if Point(split_[0]).distance(pt1) > Point(split_[-1]).distance(pt1):
            split_ = split_[::-1]

    # boundary_left = , df: pd.DataFrameLineString(boundary_left)
    # boundary_right = LineString(boundary_right)

    boundaries_left.append(boundary_left)
    boundaries_right.append(boundary_right)

    # lane["boundary_left"] = boundary_left
    # lane["boundary_right"] = boundary_right

    # lc = np.array(lane_coords)
    # from_c = np.array(list(from_geometry.exterior.coords))
    # to_c = np.array(list(to_geometry.exterior.coords))
    # plt.plot(lc[:, 0], lc[:, 1], c="b")
    # plt.plot(from_c[:, 0], from_c[:, 1], c="black")
    # plt.plot(to_c[:, 0], to_c[:, 1], c="purple")
    # plt.plot(boundary_left[:, 0], boundary_left[:, 1], c="green")
    # plt.plot(boundary_right[:, 0], boundary_right[:, 1], c="red")

    # plt.show()

lanes["boundary_left"] = boundaries_left
lanes["boundary_right"] = boundaries_right

# Get IDs and stats
lane_ids = list(lanes["element_id"])
lane_count = len(lane_ids)
# # Assign new lane_ids, save lane_ids changelog
# lane_count = 0
# old_lane_ids = list(dict.fromkeys(lane_ids))
# new_lane_ids, lane_count = make_unique_ids(len(old_lane_ids), lane_count)
# # print(new_lane_ids)

# id_mapping = {old_id: new_id for old_id, new_id in zip(old_lane_ids, new_lane_ids)}
# new_lane_ids_ordered = [id_mapping[old_id] for old_id in list(lanes["lane_id"])]
#
# lanes.loc[:, "lane_id"] = new_lane_ids_ordered


def clean_list(value, cast_func):
    # TODO not very efficient
    if value is None:
        value = []
    else:
        value = [cast_func(id_) for id_ in value if not id_.isspace()]

        if len(value) == 0:
            value = []

    return value


def update_ids(values, mapping):
    return [mapping[value] for value in values]


lanes["successors"] = lanes["successors"].apply(
    partial(clean_list, cast_func=lambda x: x.strip())
)
lanes["predecessors"] = lanes["predecessors"].apply(
    partial(clean_list, cast_func=lambda x: x.strip())
)
lanes["connects_to"] = lanes["connects_to"].apply(
    partial(clean_list, cast_func=lambda x: x.strip())
)
# print(lanes[["predecessors", "successors"]])
# exit()

# lanes["successors"] = lanes["successors"].apply(update_ids, args=(id_mapping,))
# lanes["predecessors"] = lanes["predecessors"].apply(update_ids, args=(id_mapping,))
## Update successor and predecessor labels with new lane_ids
# for item in ["successors", "predecessors"]:
#    updated_ids = []
#    for i, lane_ids in enumerate(lanes[item]):
#        updated_ids.append([id_mapping[int(old_id)] for old_id in lane_ids])
#
#    lanes.loc[:, item] = updated_ids

# print(lanes[["predecessors", "successors"]])
# print(lanes["successors"])
# exit()

# # Assign new element_ids
# lanes = lanes.sort_values(by=["lane_id"], ignore_index=True)
# new_ids, element_count = make_unique_ids(len(lanes), element_count)
# lanes.loc[:, "element_id"] = list(new_ids)


# print(lanes[["lane_id", "element_id", "successors", "predecessors"]])
# exit()


# print(list(lanes["lane_id"]))


# def is_bound_near_ref(bound, ref_polygon, check_successor, dist_thresh=0.5):
#     if not bound["boundary_right"]:
#         return False
#
#     neighbour_start = Point(bound["geometry"].coords[0])
#     neighbour_end = Point(bound["geometry"].coords[-1])
#
#     # Check directionality, distance to intersection
#     # and distance to neighbouring lane end
#     if (
#         ref_polygon.distance(neighbour_end) < dist_thresh
#         and (ref_start.distance(neighbour_end) < ref_start.distance(neighbour_start))
#         and (neighbour_end.distance(ref_end) > neighbour_end.distance(ref_start))
#     ):
#         return True
#
#     if (
#         ref_polygon.distance(neighbour_start) < dist_thresh
#         and ref_end.distance(neighbour_start) < ref_end.distance(neighbour_end)
#         and neighbour_start.distance(ref_end) < neighbour_start.distance(ref_start)
#     ):
#         return True


# def check_neighbours(lane, lanes, ref_polygon, field_name, dist_thresh=0.5):
#    check_successor = field_name == "successor"
#
#    neighbour_list = []
#    neighbours = lane[field_name]
#    for neighbour_id in neighbours:
#        neighbour_lane = lanes[lanes["lane_id"] == neighbour_id]
#        for _, bound in neighbour_lane.iterrows():
#            if (
#                is_bound_near_ref(bound, ref_polygon, check_successor, dist_thresh)
#                and neighbour_id not in neighbour_list
#            ):
#                neighbour_list.append(neighbour_id)
#
#    return neighbour_list


# # Adjust all successor and predecessor labels based on distance and directionality
# element_ids = intersections["element_id"]
# thresh = 0.5
# lanes_predecessors = {}
# lanes_successors = {}
# for idx, lane in lanes.iterrows():
#     # the right boundary defines the direction of the lane
#     if lane["boundary_right"] is False:
#         continue
#
#     lane_id = lane["lane_id"]
#     ref_start = Point(lane["geometry"].coords[0])
#     ref_end = Point(lane["geometry"].coords[-1])
#
#     predecessor_list = []
#     successor_list = []
#
#     # Check lane_id and close intersections
#     for _, intersection in intersections.iterrows():
#         ref_polygon = Polygon(intersection["geometry"])
#         dist_succ = LineString(nearest_points(ref_end, ref_polygon)).length
#         dist_pred = LineString(nearest_points(ref_start, ref_polygon)).length
#
#         if dist_pred < thresh and lane["predecessors"]:
#             predecessor_list = check_neighbours(
#                 lane, lanes, ref_polygon, "predecessors", dist_thresh=thresh
#             )
#
#         if dist_succ < thresh and lane["successors"]:
#             successor_list = check_neighbours(
#                 lane, lanes, ref_polygon, "successors", dist_thresh=thresh
#             )
#
#     # Update the successor and predecessor labels for lane boundaries
#     lanes_predecessors[lane_id] = predecessor_list
#     lanes_successors[lane_id] = successor_list

# lanes_ids_mapping_flipped = {val: key for key, val in lanes_ids_mapping.items()}
# for old_id, new_id in lanes_ids_mapping.items():
#     print("old id:", old_id)
#     print("new id:", new_id)
#     print(
#         list(
#             lanes.loc[lanes["lane_id"] == old_id, ["predecessors", "successors"]]
#             .reset_index(drop=True)
#             .iloc[0]
#         )
#     )
#     print(lanes_predece, df: pd.DataFramessors[old_id], lanes_successors[old_id])
#     print(lanes_predecessors[new_id], lanes_successors[new_id])
#     print()

# print(lanes_ids_mapping_flipped[419])

# print(lanes[["successors", "predecessors"]])
# lanes["predecessors"] = lanes["lane_id"].apply(lambda x: lanes_predecessors[x])
# lanes["successors"] = lanes["lane_id"].apply(lambda x: lanes_successors[x])
# print(lanes[["lane_id", "successors", "predecessors"]])


# # Run label consistency checks on lane annotations
print("Running checks on labels...")
lane_ids = lanes["element_id"].values
successors_mismatch = defaultdict(list)
predecessors_mismatch = defaultdict(list)
for col, item in tqdm(lanes.iterrows(), total=len(lanes)):
    lane_id = item["lane_id"]
    predecessors = item["predecessors"]
    successors = item["successors"]

    for p_id in predecessors:
        if p_id not in lane_ids:
            successors_mismatch[lane_id].append(p_id)
            # raise ValueError(f"Predecessor {p_id} for lane {lane_id} does not exist.")

    for s_id in successors:
        if s_id not in lane_ids:
            predecessors_mismatch[lane_id].append(s_id)
            # raise ValueError(f"Successor {s_id} for lane {lane_id} does not exist.")

if len(successors_mismatch) > 0:
    print("Successor mismatches:")
    for key, values in successors_mismatch.items():
        print(key, values)

if len(predecessors_mismatch) > 0:
    print("Predecessor mismatches:")
    for key, values in predecessors_mismatch.items():
        print(key, values)

print("Passed.")

# Create new file with pre-processed annotations
lanes["predecessors"] = lanes["predecessors"].apply(lambda x: list_to_str(x))
lanes["successors"] = lanes["successors"].apply(lambda x: list_to_str(x))
lanes["connects_to"] = lanes["connects_to"].apply(lambda x: list_to_str(x))
lanes["boundary_left"] = lanes["boundary_left"].apply(lambda x: list_to_str(x))
lanes["boundary_right"] = lanes["boundary_right"].apply(lambda x: list_to_str(x))
lanes.to_file(os.path.join(output_dir, "lanes.gpkg"), driver="GPKG")

# exit()

"""-------------------------------------------------------------------"""
"""Pre-processing of lane connectors"""

# print("Loading lanes...")
# lanes = Lanes().load(os.path.join(output_dir, "lanes.gpkg"))
# print("Loading polygons...")
# polygons = Polygons().load(os.path.join(output_dir, "polygons.gpkg"))
# print("Finished loading.")
#
# DIST_THRESH = 0.5
# MAP_EXTENT = 100
#
# # for polygon_id, polygon in polygons.elements.items():
# #    poly = Polygon(polygon.bounds.nodes_utm)
# #    if not poly.is_valid:
# #        print(polygon_id)
# #        x, y = poly.exterior.xy
# #        plt.plot(x, y)
# #        plt.show()
#
#
# print("Creating lane connections...")
# lane_connections = get_lane_connections(lanes, polygons)
#
# lane_connections_df = gpd.GeoDataFrame(
#     lane_connections, geometry="geometry", crs="EPSG:28992"
# )
# # print(lane_connections_df)
# lane_connections_df.to_file(os.path.join(output_dir, "connectors.gpkg"), driver="GPKG")
#
print("Done.")
