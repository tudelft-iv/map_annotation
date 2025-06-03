import argparse
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

from map_annotation.connectors import Connectors
from map_annotation.lanes import Lanes
from map_annotation.polygons import Polygons
from map_annotation.utils import get_lane_connections, list_to_str

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", help="path to processed annotations", required=True
)
parser.add_argument(
    "-o", "--output", help="path to processed annotations", required=True
)
parser.add_argument("-m", "--merge", help="merge raw annotations", action="store_true")

args = parser.parse_args()
input_dir = args.input
output_dir = args.output
merge = args.merge

Path(output_dir).mkdir(parents=True, exist_ok=True)

"""-------------------------------------------------------------------"""
"""Pre-processing of lane connectors"""

lanes_input_file = os.path.join(input_dir, "lanes.gpkg")
polygons_input_file = os.path.join(input_dir, "polygons.gpkg")

print("Loading lanes...")
lanes = Lanes().load(lanes_input_file)
print("Loading polygons...")
polygons = Polygons().load(polygons_input_file)
print("Finished loading.")


print("Creating lane connections...")
lane_connections = get_lane_connections(lanes, polygons)

lane_connections_df = gpd.GeoDataFrame(
    lane_connections, geometry="geometry", crs="EPSG:28992"
)
# print(lane_connections_df)

connectors = Connectors().from_df(lane_connections_df)

# update successors and predecessors in lanes df
lanes_df = gpd.read_file(lanes_input_file)
polygons_df = gpd.read_file(polygons_input_file)


updates = []
for lane_id in lanes_df["lane_id"].values:
    successors = []
    predecessors = []
    for connector in connectors:
        if not connector.legal:
            continue

        if lane_id in connector.successors:
            predecessors.append(connector.id)
        if lane_id in connector.predecessors:
            successors.append(connector.id)

    # print(predecessors, successors)
    update = [list_to_str(predecessors), list_to_str(successors)]
    # print(update)
    updates.append(update)

# print(updates)
lanes_df[["predecessors", "successors"]] = updates

# output to files
lane_connections_df.to_file(os.path.join(output_dir, "connectors.gpkg"), driver="GPKG")
lanes_df.to_file(os.path.join(output_dir, "lanes.gpkg"), driver="GPKG")
polygons_df.to_file(os.path.join(output_dir, "polygons.gpkg"), driver="GPKG")

print("Done.")
