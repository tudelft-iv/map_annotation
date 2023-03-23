import argparse
import os

import matplotlib.pyplot as plt

from map_annotation.connectors import Connectors
from map_annotation.lanes import Lanes
from map_annotation.polygons import Polygons
from map_annotation.visualization import (visualize_connectors,
                                          visualize_lanes, visualize_polygons)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", help="path to processed annotations", required=True
)
args = parser.parse_args()
input_dir = args.input

if __name__ == "__main__":
    fig = plt.figure()

    print("Visualizing lanes...")
    lanes = Lanes().load(os.path.join(input_dir, "lanes.gpkg"))
    fig = visualize_lanes(lanes, fig)

    print("Visualizing polygons...")
    polygons = Polygons().load(os.path.join(input_dir, "polygons.gpkg"))
    fig = visualize_polygons(polygons, fig)

    print("Visualizing connectors...")
    lane_connectors = Connectors().load(os.path.join(input_dir, "connectors.gpkg"))
    fig = visualize_connectors(lane_connectors, fig)

    plt.show()
