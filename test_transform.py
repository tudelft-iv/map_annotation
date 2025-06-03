
import argparse
import numpy as np
import os

import matplotlib.pyplot as plt

from map_annotation.lanes import Lanes
from map_annotation.transforms import CoordTransformer

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", help="path to processed annotations", required=True
)
args = parser.parse_args()
input_dir = args.input

if __name__ == "__main__":
    transformer = CoordTransformer()

    lanes = Lanes().load(os.path.join(input_dir, "lanes.gpkg"))

    for lane in lanes:
        nodes_utm = lane.centerline.nodes

        print("Nodes UTM 31U: \n", nodes_utm)
        nodes_transformed = np.array([transformer.t_utm_global(*node) for node in nodes_utm])
        print("Nodes EPSG 28992: \n", nodes_transformed)
        exit()

