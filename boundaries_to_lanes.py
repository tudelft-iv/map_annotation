import argparse
import glob
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import MultiLineString, Polygon

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", help="path to directory with raw annotations", required=True
)
parser.add_argument(
    "-o", "--output", help="path to store processed annotations", required=True
)
parser.add_argument("-m", "--merge", help="merge raw annotations", action="store_true")

args = parser.parse_args()
input_dir = args.input
output_dir = args.output

# Load boundarie segments
boundaries_files = glob.glob(os.path.join(input_dir, "Boundaries*.gpkg"))
boundaries_dfs = [
    gpd.GeoDataFrame.explode(gpd.read_file(filepath), index_parts=False)
    for filepath in boundaries_files
]
boundaries = pd.concat(boundaries_dfs, ignore_index=True)

# print(lanes)
lane_ids = boundaries["lane_id"].unique()
# print(lane_ids)

lanes_mls_list = []
lanes_polygon_list = []
for lane_id in lane_ids:
    lane_bounds = boundaries[boundaries["lane_id"] == lane_id]

    lbound = lane_bounds[lane_bounds["boundary_left"]]
    rbound = lane_bounds[lane_bounds["boundary_right"]]
    # print(lbound)
    # print(rbound)
    # exit()

    lbound_geom = np.array(lbound["geometry"].item().coords)
    rbound_geom = np.array(rbound["geometry"].item().coords)

    # check directions of bounds
    # they need to be in oppposite directions for this to work
    # use right boundary as guide
    ref = rbound_geom[-1]

    start_dist = np.sum((lbound_geom[0] - ref) ** 2)
    end_dist = np.sum((lbound_geom[-1] - ref) ** 2)

    if end_dist < start_dist:
        # the left bound needs to be reversed to make a polygon
        lbound_geom = lbound_geom[::-1]

    # combine lane bound nodes to make polygon (final elem for closure)
    polygon = np.vstack([rbound_geom, lbound_geom, rbound_geom[0]])
    # print(polygon)

    mls = [[start, end] for start, end in zip(polygon[:-1], polygon[1:])]
    mls = MultiLineString(mls)

    # construct lane, using right boundary as reference
    lane_mls = {
        "lane_id": lane_id,
        "element_id": rbound["element_id"].item(),
        "boundary_type": rbound["boundary_type"].item(),
        "road_type": rbound["road_type"].item(),
        "allowed_agents": rbound["allowed_agents"].item(),
        "successors": rbound["successors"].item(),
        "predecessors": rbound["predecessors"].item(),
        "geometry": mls,
    }
    # print(lane_mls)

    lanes_mls_list.append(lane_mls)
    # plt.plot(polygon[:, 0], polygon[:, 1], c="r")
    ## plt.plot(lbound_geom[:, 0], lbound_geom[:, 1], c="b")
    ## plt.plot(rbound_geom[:, 0], rbound_geom[:, 1], c="b")
    # plt.show()

    polygon_object = Polygon(polygon)
    lane_polygon = {
        "lane_id": lane_id,
        "element_id": rbound["element_id"].item(),
        "boundary_type": rbound["boundary_type"].item(),
        "road_type": rbound["road_type"].item(),
        "allowed_agents": rbound["allowed_agents"].item(),
        "successors": rbound["successors"].item(),
        "predecessors": rbound["predecessors"].item(),
        "geometry": polygon_object,
    }
    lanes_polygon_list.append(lane_polygon)


lanes_mls_df = gpd.GeoDataFrame(lanes_mls_list).set_crs("EPSG:28992")
lanes_mls_df.to_file(os.path.join(output_dir, "LanesMLS.gpkg"))
lanes_polygon_df = gpd.GeoDataFrame(lanes_polygon_list).set_crs("EPSG:28992")
lanes_polygon_df.to_file(os.path.join(output_dir, "Lanes.gpkg"))
exit()
