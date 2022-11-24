import pyproj
import os
import numpy as np
import geopandas as gpd
import pandas as pd
import fiona

from shapely.geometry import Polygon, LineString, Point
from shapely.ops import substring, nearest_points

import matplotlib.pyplot as plt

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
intersections['type'] = 'intersection'
crosswalks['type'] = 'crosswalk'
offroad['type'] = 'off_road'

#Create single file with all polygons
polygons = pd.concat([intersections, crosswalks, offroad], ignore_index=True)

# Assign correct element_ids
for idx, id in enumerate(polygons['element_id']):
    polygons['element_id'][idx] = idx + 1

# Create new file with pre-processed annotations
polygons.to_file('data/polygons_preprocessed.gpkg', driver='GPKG')

"""-------------------------------------------------------------------"""
"""Pre-processing of lanes"""
# Load lane segments
df7= gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Lanes_b.gpkg')
lanes_b = gpd.GeoDataFrame.explode(df7, index_parts=False)
df8 = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Lanes_h.gpkg')
lanes_h = gpd.GeoDataFrame.explode(df8, index_parts=False)
lanes = pd.concat([lanes_b, lanes_h], ignore_index=True)

# Assign correct lane_ids, save lane_ids changelog
lane_ids = list(lanes['lane_id'])
old_ids = list(dict.fromkeys(lane_ids))
new_ids = np.arange(1,len(old_ids)+1, 1)

for idx, id in enumerate(lanes['lane_id']):
    i = old_ids.index(id)
    lanes['lane_id'][idx] = new_ids[i]

# Update successor and predecessor labels with new lane_ids
for item in ['successors', 'predecessors']:
    for i, lane_ids in enumerate(lanes[item]):
        if lane_ids:
            lane_numb = lane_ids.split(',')
            updated_ids = []
            for id in lane_numb:
                id = int(id)
                idx = old_ids.index(id)
                updated_ids.append(new_ids[idx])
            lanes[item][i] = str(updated_ids)[1:-1]

# Assign correct element_ids
lanes = lanes.sort_values(by=['lane_id'], ignore_index=True)
for idx, id in enumerate(lanes['element_id']):
    lanes['element_id'][idx] = idx + 1

# Duplicate bi-directional lanes, flip the direction and split the predecessors and successors based on the lane annotations
old = []
new = []

for idx, lane in lanes.iterrows():
    if not lane['one-way']:
        # Select both right and left bound
        id = lane['lane_id']
        if id not in old:
            if lane['lane_id'] == id and lane['boundary_right']:
                right_bound_id = lane['element_id']
                options = lanes[lanes['lane_id'] == id]
                for x, lane in options.iterrows():
                    if lane['boundary_left']:
                        left_bound_id = lane['element_id']
            elif lane['lane_id'] == id and lane['boundary_left']:
                left_bound_id = lane['element_id']
                options = lanes[lanes['lane_id'] == id]
                for x, lane in options.iterrows():
                    if lane['boundary_right']:
                        right_bound_id = lane['element_id']

            # Update the element_ids, lane_ids, one_way and lane directionality of the old and new elements

            right_bound = lanes[lanes['element_id'] == right_bound_id]
            left_bound = lanes[lanes['element_id'] == left_bound_id]
            left_bound_new = right_bound
            right_bound_new = left_bound

            right_bound_new['boundary_right'] = True
            right_bound_new['boundary_left'] = False
            left_bound_new['boundary_left'] = True
            left_bound_new['boundary_right'] = False

            right_bound_new['element_id'] = int(lanes['element_id'].iat[-1]) + 1
            left_bound_new['element_id'] = int(lanes['element_id'].iat[-1]) + 2
            right_bound_new['lane_id'] = int(lanes['lane_id'].iat[-1]) + 1
            left_bound_new['lane_id'] = int(lanes['lane_id'].iat[-1]) + 1

            #right_bound_new['geometry'] = LineString(right_bound_new['geometry'].tolist()[0].coords[::-1])
            # Reverse the bounds and check directionality using the new left lane/old right lane (annotated in driving direction)
            left_bound_new['geometry'] = substring(left_bound_new['geometry'].tolist()[0], 1, 0, normalized=True) #Is called right_bound as per convention of direction in the dataset, while it actually represents the right bound of the original lane traversed in opposite direction
            right_bound_new['geometry'] = substring(right_bound_new['geometry'].tolist()[0], 1, 0, normalized=True) 

            ref_point = Point(left_bound_new['geometry'].tolist()[0].coords[0])
            if ref_point.distance(Point(right_bound_new['geometry'].tolist()[0].coords[0])) > ref_point.distance(Point(right_bound_new['geometry'].tolist()[0].coords[-1])):
                right_bound_new['geometry'] = substring(right_bound_new['geometry'].tolist()[0], 1, 0, normalized=True)

            # Adjust one-way label
            lanes['one-way'][right_bound_id - 1] = True
            lanes['one-way'][left_bound_id - 1] = True
            right_bound_new['one-way'] = True
            left_bound_new['one-way'] = True

            # Changelog of successors and predecessors
            old.append(id)
            new.append(right_bound_new['lane_id'].tolist()[0])

            # Append new lanes to the lanes dataframe 
            lanes = pd.concat([lanes, right_bound_new, left_bound_new], ignore_index=True)

# Add all new successor and predecessor candidate labels based on the new lane copies
for item in ['successors', 'predecessors']:
    for i, lane_element in enumerate(lanes[item]):
        if lane_element:
            lane_numb = lane_element.split(',')
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
element_ids = intersections['element_id']

for idx, lane in lanes.iterrows():
    if lane['boundary_right']:
        lane_id = lane['lane_id']
        ref_start = Point(lane['geometry'].coords[0])
        ref_end = Point(lane['geometry'].coords[-1])

        predecessor_list = []
        successor_list = []

        # Check lane_id and close intersections
        for x, intersection in intersections.iterrows():
            ref_polygon = Polygon(intersection['geometry'])
            dist_succ = LineString(nearest_points(ref_end, ref_polygon)).length
            dist_pred = LineString(nearest_points(ref_start, ref_polygon)).length
        
            if dist_pred < 0.5:
                if lane['predecessors']:
                    predecessors = lane['predecessors'].split(',')
                    for predecessor in predecessors:
                        predecessor_id = int(predecessor)
                        predecessor_lane = lanes[lanes['lane_id'] == predecessor_id]
                        for i, bound in predecessor_lane.iterrows():
                            if bound['boundary_right']:
                                predecessor_start = Point(bound['geometry'].coords[0])
                                predecessor_end = Point(bound['geometry'].coords[-1])
                                # Check directionality, distance to intersection and distance to other lane end
                                if ref_polygon.distance(predecessor_end) < 0.5:
                                    if ref_start.distance(predecessor_end) < ref_start.distance(predecessor_start):
                                        if predecessor_end.distance(ref_end) > predecessor_end.distance(ref_start):
                                            predecessor_list.append(predecessor_id)

            if dist_succ < 0.5:
                if lane['successors']:
                    successors = lane['successors'].split(',')
                    for successor in successors:
                        successor_id = int(successor)
                        successor_lane = lanes[lanes['lane_id'] == successor_id]
                        for i, bound in successor_lane.iterrows():
                            if bound['boundary_right']:
                                successor_start = Point(bound['geometry'].coords[0])
                                successor_end = Point(bound['geometry'].coords[-1])
                                # Check directionality, distance to intersection and distance to other lane end
                                if ref_polygon.distance(successor_start) < 0.5:
                                    if ref_end.distance(successor_start) < ref_end.distance(successor_end):
                                        if successor_start.distance(ref_end) < successor_start.distance(ref_start):
                                            successor_list.append(successor_id)

    # Update the successor and predecessor labels
    for i, lane in lanes[lanes['lane_id'] == lane_id].iterrows():
        element_id = lane['element_id']
        lanes['predecessors'][element_id - 1] = str(predecessor_list)[1:-1]
        lanes['successors'][element_id - 1] = str(successor_list)[1:-1]

# Run label consistency checks on lane annotations
for col, item in lanes.iterrows():
    if (item['boundary_right'] == True and item['boundary_left'] == True) or (item['boundary_right'] == False and item['boundary_left'] == False):
        assert(f'Error! Lane has incorrect boundary booleans.')
        print(item['element_id'])

    lane_id = item['lane_id']

    for col, item2 in lanes.iterrows():
        if item['lane_id'] == item2['lane_id']:
            if item['element_id'] != item2['element_id']:
                if item['boundary_left'] == item2['boundary_left'] or item['boundary_right'] == item2['boundary_right']:
                    print(item['lane_id'], 'has inconsistent boundaries')
                if item['one-way'] != item2['one-way']:
                    print(item['lane_id'], 'has inconsistent direction')
                if item['predecessors'] != item2['predecessors']:
                    print(item['lane_id'], 'has inconsistent predecessors')
                if item['successors'] != item2['successors']:
                    print(item['lane_id'], 'has inconsistent successors')
                if not item['geometry']:
                    print(item['lane_id'], 'has no geometry')

print('Done, all good to go!')

# for geom in lanes['geometry']:
#     plt.plot(*geom.xy)

# plt.show()

# Create new file with pre-processed annotations
lanes.to_file('data/lanes_preprocessed.gpkg', driver='GPKG')


