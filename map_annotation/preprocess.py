import pyproj
import os

import numpy as np
import geopandas as gpd


"""Pre-processing of polygons"""
# Load different polygon types
df = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Intersections.gpkg')
intersections = gpd.GeoDataFrame.explode(df, index_parts=False)

df2 = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Crosswalks.gpkg')
crosswalks = gpd.GeoDataFrame.explode(df2, index_parts=False)

df3 = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Offroad.gpkg')
offroad = gpd.GeoDataFrame.explode(df3, index_parts=False)

# Assign type to polygon type
intersections['type'] = 'intersection'
crosswalks['type'] = 'crosswalk'
offroad['type'] = 'off-road'

#Create single file with all polygons
polygons = intersections.append(crosswalks)
polygons = polygons.append(offroad)

# Assign correct element_ids
for idx, id in enumerate(polygons['element_id']):
    polygons['element_id'][idx] = idx + 1

# Create new file with pre-processed annotations
polygons.to_file('data/polygons.gpkg', driver='GPKG', layer='polygons')

"""-------------------------------------------------------------------"""
"""Pre-processing of lanes"""
# Load lane segments
df4 = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/lanes_raw_2.gpkg')
lanes = gpd.GeoDataFrame.explode(df4, index_parts=False)

# Assign correct element_ids
for idx, id in enumerate(lanes['element_id']):
    lanes['element_id'][idx] = idx + 1

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
            lanes[item][i] = updated_ids

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

print('Done, all good to go!')

# Create new file with pre-processed annotations
polygons.to_file('data/lanes.gpkg', driver='GPKG', layer='lanes')