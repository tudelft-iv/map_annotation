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

"""Pre-processing of lanes"""
# Load lane segments
df4 = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Lanes_raw.gpkg')
lanes = gpd.GeoDataFrame.explode(df4, index_parts=False)

# Assign correct element_ids
for idx, id in enumerate(lanes['element_id']):
    lanes['element_id'][idx] = idx + 1

# Assign correct lane_ids
count = 1

for i in np.arange(0, len(lanes), 2):
    lanes['lane_id'][i] = count
    lanes['lane_id'][i+1] = count
    count += 1

# Create new file with pre-processed annotations
polygons.to_file('data/lanes.gpkg', driver='GPKG', layer='lanes')