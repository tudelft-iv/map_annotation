import geopandas as gpd

filepath = "data/processed_new/connectors.gpkg"

file_ = gpd.read_file(filepath)
df = gpd.GeoDataFrame.explode(file_, index_parts=False)

print(df)
