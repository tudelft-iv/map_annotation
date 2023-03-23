import matplotlib.pyplot as plt

from map_annotation.connectors import Connectors
from map_annotation.lanes import Lanes
from map_annotation.polygons import Polygons
from map_annotation.visualization import (visualize_connectors,
                                          visualize_lanes, visualize_polygons)

if __name__ == "__main__":
    fig = plt.figure()

    print("Visualizing lanes...")
    lanes = Lanes().load("data/preprocessed/lanes.gpkg")
    fig = visualize_lanes(lanes, fig)

    print("Visualizing polygons...")
    polygons = Polygons().load("data/preprocessed/polygons.gpkg")
    fig = visualize_polygons(polygons, fig)

    print("Visualizing connectors...")
    lane_connectors = Connectors().load("data/preprocessed/connectors.gpkg")
    fig = visualize_connectors(lane_connectors, fig)

    plt.show()
