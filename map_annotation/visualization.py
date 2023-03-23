from matplotlib import pyplot as plt

from map_annotation.lanes import Lanes
from map_annotation.polygons import Polygons
from map_annotation.connectors import Connectors


def visualize_lanes(lanes: Lanes, fig=None):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()

    for lane in lanes:
        # ax.plot(lane.centerline.nodes[:, 0], lane.centerline.nodes[:, 1], c="b")
        # ax.plot(lane.left_boundary.nodes[:, 0], lane.left_boundary.nodes[:, 1], c="g")
        # ax.plot(lane.right_boundary.nodes[:, 0], lane.right_boundary.nodes[:, 1], c="r")
        ax.plot(lane.centerline.nodes_utm[:, 0], lane.centerline.nodes_utm[:, 1], c="b")
        ax.plot(
            lane.left_boundary.nodes_utm[:, 0],
            lane.left_boundary.nodes_utm[:, 1],
            c="g",
        )
        ax.plot(
            lane.right_boundary.nodes_utm[:, 0],
            lane.right_boundary.nodes_utm[:, 1],
            c="r",
        )

    return fig


def visualize_polygons(polygons: Polygons, fig=None):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()

    for polygon in polygons:
        ax.plot(
            polygon.bounds.nodes_utm[:, 0],
            polygon.bounds.nodes_utm[:, 1],
            # polygon.bounds.nodes[:, 0],
            # polygon.bounds.nodes[:, 1],
            c="g",
            alpha=0.5,
        )

    return fig


def visualize_connectors(connectors: Connectors, fig=None):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()

    for connector in connectors:
        # print(connector.nodes)
        ax.plot(connector.nodes[:, 0], connector.nodes[:, 1], c="b")
    return fig
