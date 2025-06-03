import matplotlib.patches as mpatches
from matplotlib import pyplot as plt

from map_annotation.connectors import Connectors
from map_annotation.lanes import Lanes
from map_annotation.polygons import Polygons

LANE_COLOURS = {
    # "URBAN": (120, 120, 120),
    # "CAR": (150, 150, 150),
    # "BIKE": (100, 0, 0),
    # "BUS": (100, 100, 100),
    1: [150, 150, 150],
    2: [120, 120, 120],
    3: [100, 0, 0],
    4: [100, 0, 0],
    5: [100, 0, 0],
    6: [100, 100, 100],
}

POLYGON_COLOURS = {
    # "intersection": [0, 0, 128],
    "intersection": [0, 0, 128],
    "crosswalk": [128, 0, 0],
    "offroad": [128, 0, 128],
}


def visualize_lanes(lanes: Lanes, fig=None):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()

    for lane in lanes:
        color = LANE_COLOURS.get(lane.type, [100, 100, 100])
        color = [c / 255 for c in color]

        centerline = lane.centerline.discretize(resolution=1, s=0)

        # ax.plot(lane.centerline.nodes[:, 0], lane.centerline.nodes[:, 1], c="b")
        # ax.plot(lane.left_boundary.nodes[:, 0], lane.left_boundary.nodes[:, 1], c="g")
        # ax.plot(lane.right_boundary.nodes[:, 0], lane.right_boundary.nodes[:, 1], c="r")
        # ax.plot(lane.centerline.nodes[:, 0], lane.centerline.nodes[:, 1], c="g")
        ax.plot(centerline[:, 0], centerline[:, 1], c="g")
        ax.plot(
            lane.left_boundary.nodes[:, 0],
            lane.left_boundary.nodes[:, 1],
            # c="b",
            c=color,
        )
        ax.plot(
            lane.right_boundary.nodes[:, 0],
            lane.right_boundary.nodes[:, 1],
            # c="r",
            c=color,
        )
        # ax.scatter(
        #     lane.left_boundary.nodes[:, 0],
        #     lane.left_boundary.nodes[:, 1],
        #     c="b",
        # )
        # ax.scatter(
        #     lane.right_boundary.nodes[:, 0],
        #     lane.right_boundary.nodes[:, 1],
        #     c="r",
        # )

    return fig


def visualize_polygons(polygons: Polygons, fig=None):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()

    for polygon in polygons:
        if polygon.type == "terminal":
            continue
        # ax.plot(
        #    polygon.bounds.nodes[:, 0],
        #    polygon.bounds.nodes[:, 1],
        #    # polygon.bounds.nodes[:, 0],
        #    # polygon.bounds.nodes[:, 1],
        #    c="g",
        #    alpha=0.5,
        # )
        color = POLYGON_COLOURS.get(polygon.type, [100, 100, 100])
        color = [c / 255 for c in color]

        plt.plot(
            polygon.bounds.nodes[:, 0],
            polygon.bounds.nodes[:, 1],
            color=color,
            alpha=0.3,
        )
        patch = mpatches.Polygon(polygon.bounds.nodes, color=color, alpha=0.3)
        ax.add_patch(patch)
    return fig


def visualize_connectors(connectors: Connectors, fig=None, plot_boundaries=False):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()
    colors = {True: "purple", False: "lightgray"}
    alphas = {True: 1.0, False: 0.3}
    zorders = {True: 200, False: 5}
    for connector in connectors:
        # try:
        #    centerline = connector.centerline.discretize(resolution=2)
        # except ValueError:
        #    continue

        color = colors[connector.legal]
        alpha = alphas[connector.legal]
        zorder = zorders[connector.legal]
        ax.plot(
            connector.centerline.nodes[:, 0],
            connector.centerline.nodes[:, 1],
            # centerline[:, 0],
            # centerline[:, 1],
            c=color,
            alpha=alpha,
            zorder=zorder,
        )
        if plot_boundaries:
            ax.plot(
                connector.left_boundary.nodes[:, 0],
                connector.left_boundary.nodes[:, 1],
                c=color,
                alpha=alpha,
                zorder=zorder,
            )
            ax.plot(
                connector.right_boundary.nodes[:, 0],
                connector.right_boundary.nodes[:, 1],
                c=color,
                alpha=alpha,
                zorder=zorder,
            )
    return fig
