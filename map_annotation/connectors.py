import ast

import numpy as np

from map_annotation.lanes import Lane
from map_annotation.road_elements import (RoadElement, RoadElementCollection,
                                          RoadLine)
from map_annotation.utils import parse_str_to_list


class Connectors(RoadElementCollection):
    """
    Class to handle lane connector functionality
    """

    def __init__(self):
        super().__init__()

    def from_df(self, df):
        """
        Retrieve lane information from labelled data.
        """
        self.elements = {}
        self.element_ids = list(set(df["connector_id"].astype(str).values))

        # Retrieve connector data
        for idx, connector in df.iterrows():
            element_id = str(connector["connector_id"])
            intersection_id = str(connector["intersection_id"])
            legal = connector["legal"]
            lane_type = connector["lane_type"]

            source = str(connector["source"])
            dest = str(connector["dest"])

            element_nodes = np.array(connector["geometry"].coords)[:, :2]
            centerline = RoadLine(
                element_id,
                element_nodes,
                line_type=RoadLine.LineType.CONNECTOR,
                road_type=lane_type,
            )

            lbound = np.array(list(ast.literal_eval(connector["left_boundary"])))
            rbound = np.array(list(ast.literal_eval(connector["right_boundary"])))
            left_boundary = RoadLine(
                element_id,
                np.array(lbound),
                line_type=RoadLine.LineType.BOUNDARY,
                road_type=lane_type,
            )
            right_boundary = RoadLine(
                element_id,
                np.array(rbound),
                line_type=RoadLine.LineType.BOUNDARY,
                road_type=lane_type,
            )

            polygon = np.array(list(ast.literal_eval(connector["polygon"])))

            element = Connector(
                element_id,
                left_boundary,
                right_boundary,
                source,
                dest,
                legal,
                intersection_id,
                lane_type=lane_type,
                polygon=polygon,
                centerline=centerline,
            )
            self.elements[element_id] = element

        return self

    def get_connectors_in_box(
        self,
        box,
    ):
        """
        Select the lanes that are within the specified bounding box.
        """
        connectors = self.elements
        return self._get_connectors_in_box(connectors, box)

    def _get_connectors_in_box(self, connectors, box):
        """
        Select the connectors that are within the specified bounding box.
        """

        # Checks for lanes that geometrically match the region of the frame of interest
        connectors_in_box = []

        for connector_id in connectors:
            connector = self.elements[connector_id]
            line = connector.centerline

            if line.intersects(box):
                connectors_in_box.append(connector_id)

        return connectors_in_box

    def discretize_by_ids(self, connector_ids, resolution):
        """Discretize connectors with given ids"""
        return [
            self.elements[connector_id].discretize(resolution)
            for connector_id in connector_ids
        ]

    def discretize(self, resolution):
        """Discretize all connectors"""
        return [connector.discretize(resolution) for connector in self.elements]


class Connector(Lane):
    """
    Class that implements lane connector functionality
    """

    def __init__(
        self,
        connector_id,
        left_boundary,
        right_boundary,
        source,
        dest,
        legal,
        intersection_id=None,
        lane_type=None,
        polygon=None,
        centerline=None,
    ):
        self.legal = legal

        if lane_type is not None:
            lane_type = RoadElement.RoadType(lane_type)
        allowed_agents = self._get_allowed_agents(lane_type)

        super().__init__(
            connector_id,
            left_boundary,
            right_boundary,
            [source],
            [dest],
            allowed_agents,
            lane_type,
            polygon,
            centerline,
        )

        self.intersection_id = intersection_id
        self.source = source
        self.dest = dest

    def _get_allowed_agents(self, lane_type):
        if not self.legal or lane_type is None:
            return []

        allowed_agents = RoadElement.RoadType_AllowedAgents_map[lane_type]
        return allowed_agents
