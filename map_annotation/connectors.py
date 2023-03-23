import numpy as np

from map_annotation.road_elements import RoadElementCollection, RoadLine


class Connectors(RoadElementCollection):
    """
    Lanes class to handle all required lane operations
    """

    def __init__(self):
        super().__init__()

    def from_df(self, df):
        """
        Retrieve lane information from labelled data.
        """
        self.elements = {}
        self.element_ids = list(set(df["connector_id"]))

        # Retrieve lane data
        for element_id in self.element_ids:
            element_data = df[df["connector_id"] == element_id]
            intersection_id = element_data["intersection_id"]

            element_nodes = np.array(element_data["geometry"].tolist()[0].coords)[:, :2]
            element = Connector(element_id, element_nodes, intersection_id)
            self.elements[element_id] = element

        return self

    def get_connectors_in_box(
        self,
        box,
        frame="utm",
    ):
        """
        Select the lanes that are within the specified bounding box.
        """
        if frame == "lon-lat":
            pass
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


class Connector(RoadLine):
    def __init__(self, connector_id, nodes, intersection_id=None):
        super().__init__(connector_id, nodes, line_type=RoadLine.LineType.CONNECTOR)
        self.intersection_id = intersection_id

    @property
    def centerline(self):
        # Alias for the nodes
        return self.nodes
