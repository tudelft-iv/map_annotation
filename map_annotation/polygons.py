import numpy as np

from map_annotation.road_elements import (RoadElement, RoadElementCollection,
                                          RoadLine)
from map_annotation.utils import polygon_intersection


class Polygons(RoadElementCollection):
    """
    Polygon class to handle all required polygon map operations
    """

    POLYGON_TYPE_MAP = {
        "intersection": "i",
        "offroad": "o",
        "terminal": "t",
        "crosswalk": "c",
    }

    def __init__(self):
        super().__init__()

    def from_df(self, df):
        self.elements = {}

        self.element_ids = []
        for idx, polygon_data in df.iterrows():
            polygon_type = polygon_data["type"]
            element_id = str(polygon_data["element_id"])
            element_id = f"{Polygons.POLYGON_TYPE_MAP[polygon_type]}{element_id}"

            geometry = polygon_data["geometry"]

            # TODO process road_type and allowed_agents

            polygon = Polygon(
                element_id,
                polygon_type,
                geometry,
                road_type=None,
                allowed_agents=None,
            )
            self.elements[element_id] = polygon
            self.element_ids.append(element_id)

        return self

    def __getitem__(self, key):
        return self.elements[key]

    def get_polygons_in_box(self, box):
        return self._get_polygons_in_box(box)

    def _get_polygons_in_box(self, box):
        polygons_in_box = []
        for element_id, polygon in self.elements.items():
            if len(polygon_intersection(polygon.bounds.nodes_utm, box)) > 0:
                polygons_in_box.append(element_id)

        return polygons_in_box


class Polygon(RoadElement):
    def __init__(
        self, element_id, polygon_type, geometry, road_type=None, allowed_agents=None
    ):
        super().__init__(element_id, road_type, allowed_agents)
        self.geometry = geometry
        self.type = polygon_type
        self._bounds = None

    @property
    def bounds(self):
        if self._bounds is None:
            self._bounds = self._calculate_bounds()

        return self._bounds

    def _calculate_bounds(self):
        nodes = np.array(self.geometry.exterior.coords)
        bounds = RoadLine(self.id, nodes, road_type=self.road_type)

        return bounds
