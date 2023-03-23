import numpy as np

from map_annotation.road_elements import (RoadElement, RoadElementCollection,
                                          RoadLine)


class Polygons(RoadElementCollection):
    """
    Polygon class to handle all required polygon map operations
    """

    def __init__(self):
        super().__init__()

    def from_df(self, df):
        self.elements = {}
        self.element_ids = list(set(df["element_id"].astype("int64").values))

        for element_id in self.element_ids:
            geometry_data = df[df["element_id"] == element_id]

            # nodes = np.array((geometry_data['geometry'][0]).exterior.coords)
            road_type = geometry_data["road_type"]
            allowed_agents = geometry_data["allowed_agents"]
            geometry = geometry_data["geometry"]

            polygon = Polygon(
                element_id,
                road_type,
                allowed_agents,
                geometry,
            )
            self.elements[element_id] = polygon

        return self

    def __getitem__(self, idx):
        return self.elements[idx]

    def get_polygons_in_box(self, pose, map_extent, frame="utm"):
        if frame == "lon-lat":
            pass
        self._get_polygons_in_box(self.elements, pose, map_extent)

    def _get_polygons_in_box(self, polygons, pose, map_extent):
        box = np.array(
            [
                [pose[0] - map_extent, pose[0] + map_extent],
                [pose[1] - map_extent, pose[1] + map_extent],
            ]
        )

        x_min = box[0, 0]
        x_max = box[0, 1]
        y_min = box[1, 0]
        y_max = box[1, 1]

        polygons_in_box = []
        for element_id in polygons:
            polygon = polygons[element_id]
            for node in polygon.nodes:
                if element_id in polygons_in_box:
                    continue
                if (x_min <= node[0] < x_max) & (y_min <= node[1] < y_max):
                    polygons_in_box.append(element_id)

        polygons = [polygons[polygon] for polygon in polygons_in_box]

        return polygons


class Polygon(RoadElement):
    def __init__(self, element_id, road_type, allowed_agents, geometry):
        super().__init__(element_id, road_type, allowed_agents)
        self.geometry = geometry
        self._bounds = None

    @property
    def bounds(self):
        if self._bounds is None:
            self._bounds = self._calculate_bounds()

        return self._bounds

    def _calculate_bounds(self):
        poly = self.geometry.tolist()
        bounds_x, bounds_y = np.array(poly[0].exterior.coords.xy)
        nodes = np.stack((bounds_x, bounds_y), axis=-1)

        bounds = RoadLine(self.id, nodes, road_type=self.road_type)

        return bounds
