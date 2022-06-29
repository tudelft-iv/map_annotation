import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj

from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import nearest_points, unary_union

class Lanes:
    """
    Lanes class to handle all required lane operations
    """

    def __init__(self, data):
        """
        Initialize lane geometry 
        :param data: lane geometry with corresponding labels
        """

        self.data = data

    def get_coordinates(self) -> np.ndarray:
        """
        Gets coordinates points of the lane segment polylines
        :return coords: np.ndarray with the coordinates of lane segments
        """

        coords = []
        line_segments = self.data['geometry']

        for line in line_segments:
            coords_point = list(line.coords)
            coords.append(coords_point)

        coords = np.asarray(coords, dtype="object")
        
        return coords

    def get_lane_boundaries(self, lane_id) -> list:
        """
        Retrieves the coordinates of the left and right boundaries of a lane segment 
        :param lane_id: lane identifier for which to execute the operation
        :return boundary_left: list with boundary points of the left side of the lane segment 
        :return boundary_right: list with boundary points of the right side of the lane segment 
        """

        coords = self.get_coordinates()

        for idx in range(len(self.data)):
            if self.data['lane_id'][idx] == lane_id:
                if self.data['boundary_right'][idx]:
                    boundary_right = coords[idx]
                if self.data['boundary_left'][idx]:
                    boundary_left = coords[idx]
        
        return boundary_left, boundary_right

    def interpolate_lane_boundaries(self, lane_id) -> np.ndarray:
        """
        Interpolates the left and right boundary of a lane segment
        :param lane_id: lane identifier for which to execute the operation
        :return left_line: np.ndarray of interpolated points along the left lane boundary
        :return right_line: np.ndarray of points along the right lane boundary
        """
        boundary_left, boundary_right = self.get_lane_boundaries(lane_id)

        left_string = LineString(boundary_left)
        right_string = LineString(boundary_right) 

        n_points = 100
        distances_left = np.linspace(0, left_string.length, n_points)
        distances_right = np.linspace(0, right_string.length, n_points)

        points_left = [left_string.interpolate(distance) for distance in distances_left]
        points_right = [right_string.interpolate(distance) for distance in distances_right] 

        left_line = np.array(LineString(points_left).coords)
        right_line = np.array(LineString(points_right).coords)

        return left_line, right_line

    def calculate_centerline(self, lane_id) -> np.ndarray:
        """
        Determines the lane centerline
        :param lane_id: lane identifier for which to execute the operation
        :return centerline: np.ndarray of the centerline coordinates of a lane segment
        """

        left_line, right_line = self.interpolate_lane_boundaries(lane_id)
        assert len(left_line) == len(right_line), 'Error! The left and right boundaries do not consists of equal points.'

        midpoints = []

        for i in range(len(left_line)):
            point = [((left_line[i][0] + right_line[i][0]) / 2), ((left_line[i][1] + right_line[i][1]) / 2) , ((left_line[i][0] + right_line[i][0]) / 2)]
            midpoints.append(point)

        centerline_l = LineString(midpoints) #Preferable to return a LineString element?
        centerline = np.array(midpoints)

        plt.scatter(left_line[:,0],left_line[:,1], label='left')
        plt.scatter(right_line[:,0],right_line[:,1], label='right')
        plt.scatter(centerline[:,0],centerline[:,1], label='center')
        plt.legend()
        plt.show()

        return centerline

    def convert_boundaries_to_polygon(self, lane_id) -> Polygon:
        """
        Gets road boundaries and generates a polygon of the lange segment
        :param lane_id: lane identifier for which to execute the operation
        :return polygon: Polygon of a lane segment
        """

        boundary_left, boundary_right = self.get_lane_boundaries(lane_id)
        polygon = Polygon(np.vstack([boundary_left, boundary_right[-1], boundary_right[::-1], boundary_left[0]]))
        
        return polygon

    def calculate_neighbouring_lanes(self, lane_id) -> list:
        """
        Get all neighbouring lane segments of a given lane segment
        :param lane_id: lane identifier for which to execute the operation
        :return left_neighbours: List of lane segment identifiers that are left neighbours
        :return right_neighbours: List of lane segment identifiers that are right neighbours
        """

        left_line, right_line = self.interpolate_lane_boundaries(lane_id)
        left_line = LineString(left_line)
        right_line = LineString(right_line)

        potential_neighbours = []
        left_neighbours = []
        right_neighbours = []
        d_threshold = 0.3 # [m]

        geod = pyproj.Geod(ellps="WGS84") # Convert coordinates to meters

        for i in range(len(self.data['geometry'][0:8])): #remove [0:8], currently there for testing purposes. 
            if (self.data['lane_id'][i] != lane_id) and (self.data['lane_id'][i] not in potential_neighbours):
                potential_neighbours.append(self.data['lane_id'][i])
            
        for neighbour in potential_neighbours:
            left_line_other, right_line_other = self.interpolate_lane_boundaries(neighbour)

            left_line_other = LineString(left_line_other)
            right_line_other = LineString(right_line_other)

            distance1 = geod.geometry_length(LineString(nearest_points(left_line, right_line_other)))
            if distance1 <= d_threshold:
                left_neighbours.append(neighbour)

            distance2 = geod.geometry_length(LineString(nearest_points(right_line, left_line_other)))
            if distance2 <= d_threshold:
                right_neighbours.append(neighbour)

        return left_neighbours, right_neighbours

    def calculate_drivable_area(self, polygons):
        """
        Gets polygons of lane segments and calculates the drivable area for vehicles.py
        :input polygons: Polygons of individual lane segments
        :return: drivable_area: Multipolygon of fused lane segments
        """
        drivable_area = MultiPolygon(polygons)

        return drivable_area


    def visualize_lanes(self, polygons):
        """
        Visualizes lane segments as polygons 
        :param polygons: list of Polygons of lane segments
        """

        p = gpd.GeoSeries(polygons)    
        p.plot(alpha=0.15, edgecolor='blue')
    
        return plt.show()

    def visualize_drivable_area(self, drivable_area):
        """
        Visualizes the drivable area for vehicles
        :param drivable_area: Multipolygon of fused lane segments
        """

        da = unary_union(drivable_area)
        gpd.GeoSeries(da).plot(alpha=0.15)

        return plt.show()

'Plotting Lanes'
df = gpd.read_file('~/Desktop/QGIS_files/annotation_layers_copy/Lanes.gpkg')
data = gpd.GeoDataFrame.explode(df, index_parts=False)

Lane = Lanes(data)

lane_id = []

for idx in range(len(data)):
    if data['lane_id'][idx] not in lane_id:
        lane_id.append(data['lane_id'][idx])

lane_id = lane_id[0:5] #for testing purposes, needs to be removed eventually.

polygons = []

for lane in lane_id:
    lane_id = lane
    polygon = Polygon(Lane.convert_boundaries_to_polygon(lane_id))
    polygons.append(polygon)
    centerline = Lane.calculate_centerline(lane_id)
    Lane.calculate_neighbouring_lanes(lane_id)

Lane.visualize_lanes(polygons)    
drivable_area = Lane.calculate_drivable_area(polygons)
Lane.visualize_drivable_area(drivable_area)
    