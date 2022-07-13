
import utm
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import scipy
from scipy.interpolate import interp1d, UnivariateSpline

from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import nearest_points, unary_union

from utils import non_decreasing, non_increasing, monotonic


class Lanes:
    """
    Lanes class to handle all required lane operations
    """

    def __init__(self):
        """
        Initialize lane geometry 
        """
        self.lanes = None
        self.geod = pyproj.Geod(ellps="WGS84") # Convert coordinates to meters

    def from_df(self, df):
        #print(df)
        self.lanes = {}
        self.lane_ids = list(set(df['lane_id'].astype('int64').values))
        for lane_id in self.lane_ids:
            lane_data = df[df['lane_id'] == lane_id]
            left_bound_data = lane_data[lane_data['boundary_left']].squeeze().to_dict()
            right_bound_data = lane_data[lane_data['boundary_right']].squeeze().to_dict()

            # TODO remove third coordinate (lon-lat frame does not allow for it)
            left_boundary = RoadLine(
                    left_bound_data['lane_id'],
                    left_bound_data['road_type'],
                    np.array(left_bound_data['geometry'].coords)[:,:2])
            right_boundary = RoadLine(
                    right_bound_data['lane_id'],
                    right_bound_data['road_type'],
                    np.array(right_bound_data['geometry'].coords)[:,:2])
            
            predecessors = right_bound_data['predecessors'] 
            successors = right_bound_data['successors'] 
            allowed_agents = right_bound_data['allowed_agents'] 
            lane = Lane(lane_id, left_boundary, right_boundary, predecessors, successors, allowed_agents)
            self.lanes[lane_id] = lane
            #exit()
        return self

    def __getitem__(self, idx):
        return self.lanes[idx]

    def get_lanes_in_box(self, box, frame='lonlat'):
        if frame == 'utm':
            pass
        lanes = self.lanes
        print(self.lanes)
        exit()
        self._get_lanes_in_box(lanes, box)

    def _get_lanes_in_box(self, lanes, box):
        x_min = box[0,0]
        x_max = box[1,0]
        y_min = box[1,1]
        y_max = box[0,1]

        mask = (x_min <= lanes[:,0] < x_max) & (y_min <= lanes[:,1] < y_max)
        pass
        lanes_in_box = lanes[mask]

    def get_neighbouring_lanes(self, lane_id, d_threshold=0.3) -> list:
        """
        Get all neighbouring lane segments of a given lane segment
        :param lane_id: lane identifier for which to execute the operation
        :return left_neighbours: List of lane segment identifiers that are left neighbours
        :return right_neighbours: List of lane segment identifiers that are right neighbours
        """

        # TODO clean function!!
        left_line, right_line = self.interpolate_lane_boundaries(lane_id)
        left_line = LineString(left_line)
        right_line = LineString(right_line)

        potential_neighbours = []
        left_neighbours = []
        right_neighbours = []

        for i in range(len(self.data['geometry'][0:8])): #remove [0:8], currently there for testing purposes. 
            if (self.data['lane_id'][i] != lane_id) and (self.data['lane_id'][i] not in potential_neighbours):
                potential_neighbours.append(self.data['lane_id'][i])
            
        for neighbour in potential_neighbours:
            left_line_other, right_line_other = self.interpolate_lane_boundaries(neighbour)

            left_line_other = LineString(left_line_other)
            right_line_other = LineString(right_line_other)

            distance1 = self.geod.geometry_length(LineString(nearest_points(left_line, right_line_other)))
            if distance1 <= d_threshold:
                left_neighbours.append(neighbour)

            distance2 = self.geod.geometry_length(LineString(nearest_points(right_line, left_line_other)))
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

    @staticmethod
    def visualize_lanes(polygons):
        """
        Visualizes lane segments as polygons 
        :param polygons: list of Polygons of lane segments
        """
        p = gpd.GeoSeries(polygons)    
        p.plot(alpha=0.15, edgecolor='blue')
    
        return plt.show()

    @staticmethod
    def visualize_drivable_area(drivable_area):
        """
        Visualizes the drivable area for vehicles
        :param drivable_area: Multipolygon of fused lane segments
        """
        da = unary_union(drivable_area)
        gpd.GeoSeries(da).plot(alpha=0.15)

        return plt.show()


class RoadLine:
    def __init__(self, boundary_id, boundary_type, nodes):
        self.id = boundary_id
        self.type = boundary_type
        self.nodes = nodes

    @property
    def nodes_utm(self):
        # nodes are in (lon, lat) format, so need to be reversed
        #print(self.nodes)
        nodes_utm = utm.from_latlon(self.nodes[:,1], self.nodes[:,0])
        self.utm_zone = nodes_utm[2:]
        return np.stack(nodes_utm[:2], axis=-1)

    def _get_nodes_in_frame(self, frame):
        if frame == 'lonlat':
            return self.nodes
        elif frame == 'utm':
            return self.nodes_utm
        else:
            raise ValueError(f'Frame "{frame}" not valid.')

    def interpolate(self, frame='utm', n_points=100, kind='linear'): 
        nodes = self._get_nodes_in_frame(frame)
        #print(self.nodes)
        path_t = np.linspace(0, 1, nodes.size//2)
        
        path_x = nodes[:,0]
        path_y = nodes[:,1]
        #print(path_x.shape)
        #print(path_y.shape)
        r = nodes.T
        #print(r.shape)
        #print(path_t.shape)
        spline = interp1d(path_t, r, kind=kind)

        t = np.linspace(np.min(path_t), np.max(path_t), n_points)
        r = spline(t)

        return r.T 

class Lane:
    def __init__(self,
            lane_id,
            left_boundary,
            right_boundary,
            predecessors,
            successors,
            allowed_agents,
            lane_type=None):
        self.id = lane_id 
        self.predecessors = predecessors
        self.successors = successors
        self.allowed_agents = allowed_agents
        self.type = lane_type
        
        self._left_boundary = left_boundary
        self._right_boundary = right_boundary
        self._centerline = None 

    @property
    def left_boundary(self):
        return self._left_boundary

    @left_boundary.setter
    def left_boundary(self, data): 
        self._left_boundary = data
        self._centerline = None

    @property
    def right_boundary(self):
        return self._right_boundary

    @right_boundary.setter
    def right_boundary(self, data): 
        self._right_boundary = data
        self._centerline = None

    @property
    def centerline(self):
        if self._centerline is None:
            self._centerline = self._calculate_centerline()

        return self._centerline

    def _calculate_centerline(self):
        left_line = self.left_boundary.interpolate(frame='lonlat')
        right_line = self.right_boundary.interpolate(frame='lonlat')
        assert len(left_line) == len(right_line), 'Error! The left and right boundaries do not consist of equal points.'
        
        lr_line = np.stack([left_line, right_line], axis=-1)
        midpoints = [[(left_coord + right_coord)/2 for left_coord, right_coord in point] for point in lr_line]

        centerline = RoadLine(None, 'lane_centerline', np.array(midpoints))

        return centerline


if __name__ == '__main__':
    import os

    # plotting lanes
    df = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Lanes.gpkg')
    data = gpd.GeoDataFrame.explode(df, index_parts=False)
    #print(data)
    #exit()

    #data = data[data['lane_id'] < 5]    

    #print(data)
    lanes = Lanes().from_df(data)
    #print(lanes.lanes)
    for lane_id, lane in lanes.lanes.items():
        #print(lane_example.centerline.nodes_utm)
        try:
            lbound_nodes = lane.left_boundary.nodes_utm
            lbound = lane.left_boundary.interpolate()
        except ValueError:
            continue
        #print(lbound.shape)
        plt.title(f'{lane_id}')
        plt.plot(lbound_nodes[:,0], lbound_nodes[:,1],'or')
        plt.plot(lbound[:,0], lbound[:,1],'-k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    #exit()
    #lanes = Lanes(data)

    #box = [[],[]] 
    #lanes.get_lanes_in_box()

    lane_ids = list(set(data['lane_id'].values))
    #print(lane_id)

    #polygons = []
    #for lane_id in lane_ids:
    #    polygon = lanes.convert_boundaries_to_polygon(lane_id)
    #    polygons.append(polygon)
    #    centerline = lanes.calculate_centerline(lane_id)
    #    lanes.calculate_neighbouring_lanes(lane_id)

    #lanes.visualize_lanes(polygons)    
    #drivable_area = lanes.calculate_drivable_area(polygons)
    #lanes.visualize_drivable_area(drivable_area)
        
