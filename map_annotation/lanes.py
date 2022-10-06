
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import utm

from scipy.interpolate import interp1d
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import nearest_points, unary_union

from polygons import Polygons
from utils import non_decreasing, non_increasing, monotonic
from transforms import CoordTransformer

class Lanes:
    """
    Lanes class to handle all required lane operations
    """

    def __init__(self):
        """
        Initialize lane geometry 
        """
        self.lanes = None

        # Convert coordinates to meters
        self.geod = pyproj.Geod(ellps="WGS84")

    def from_df(self, df):
        """
        Retrieve lane information from labelled data.
        """
        self.lanes = {}
        self.lane_ids = list(set(df['lane_id'].astype('int64').values))

        # Retrieve lane data
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
            
            self.lane = Lane(lane_id, left_boundary, right_boundary, predecessors, successors, allowed_agents)
            self.lanes[lane_id] = self.lane
            
        return self

    def __getitem__(self, idx):
        return self.lanes[idx]

    def get_frame_location(self, global_pose, map_extent):

        # Determine location of interest
        bounding_box = np.array([[global_pose[0] - map_extent, global_pose[0] + map_extent],
                        [global_pose[1] - map_extent, global_pose[1] + map_extent]])

        return bounding_box

    def get_lanes_in_box(self, global_pose, map_extent, frame='utm'):
        if frame == 'lon-lat':
            pass
        lanes = self.lanes
        self._get_lanes_in_box(lanes, global_pose, map_extent)

    def _get_lanes_in_box(self, lanes, global_pose, map_extent):

        box = self.get_frame_location(global_pose, map_extent)

        x_min = box[0,0]
        x_max = box[0,1]
        y_min = box[1,0]
        y_max = box[1,1]

        # Checks for lanes that geometrically match the frame of interest
        lanes_in_box = []
        for lane_id in lanes:
            lane = lanes[lane_id]
            for node in lane.centerline.nodes_utm:
                if lane_id in lanes_in_box:
                    continue
                if (x_min <= node[0] < x_max) & (y_min <= node[1] < y_max):
                    lanes_in_box.append(lane_id)
        return lanes_in_box

    def get_lane_connections(self, lane_id, element_ids):
        successors = self.lanes[lane_id].successors
        successors = successors.split(',')
        
        lane_connectors = []

        connection_points1 = self.lanes[lane_id].centerline.nodes_utm[-3:]
        reference_point = self.lanes[lane_id].centerline.nodes_utm[-1]
        dist_threshold = 1 # in [m]

        for element_id in range(1,element_ids):
            # Determines whether intersection geomatches lane end point
            ref_point = Point(reference_point)
            ref_polygon =  Polygon(polygons[element_id].bounds.nodes_utm)
            dist = LineString(nearest_points(ref_point, ref_polygon)).length

            if dist < dist_threshold: 
                for successor in successors:

                    successor = int(successor)
                    connection_points2 = self.lanes[successor].centerline.nodes_utm[:3]

                    connection_points = [connection_points1, connection_points2]
                    connection_line = []

                    for line in connection_points:
                        for guiding_point in line:
                            connection_line.append(guiding_point)

                    x = np.asarray([i[0] for i in connection_line])
                    y = np.asarray([i[1] for i in connection_line])

                    xt, yt = self.interpolate_lane_connector(x, y)

                    # Remove points that do not lie within the intersection
                    points = list(zip(xt, yt))
                    pop = []

                    for idx, point in enumerate(points):
                        point = Point(point)
                        # Use distance as contain/within methods have rounding errors
                        if point.distance(ref_polygon) > 1e-3:
                            pop.append(idx)

                    pop.reverse()

                    for to_pop in pop:
                        points.pop(to_pop)

                    connector_geom = LineString(points)

                    # x_val = [i[0] for i in points]
                    # y_val = [i[1] for i in points]
            
                    # plt.scatter(x_val, y_val)
                    # plt.scatter(xt, yt, alpha=0.2)
                    # plt.scatter(connection_points1[:,0], connection_points1[:,1], color='r')
                    # plt.scatter(connection_points2[:,0], connection_points2[:,1], color='r')
                    # plt.plot(polygons[element_id].bounds.nodes_utm[:,0], polygons[element_id].bounds.nodes_utm[:,1])
                    # plt.show()

                    lane_connections = {'lane_id': lane_id, 'intersection_id': element_id, 'successor': successor, 'connection_line': connector_geom}
                    lane_connectors.append(lane_connections)
        
        return lane_connectors

    def interpolate_lane_connector(self, x, y):

        points = np.array([x, y]).T

        # Linear length along the line:
        distance = np.cumsum(np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )))
        distance = np.insert(distance, 0, 0)/distance[-1]

        n = np.linspace(0, 1, 75)

        interpolator =  interp1d(distance, points, kind='quadratic', axis=0)
        connector_nodes = interpolator(n)

        out_x = connector_nodes.T[0]
        out_y = connector_nodes.T[1]

        return out_x, out_y
    
    def get_neighbouring_lanes(self, lane_id, d_threshold) -> list:
        """
        Get all neighbouring lane segments of a given lane segment
        :param lane_id: lane identifier for which to execute the operation
        :return left_neighbours: List of lane segment identifiers that are left neighbours
        :return right_neighbours: List of lane segment identifiers that are right neighbours
        """

        # TODO clean function, add functionality of boundary type
        left_line, right_line = self.lanes[lane_id].left_boundary, self.lanes[lane_id].right_boundary

        left_line = LineString(left_line.nodes_utm)
        right_line = LineString(right_line.nodes_utm)
        
        potential_neighbours = []
        left_neighbours = []
        right_neighbours = []

        # TODO revise lane numbers of Hidde's annotations. 
        for i in range(1, 120):
        #for i in range(1,len(self.lanes)): 
            if self.lanes[lane_id].left_boundary.type == 2 or self.lanes[lane_id].right_boundary.type == 2:
                if self.lanes[i].left_boundary.type == 2 or self.lanes[i].right_boundary.type == 2:
                    if (self.lanes[i].id != lane_id) and (self.lanes[i].id not in potential_neighbours):
                        potential_neighbours.append(self.lanes[i].id)
            
        for neighbour in potential_neighbours:
            left_line_other, right_line_other = self.lanes[neighbour].left_boundary, self.lanes[neighbour].right_boundary

            left_line_other = LineString(left_line_other.nodes_utm)
            right_line_other = LineString(right_line_other.nodes_utm)

            if self.lanes[lane_id].left_boundary.type == 2 and self.lanes[neighbour].right_boundary.type == 2:
                distance1 = LineString(nearest_points(left_line, right_line_other)).length

                if distance1 <= d_threshold:
                    left_neighbours.append(neighbour)

            if self.lanes[lane_id].right_boundary.type == 2 and self.lanes[neighbour].left_boundary.type == 2:
                distance2 = LineString(nearest_points(right_line, left_line_other)).length

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
        lon, lat = self.nodes_lonlat
        nodes_utm = utm.from_latlon(lat, lon)
        self.utm_zone = nodes_utm[2:]
        return np.stack(nodes_utm[:2], axis=-1)

    @property
    def nodes_lonlat(self):
        trans = CoordTransformer()
        nodes_global = list(trans.t_global_nl(self.nodes[:,0], self.nodes[:,1]))
        return nodes_global

    def _get_nodes_in_frame(self, frame):
        if frame == 'nl':
            return self.nodes
        elif frame == 'lonlat':
            return self.nodes_lonlat
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
        left_line = self.left_boundary.interpolate(frame='nl')
        right_line = self.right_boundary.interpolate(frame='nl')
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

    df2 = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/Intersections.gpkg')
    data2 = gpd.GeoDataFrame.explode(df2, index_parts=False)

    lanes = Lanes().from_df(data)
    lane_id = 2
    polygons = Polygons().from_df(data2)

    # for lane_id, lane in lanes.lanes.items():
    #     #print(lane_example.centerline.nodes_utm)
    #     try:
    #         lbound_nodes = lane.left_boundary.nodes_utm
    #         lbound = lane.left_boundary.interpolate()
    #     except ValueError:
    #         continue
    #     #print(lbound.shape)
    #     plt.title(f'{lane_id}')
    #     plt.plot(lbound_nodes[:,0], lbound_nodes[:,1],'or')
    #     plt.plot(lbound[:,0], lbound[:,1],'-k')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.show()
    

    #box = [[],[]] 
    
    element_ids = 90
    global_pose = 593254.12215427, 5763031.3443488 # Global coordinates of the prius at frame location within lanes file
    map_extent = 100

    #lanes.get_lanes_in_box(global_pose=global_pose, map_extent=100, frame="utm")
    #lane_ids = list(set(data['lane_id'].values))
    lanes.get_lane_connections(lane_id=lane_id, element_ids=90)


        
