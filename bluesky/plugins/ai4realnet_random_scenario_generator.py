"""
    AI4REALNET - BlueSky plugin for deploying RL-based model
    ENV: StaticObstacleSectorCREnv-v0
    Authors: Giulia Leto
"""
from bluesky import core, stack, traf, tools, settings 
import numpy as np
import bluesky as bs
import pandas as pd
from bluesky.network.publisher import state_publisher, StatePublisher
from bluesky.plugins.ai4realnet_deploy_RL_tools import functions, constants_general

import debug

# Global data
N_AC = 20  # Number of aircraft in the randomised scenario
N_OBSTACLES = 5 

sector_name = 'LISBON_FIR'
latitude_bounds = (31.4, 43.0)
longitude_bounds = (-18.3, -6.1)

# smaller bounds for testing stage
latitude_bounds = (33.0, 36.0)
longitude_bounds = (-18.0, -12.0)

scenario_generator = None

def init_plugin():
    global scenario_generator
    scenario_generator = ScenarioGenerator()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'scn_gen',
        # The type of this plugin.
        'plugin_type':     'sim',
        }
    
    stackfunctions = {
        'INITIALIZE_SCENARIO': [
            'INITIALIZE_SCENARIO [N_AC] [N_OBSTACLES]',
            '[int] [int]',
            scenario_generator.initialize_scenario,
            'Generates a random scenario inside of Lisbon FIR',
        ]
    }

    return config, stackfunctions

class ScenarioGenerator(core.Entity):  
    def __init__(self):
        super().__init__()
        self.scn_idx = 0
        
        # logging
        import debug
        debug.light_blue(f'initialised SCN_GEN at {self.scn_idx}')
        self.initialise_observation_flag = False
    
    def reset(self):
        import debug
        debug.light_green(f'reset SCN_GEN at {self.scn_idx}')
        
        # stack.process('pcall ai4realnet_deploy_RL/sector.scn;initialize_scenario;DTMULT 5000')
        # stack.stack('initialize_scenario 20 5')
        # self.initialise_observation_flag = True

    # @stack.command(name ='INITIALIZE_SCENARIO', annotations= '', aliases=('INIT_SCENARIO', 'INITIALISE_SCENARIO'))
    def initialize_scenario(self, number_aircraft: int = N_AC, number_obstacles: int = N_OBSTACLES):
        """
        Initialize a new random scenario with the specified number of aircraft and obstacles.

        Args:
            number_aircraft (int): Number of aircraft to generate in the scenario.
            number_obstacles (int): Number of random restricted areas to create.

        Example:
            INITIALIZE_SCENARIO 20 5
        """
            # bs.sim.step()
        stack.process('pcall ai4realnet_deploy_RL/sector.scn')
        stack.process('pcall ai4realnet_deploy_RL/config_screen')

        # check if Lisbon FIR is already loaded
        if bs.tools.areafilter.basic_shapes.get(sector_name) is None:
            raise Exception(f"{sector_name} not loaded")

        self.sample_obstacle = True
        while self.sample_obstacle:
            self._generate_random_restricted_areas(number_obstacles)

        _generate_random_aircraft(number_aircraft, sector_name, self.obstacle_names, latitude_bounds, longitude_bounds)

        # bs.sim.step()


    def _generate_random_restricted_areas(self, num_obstacles: int):
        altitude = 350

        # delete existing obstacles from previous episode in BlueSky ##DELETE after reset is functioning. 
        all_obstacle_names = []
        for shape_name, shape in tools.areafilter.basic_shapes.items():
            # print(f'Found existing obstacle: {shape_name}')
            all_obstacle_names.append(shape_name)
            
        for shape_name in all_obstacle_names:
            if shape_name != sector_name:
                # print(f'Deleting existing obstacle: {shape_name}')
                bs.tools.areafilter.deleteArea(shape_name)
                # stack.process(f"DELETE {shape_name}")

        obstacle_names = []
        obstacle_vertices = []
        obstacle_radius = []

        self.obstacle_names = []
        self.obstacle_vertices = []
        self.obstacle_radius = []

        self._generate_coordinates_centre_obstacles(num_obstacles)

        obstacle_dict = {}  # Initialize the dictionary to store obstacles for overlap checking

        for i in range(num_obstacles):

            centre_obst = (self.obstacle_centre_lat[i], self.obstacle_centre_lon[i])
            _, p, R = self._generate_polygon(centre_obst)
            
            # Ensure the polygon is closed
            if not np.allclose(p[0], p[-1]):
                p.append(p[0])

            points = [coord for point in p for coord in point] # Flatten the list of points
            poly_name = 'RESTRICTED_AREA_' + str(i+1)
            # bs.tools.areafilter.defineArea(poly_name, 'POLY', points)

            stack.process(f'POLY {poly_name}, ' + ', '.join(map(str, points)))
            stack.process(f"COLOR {poly_name}, RED")

            # stack.stack(f'POLY {poly_name}, ' + ', '.join(map(str, points)))
            # stack.stack(f"COLOR {poly_name}, RED")

            # stack.stack(f"ECHO COLOR {poly_name}, RED")
            # print(f'Defined obstacle {poly_name} with vertices: {points}')

            obstacle_vertices_coordinates = []
            for k in range(0,len(points),2):
                obstacle_vertices_coordinates.append([points[k], points[k+1]])
            # print(f'Obstacle vertices coordinates: {obstacle_vertices_coordinates}')
            # overlap = bs.tools.areafilter.checkInside(poly_name, np.array([bs.traf.lat[ac_idx]]), np.array([bs.traf.lon[ac_idx]]), np.array([bs.traf.alt[ac_idx]]))[0]
            # if overlap:
            #     self.sample_obstacle = True
            #     return

            obstacle_names.append(poly_name)
            obstacle_vertices.append(obstacle_vertices_coordinates)
            obstacle_radius.append(R)
        # bs.sim.step()

        for i in range(num_obstacles):
            overlap_list = []  # List to store overlap information
            # Check for overlaps with existing obstacles
            for j in range(num_obstacles):
                if i == j:
                    continue  # Skip checking the same obstacle

                found_overlap = False
                for k in range(0, len(obstacle_vertices[j])):
                    # check if the vertices of the obstacle are inside the other obstacles
                    overlap = bs.tools.areafilter.checkInside(obstacle_names[i], np.array([obstacle_vertices[j][k][0]]), np.array([obstacle_vertices[j][k][1]]), np.array([altitude]))[0]
                    if overlap:
                        overlap_list.append(obstacle_names[j])
                        break #break vertex loop
                    # check if points along the edges of the obstacle are inside the other obstacles
                    if k == len(obstacle_vertices[j]) -1:
                        interpolated_points = functions.interpolate_along_obstacle_vertices(obstacle_vertices[j][k], obstacle_vertices[j][0])
                    else:
                        interpolated_points = functions.interpolate_along_obstacle_vertices(obstacle_vertices[j][k], obstacle_vertices[j][k+1])
                    for point in interpolated_points:
                        overlap = bs.tools.areafilter.checkInside(obstacle_names[i], np.array([point[0]]), np.array([point[1]]), np.array([altitude]))[0]
                        if overlap:
                            overlap_list.append(obstacle_names[j])
                            found_overlap = True
                            break # break interpolation loop
                    if found_overlap:
                        break
        
            obstacle_dict[obstacle_names[i]] = overlap_list
        
        max_overlaps_allowed = 0

        too_many_overlapping_obstacles = any(len(overlaps) > max_overlaps_allowed for overlaps in obstacle_dict.values())
        # too_many_overlapping_obstacles = False  # Temporarily disable overlap checking for testing
        # print(f'Overlap information: {obstacle_dict}')

        if too_many_overlapping_obstacles:
            self.sample_obstacle = True
        else:
            self.sample_obstacle = False

            # for index, name in enumerate(obstacle_names):
            #     (center_latlon, radius_m) = RLtools.functions.bounding_circle_geodesic(obstacle_vertices[index])

            # for shape_name, shape in tools.areafilter.basic_shapes.items():
            #     # print(f'Restricted area name: {shape_name}')
            #     if shape_name != 'LISBON_FIR':
            #         coordinates = shape.coordinates
            #         latitudes = coordinates[::2]
            #         longitudes = coordinates[1::2]
            #         (center_latlon, radius_m) = RLtools.functions.bounding_circle_geodesic(list(zip(latitudes, longitudes)))
                    
            #         self.obstacle_centre_lat.append(center_latlon[0])
            #         self.obstacle_centre_lon.append(center_latlon[1])
            #         self.obstacle_radius.append(radius_m)

                    # Add circles around the restricted area for visualization of the observation
                    # print(f'Obstacle {shape_name} center: {center_latlon}, radius: {radius_m}')
                    # stack.stack(f"CIRCLE {shape_name}_bounding_circle, {center_latlon[0]}, {center_latlon[1]}, {radius_m/RLtools.constants.NM2KM/1000}")
                    # stack.stack(f"COLOR {shape_name}_bounding_circle, YELLOW")

            # Store the generated obstacles in the environment
            self.obstacle_names = obstacle_names
            self.obstacle_vertices = obstacle_vertices
            self.obstacle_radius = obstacle_radius

            # Scaling factor for the radius in the observation vector
            self.max_obstacle_radius = max(self.obstacle_radius)
            # Find observation points for sector

    def _generate_polygon(self, centre):
        OBSTACLE_AREA_RANGE = (1000, 10000) # In NM^2

        poly_area = np.random.randint(OBSTACLE_AREA_RANGE[0]*2, OBSTACLE_AREA_RANGE[1])
        R = np.sqrt(poly_area/ np.pi)
        p = [functions.random_point_on_circle(R) for _ in range(3)] # 3 random points to start building the polygon
        p = functions.sort_points_clockwise(p)
        p_area = functions.polygon_area(p)
        
        while p_area < OBSTACLE_AREA_RANGE[0]:
            p.append(functions.random_point_on_circle(R))
            p = functions.sort_points_clockwise(p)
            p_area = functions.polygon_area(p)
        
        p = [functions.nm_to_latlong(centre, point) for point in p] # Convert to lat/long coordinateS
        
        return p_area, p, R

    def _generate_coordinates_centre_obstacles(self, num_obstacles: int):
        self.obstacle_centre_lat = []
        self.obstacle_centre_lon = []
        OBSTACLE_DISTANCE_MIN = 20 # KM
        OBSTACLE_DISTANCE_MAX = 500 # KM

        for i in range(num_obstacles):
            obstacle_dis_from_reference = np.random.randint(OBSTACLE_DISTANCE_MIN, OBSTACLE_DISTANCE_MAX)
            obstacle_hdg_from_reference = np.random.randint(0, 360)
            # ac_idx = bs.traf.id2idx(acid)

            obstacle_centre_lat, obstacle_centre_lon = functions.get_point_at_distance(latitude_bounds[0] + (latitude_bounds[1]-latitude_bounds[0])/2, longitude_bounds[0] + (longitude_bounds[1]-longitude_bounds[0])/2, obstacle_dis_from_reference, obstacle_hdg_from_reference)    
            self.obstacle_centre_lat.append(obstacle_centre_lat)
            self.obstacle_centre_lon.append(obstacle_centre_lon)

def _generate_random_aircraft(n_ac, sector_name, obstacle_names, latitude_bounds, longitude_bounds):
    """
    Generate random scenario with random initial positions and random destinations for the aircraft
    """
    # HARDCODED
    orig_altitude = 350
    dest_altitude = 350

    # if the aircraft is generated inside the sector, keep it, otherwise regenerate
    min_lat, max_lat = latitude_bounds
    min_lon, max_lon = longitude_bounds

    lat_orig = np.full(n_ac, np.nan)
    lon_orig = np.full(n_ac, np.nan)
    good = np.zeros(n_ac, dtype=bool)

    max_iter = 10_000  # safety guard
    counter = 0

    while np.isnan(lat_orig).any():
        counter += 1
        if counter > max_iter:
            raise RuntimeError("ORIG Too many iterations - check bounds/sector overlap.")

        # indices that still need valid points
        # remaining_idx = np.isnan(lat_orig)
        remaining_idx = np.flatnonzero(~good)
        m = remaining_idx.size

        # sample only for the remaining positions
        lat_try = np.random.uniform(min_lat, max_lat, size=m)
        lon_try = np.random.uniform(min_lon, max_lon, size=m)
        altitude = np.ones(m)*orig_altitude

        inside_sector = bs.tools.areafilter.checkInside(
            sector_name,
            lat_try,
            lon_try,
            altitude
        ).astype(bool)

        # debug.cyan(f'inside_sector: {inside_sector}')
        inside_any_restricted = np.zeros_like(inside_sector, dtype=bool)
        # debug.cyan(f'inside_any_restricted: {inside_any_restricted}')

        for name in obstacle_names:
            # print(f'name: {name}')

            inside_restricted_area = bs.tools.areafilter.checkInside(
                name,
                lat_try,
                lon_try,
                altitude
            ).astype(bool)
            # debug.cyan(f'inside_restricted: {inside_restricted_area}')
            inside_any_restricted |= inside_restricted_area
            # debug.cyan(f'inside_any_restricted: {inside_any_restricted}')


        inside = inside_sector & (~inside_any_restricted)
        # debug.cyan(f'inside: {inside}')

        # place successful samples into their slots
        lat_orig[remaining_idx[inside]] = lat_try[inside]
        lon_orig[remaining_idx[inside]] = lon_try[inside]
        good[remaining_idx[inside]] = True

    lat_dest = np.full(n_ac, np.nan)
    lon_dest = np.full(n_ac, np.nan)
    good = np.zeros(n_ac, dtype=bool)

    max_iter = 10_000  # safety guard
    counter = 0

    while np.isnan(lat_dest).any():
        counter += 1
        if counter > max_iter:
            raise RuntimeError("DEST Too many iterations — check bounds/sector overlap.")

        # indices that still need valid points
        # remaining_idx = np.isnan(lat_dest)
        remaining_idx = np.flatnonzero(~good)
        m = remaining_idx.size

        # sample only for the remaining positions
        lat_try = np.random.uniform(min_lat, max_lat, size=m)
        lon_try = np.random.uniform(min_lon, max_lon, size=m)
        altitude = np.ones(m)*dest_altitude

        inside_sector = bs.tools.areafilter.checkInside(
            sector_name,
            lat_try,
            lon_try,
            altitude
        ).astype(bool)
        # debug.red(f'inside_sector: {inside_sector}')

        inside_any_restricted = np.zeros_like(inside_sector, dtype=bool)
        # debug.red(f'inside_any_restricted: {inside_any_restricted}')

        for name in obstacle_names:
            # print(f'name: {name}')
            inside_restricted_area = bs.tools.areafilter.checkInside(
                name,
                lat_try,
                lon_try,
                altitude
            ).astype(bool)
            # debug.red(f'inside_restricted_area: {inside_restricted_area}, lat_try: {lat_try}, lon_try: {lon_try}, altitude: {altitude} ')
            inside_any_restricted |= inside_restricted_area
            # debug.red(f'inside_any_restricted: {inside_any_restricted}')

        inside = inside_sector & (~inside_any_restricted)
        # debug.red(f'inside: {inside}')

        # place successful samples into their slots
        lat_dest[remaining_idx[inside]] = lat_try[inside]
        lon_dest[remaining_idx[inside]] = lon_try[inside]
        good[remaining_idx[inside]] = True
        # print(f'lat_dest: {lat_dest}')
        # print(f'lon_dest: {lon_dest}')
    heading, _ = bs.tools.geo.kwikqdrdist(lat_orig, lon_orig, lat_dest, lon_dest)

    for ac_idx in range(n_ac):
        bs.stack.process(f'CRE AC{ac_idx+1}, B787, {lat_orig[ac_idx]}, {lon_orig[ac_idx]}, {heading[ac_idx]}, {orig_altitude}, 150')
        bs.stack.process(f'DEST AC{ac_idx+1} {lat_dest[ac_idx]} {lon_dest[ac_idx]}')
        # bs.stack.stack(f'CRE AC{ac_idx+1}, B787, {lat_orig[ac_idx]}, {lon_orig[ac_idx]}, {heading[ac_idx]}, {orig_altitude}, 150')
        # bs.stack.stack(f'DEST AC{ac_idx+1} {lat_dest[ac_idx]} {lon_dest[ac_idx]}')