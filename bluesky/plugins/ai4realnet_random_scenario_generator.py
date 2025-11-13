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

    @stack.command(name ='INITIALIZE_SCENARIO', annotations= '', aliases=('INIT_SCENARIO', 'INITIALISE_SCENARIO'))
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

        self.weather_active = False

        self.sample_obstacle = True
        while self.sample_obstacle:
            self._generate_random_restricted_areas(number_obstacles)

        _generate_random_aircraft(number_aircraft, sector_name, self.obstacle_names, latitude_bounds, longitude_bounds)

        # bs.sim.step()

    @core.timed_function(name='weather_disturbance', dt=constants_general.ACTION_FREQUENCY)
    def generate_weather_disturbance(self):
        """
        Implements ATM 1: Weather perturbations as adverse, drifting storm cells.

        Spawning:
            - With probability pweather (each call), spawn a new disturbance if none is active.
            - Lifetime is sampled from an exponential distribution with mean qweather (Poisson).

        Evolution:
            - Radius follows a triangular profile: base -> peak -> base over lifetime.
            - Center drifts with constant (sampled) heading/speed.
            - Shape is a polygon with per-vertex jitter (stable over the cell lifetime).

        Deletion:
            - When simt >= start + lifetime, the cell is removed.
        """
        # --- Short names to constants (adjust names to your actual constants) ---
        pweather = 0.03                      # spawn probability per tick
        qweather = 1200               # mean lifetime [s] for exponential
        min_r_nm = 6          # small seed radius
        peak_r_nm = 28        # peak radius
       
        # jitter   = 0.15 # per-vertex shape jitter factor, 15%
        vmin, vmax = (5.0, 100.0)  # speed range [kt]
        NM2KM = constants_general.NM2KM
        weather_cell_lifetime_min = 600  # minimum lifetime [s]
        shape_name = 'WEATHER_CELL'
        simt = bs.sim.simt
        # dt   = (simt - getattr(self, "weather_last_update_t", simt))
        # self.weather_last_update_t = simt
        dt=constants_general.ACTION_FREQUENCY

        # States
        if not hasattr(self, "weather_active"):
            self.weather_active = False
            self.weather_shape_jitter = None  # per-vertex scale factors (stable)
            self.weather_angles_deg = None

        if self.weather_active:

            if (simt - self.weather_disturbance_start) >= self.weather_cell_lifetime:
                # Delete weather disturbance (expired)
                stack.process(f"DELETE {shape_name}")
                self.weather_active = False
                self.weather_shape_jitter = None
                self.weather_angles_deg = None
            else:
                # Evolve the active disturbance
                age_s   = simt - self.weather_disturbance_start
                tau     = age_s / max(1e-6, self.weather_cell_lifetime) # [0,1] progress through lifetime
                # Triangular growth-shrink radius profile
                if tau <= 0.5:
                    radius_nm = min_r_nm + 2.0 * tau * (peak_r_nm - min_r_nm)
                else:
                    radius_nm = peak_r_nm - 2.0 * (tau - 0.5) * (peak_r_nm - min_r_nm)

                # Drift center since last update
                if self.weather_cell_speed > 0.0:
                    # distance in NM = speed[kt]*dt[h]
                    distance_km = self.weather_cell_speed * (dt / 3600.0) * NM2KM
                    new_lat, new_lon = functions.get_point_at_distance(self.weather_cell_center_lat, self.weather_cell_center_lon, distance_km, self.weather_cell_heading)
                    self.weather_cell_center_lat = new_lat
                    self.weather_cell_center_lon = new_lon

                
                # Build polygon with stable jitter
                if self.weather_shape_jitter is None:
                    # Stable per-vertex jitter and angles for the lifetime of the cell
                    jitter = 0.25
                    self.weather_angles_deg = np.linspace(0.0, 360.0, num=self.n_verts, endpoint=False)
                    self.weather_shape_jitter = 1.0 + np.random.uniform(-jitter, jitter, size=self.n_verts)
                    import debug
                    debug.pink(f'Is this ever triggered?')
                else:
                    jitter = 0.05
                    self.weather_shape_jitter = 1.0 + np.random.uniform(-jitter, jitter, size=self.n_verts)

                weather_cell_latitude_centre, weather_cell_longitude_centre = self.weather_cell_center_lat, self.weather_cell_center_lon
                weather_cell_poly = []
                weather_cell_radius = []
                for ang_deg, jit in zip(self.weather_angles_deg, self.weather_shape_jitter):
                    radius_jittered_km = radius_nm * float(jit) * NM2KM
                    weather_cell_radius.append(radius_jittered_km)
                    vertex_lat, vertex_lon = functions.get_point_at_distance(weather_cell_latitude_centre, weather_cell_longitude_centre, radius_jittered_km, ang_deg)
                    weather_cell_poly.append((vertex_lat, vertex_lon))
                # close polygon
                weather_cell_poly.append(weather_cell_poly[0])
                self.weather_cell_radius = max(weather_cell_radius)

                # Re-draw polygon (delete + poly keeps syntax simple)
                stack.process(f"DELETE {shape_name}")
                flat = ", ".join([f"{lat:.6f}, {lon:.6f}" for (lat, lon) in weather_cell_poly])
                stack.process(f"POLY {shape_name}, {flat}")
                stack.process(f"COLOR {shape_name}, BLUE")
                # stack.process(f"CIRCLE {shape_name}_bounding_circle, {self.weather_cell_center_lat}, {self.weather_cell_center_lon}, {self.weather_cell_radius/constants_general.NM2KM}")
                # stack.process(f"COLOR {shape_name}_bounding_circle, YELLOW")

        else: # No active cell: sample a new one with probability pweather

            # If there is no weather distrubance active, sample with probality pweather
            weather_disturbance_flag = (np.random.rand() < pweather) and (not self.weather_active)

            if weather_disturbance_flag:
                self.n_verts  = np.random.randint(10, 16)            # polygon vertex count
                # Sample lifetime ~ Exponential(mean=qweather)
                # (Poisson process waiting time)
                weather_cell_lifetime = float(np.random.exponential(scale=max(qweather, 1e-6)))
                # Minimum lifetime guard
                weather_cell_lifetime = max(weather_cell_lifetime, weather_cell_lifetime_min)

                # Random center within sector bounds
                weather_cell_latitude_centre, weather_cell_longitude_centre = np.random.uniform(latitude_bounds[0], latitude_bounds[1]), np.random.uniform(longitude_bounds[0], longitude_bounds[1])

                # Sample stable drift and shape
                self.weather_cell_heading = int(np.random.randint(0, 360)) # deg
                self.weather_cell_speed = float(np.random.uniform(vmin, vmax)) # kts

                # Jitter the vertices of each weather cell
                jitter = 0.25
                self.weather_angles_deg = np.linspace(0.0, 360.0, num=self.n_verts, endpoint=False)
                self.weather_shape_jitter = 1.0 + np.random.uniform(-jitter, jitter, size=self.n_verts)

                # Start the weather disturbance with a small radius
                seed_radius_nm = float(np.random.uniform(0.5 * min_r_nm, min_r_nm))

                # Draw initial polygon
                weather_cell_poly = []
                weather_cell_radius = []
                for ang_deg, jit in zip(self.weather_angles_deg, self.weather_shape_jitter):
                    radius_jittered_km = seed_radius_nm * float(jit) * NM2KM
                    weather_cell_radius.append(radius_jittered_km)
                    vertex_lat, vertex_lon = functions.get_point_at_distance(weather_cell_latitude_centre, weather_cell_longitude_centre, radius_jittered_km, ang_deg)
                    weather_cell_poly.append((vertex_lat, vertex_lon))
                # close polygon
                weather_cell_poly.append(weather_cell_poly[0])

                flattened_polygon = ", ".join([f"{lat:.6f}, {lon:.6f}" for (lat, lon) in weather_cell_poly])
                stack.process(f"POLY {shape_name}, {flattened_polygon}")
                stack.process(f"COLOR {shape_name}, BLUE")

                # Persist state
                self.weather_active = True
                self.weather_cell_center_lat = float(weather_cell_latitude_centre)
                self.weather_cell_center_lon = float(weather_cell_longitude_centre)
                self.weather_disturbance_start = float(simt)
                self.weather_cell_lifetime = float(weather_cell_lifetime)
                self.weather_cell_radius = max(weather_cell_radius)
                # stack.process(f"CIRCLE {shape_name}_bounding_circle, {self.weather_cell_center_lat}, {self.weather_cell_center_lon}, {self.weather_cell_radius/constants_general.NM2KM}")
                # stack.process(f"COLOR {shape_name}_bounding_circle, YELLOW")

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