"""
    AI4REALNET - BlueSky plugin for deploying RL-based model
    ENV: StaticObstacleSectorCREnv-v0
    Authors: Giulia Leto
"""
from bluesky import core, stack, traf, tools, settings 
from stable_baselines3 import SAC
import numpy as np
import bluesky.plugins.ai4realnet_deploy_RL_tools.StaticObstacleSectorCREnv as RLtools
import bluesky as bs
import pandas as pd
from bluesky.network.publisher import state_publisher, StatePublisher

import debug

algorithm = 'SAC'
env_name = 'StaticObstacleSectorCREnv-v0'

N_AC = 20  # Number of aircraft in the randomised scenario
N_SCN = 10 # Number of testing iterations

sector_name = 'LISBON_FIR'
latitude_bounds = (31.4, 43.0)
longitude_bounds = (-18.3, -6.1)

# smaller bounds for testing stage
latitude_bounds = (33.0, 36.0)
longitude_bounds = (-18.0, -12.0)

def init_plugin():
    StaticObstacleSectorCR_randomised_scenario_loading = DeployRL()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'DEPLOY_RL_StaticObstacleSectorCREnv_randomised_scenario', #'Deploy_RL', #
        # The type of this plugin.
        'plugin_type':     'sim',
        }
    return config

class DeployRL(core.Entity):  
    def __init__(self):
        super().__init__()
        self.model = SAC.load(f"bluesky/plugins/ai4realnet_deploy_RL_tools/models/{env_name}/{env_name}_{algorithm}/model", env=None)
        self.scn_idx = 0
        
        # logging
        self.log_buffer = []   # temporary storage
        self.csv_file = (f"output/ai4realnet_deploy_RL_{env_name}_log.csv")
        import debug
        debug.light_blue(f'initialised at {self.scn_idx}')
        self.initialise_observation_flag = False
    def reset(self):
        import debug
        debug.light_green(f'reset at {self.scn_idx}')
        
        # for shape_name, _ in tools.areafilter.basic_shapes.items():
        #     print(f'Existing obstacle after reset: {shape_name}')


        # stack.process('pcall ai4realnet_deploy_RL/sector.scn;initialize_scenario;DTMULT 5000')
        stack.stack('pcall ai4realnet_deploy_RL/sector.scn')
        stack.stack('pcall ai4realnet_deploy_RL/config_screen')
        stack.stack('initialize_scenario')
        stack.stack(f'SAVEIC test_{self.scn_idx}')
        debug.light_blue(f'called save ic for {self.scn_idx}')
        self.initialise_observation_flag = True




    @stack.command
    def initialize_scenario(self):
        # bs.sim.step()

        self.weather_active = False

        self.sample_obstacle = True
        while self.sample_obstacle:
            self._generate_random_restricted_areas(RLtools.constants.NUM_OBSTACLES)

        RLtools.functions.generate_random_aircraft(N_AC, sector_name, self.obstacle_names, latitude_bounds, longitude_bounds)

        # waypoint_distances = []
        # # Give aircraft initial heading
        # for ac_idx, id in enumerate(traf.id):
        #     initial_wpt_qdr, initial_wpt_dist = tools.geo.kwikqdrdist(traf.lat[ac_idx], traf.lon[ac_idx], bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1])
        #     bs.traf.hdg[ac_idx] = initial_wpt_qdr
        #     waypoint_distances.append(initial_wpt_dist)

        # bs.sim.step()

    @core.timed_function(name='StaticObstacleSectorCR', dt=RLtools.constants.ACTION_FREQUENCY)
    def update(self):

        # if self.initialise_observation_flag:
        #     for shape_name, _ in tools.areafilter.basic_shapes.items():
        #         print(f'Existing obstacle at first update after reset: {shape_name}')

        for ac_idx, id in enumerate(traf.id):
            _, dest_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1])
            if dest_dis * RLtools.constants.NM2KM < RLtools.constants.DISTANCE_MARGIN:
                # stack.stack(f"ECHO Aircraft {id} has reached the destination waypoint.")
                # print(f"Aircraft {id} has reached the destination waypoint.")
                stack.process(f"DELETE {id}")
                continue  # skip action if the aircraft has reached its destination
            obs = self._get_obs(ac_idx)
            # action, _ = self.model.predict(obs, deterministic=True)
            # self._set_action(action, ac_idx)

        # --- logging ---
        simt = bs.sim.simt        # current sim time
        for ac_id, hdg in zip(bs.traf.id, bs.traf.hdg):
            self.log_buffer.append({
                "simt": simt,
                "id": ac_id,
                "hdg": hdg,
            })

        # flush every 100 rows
        if len(self.log_buffer) >= 5:
            df = pd.DataFrame(self.log_buffer)
            df.to_csv(self.csv_file, mode="a", index=False, header=not pd.io.common.file_exists(self.csv_file))
            self.log_buffer.clear()

        if traf.id == [] and (self.scn_idx < N_SCN+1):
            self.scn_idx += 1
            # for some reason, the weather cell on the screen is not deleted when reset is issued 
            stack.stack(f"DELETE WEATHER_CELL")
            # bs.tools.areafilter.deleteArea('WEATHER_CELL')
            bs.sim.step()
            stack.stack('RESET')
            # stack.process('RESET')

        if traf.id == [] and (self.scn_idx == N_SCN+1):
            stack.process('QUIT')
            # stack.process('QUIT')

    def _get_obs(self, ac_idx):
        """
        Observation is the normalized. Normalisation logic should be studied further
        """
        if self.initialise_observation_flag == True:
            waypoint_distances = []

            # Give aircraft initial heading
            for ac_idx, id in enumerate(traf.id):
                initial_wpt_qdr, initial_wpt_dist = tools.geo.kwikqdrdist(traf.lat[ac_idx], traf.lon[ac_idx], bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1])
                # bs.traf.hdg[ac_idx] = initial_wpt_qdr
                waypoint_distances.append(initial_wpt_dist)

            # Scaling factor for the distances in the observation vector
            self.waypoint_distance_max = max(waypoint_distances)

            sector = tools.areafilter.basic_shapes[sector_name]
            coordinates = sector.coordinates
            latitudes = coordinates[::2]
            longitudes = coordinates[1::2]
            self.sector_points = RLtools.functions.resample_closed_border(latitudes, longitudes, RLtools.constants.TOTAL_OBSERVATION_POINTS)

            # # Display in BlueSky the sector points contained in the observation 
            # print(f'sector points: {self.sector_points}, shape: {np.array(self.sector_points).shape}, type: {type(self.sector_points)}')
            # coordinates = ", ".join([f"{lat} {lon}" for lat, lon in self.sector_points])
            # stack.stack(f"POLY SECTOR_OBSERVATION, {coordinates}")

            self.initialise_observation_flag = False
        
        # intruder observation
        intruders_lat = np.delete(bs.traf.lat, ac_idx)
        intruders_lon = np.delete(bs.traf.lon, ac_idx)
        
        int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], intruders_lat, intruders_lon)

        # index of the closes N_INTRUDERS intruders for arrays of intruders_lat, intruders_lon, int_qdr, int_dis, intruders_speed, intruders_heading: the ownship is excluded a priori in these
        closest_intruders_idx = np.argsort(int_dis)[:RLtools.constants.NUM_INTRUDERS]

        intruder_distance = int_dis[closest_intruders_idx] * RLtools.constants.NM2KM

        # relative heading
        bearing = bs.traf.hdg[ac_idx] - int_qdr[closest_intruders_idx]
        for bearing_idx in range(len(bearing)):
            bearing[bearing_idx] = RLtools.functions.bound_angle_positive_negative_180(bearing[bearing_idx])

        intruder_cos_bearing = np.cos(np.deg2rad(bearing))
        intruder_sin_bearing = np.sin(np.deg2rad(bearing))

        intruders_heading = np.delete(bs.traf.hdg, ac_idx)
        intruders_speed = np.delete(bs.traf.gs, ac_idx)
        heading_difference = bs.traf.hdg[ac_idx] - intruders_heading[closest_intruders_idx]

        # relative speed
        intruder_x_difference_speed = - np.cos(np.deg2rad(heading_difference)) * intruders_speed[closest_intruders_idx]
        intruder_y_difference_speed = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * intruders_speed[closest_intruders_idx]

        # padding the observation if less than NUM_INTRUDERS are present
        if len(closest_intruders_idx) < RLtools.constants.NUM_INTRUDERS:
            num_missing_intruders = RLtools.constants.NUM_INTRUDERS - len(closest_intruders_idx)
            intruder_distance = np.pad(intruder_distance, (0, num_missing_intruders), 'constant', constant_values=(self.waypoint_distance_max,))
            intruder_cos_bearing =  np.pad(intruder_cos_bearing, (0, num_missing_intruders), 'constant', constant_values=(0,))
            intruder_sin_bearing =  np.pad(intruder_sin_bearing, (0, num_missing_intruders), 'constant', constant_values=(0,))
            intruder_x_difference_speed =  np.pad(intruder_x_difference_speed, (0, num_missing_intruders), 'constant', constant_values=(0,))
            intruder_y_difference_speed =  np.pad(intruder_y_difference_speed, (0, num_missing_intruders), 'constant', constant_values=(0,))

        # destination waypoint
        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1])
    
        destination_waypoint_distance = wpt_dis * RLtools.constants.NM2KM

        drift = bs.traf.hdg[ac_idx] - wpt_qdr
        destination_waypoint_drift = RLtools.functions.bound_angle_positive_negative_180(drift)

        destination_waypoint_cos_drift = np.cos(np.deg2rad(destination_waypoint_drift))
        destination_waypoint_sin_drift = np.sin(np.deg2rad(destination_waypoint_drift))

        # obstacles 
        obstacle_centre_distance = []
        obstacle_centre_cos_bearing = []
        obstacle_centre_sin_bearing = []

        for obs_idx in range(RLtools.constants.NUM_OBSTACLES):
            obs_centre_qdr, obs_centre_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.obstacle_centre_lat[obs_idx], self.obstacle_centre_lon[obs_idx])
            obs_centre_dis = obs_centre_dis * RLtools.constants.NM2KM #KM        
            bearing = bs.traf.hdg[ac_idx] - obs_centre_qdr
            
            bearing = RLtools.functions.bound_angle_positive_negative_180(bearing)

            obstacle_centre_distance.append(obs_centre_dis)
            obstacle_centre_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            obstacle_centre_sin_bearing.append(np.sin(np.deg2rad(bearing)))

        # weather disturbances
        if self.weather_active:
            weather_cell_centre_qdr, weather_cell_centre_distance = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.weather_cell_center_lat, self.weather_cell_center_lon)
            weather_cell_centre_distance = weather_cell_centre_distance * RLtools.constants.NM2KM #KM            
            bearing = RLtools.functions.bound_angle_positive_negative_180(bs.traf.hdg[ac_idx] - weather_cell_centre_qdr)

            weather_cell_centre_cos_bearing = np.cos(np.deg2rad(bearing))
            weather_cell_centre_sin_bearing = np.sin(np.deg2rad(bearing))

        # sector polygon edges observation
        sector_points_distance = []
        sector_points_cos_drift = []
        sector_points_sin_drift = []

        # Calculate distance and bearing from the ownship to each of the sector points
        for point_index in range(len(self.sector_points)):
            sector_points_qdr, sector_points_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.sector_points[point_index,0],self.sector_points[point_index,1])
            # point(f'point_index: {point_index}, sector_points_dis: {sector_points_dis}, sector_points_qdr: {sector_points_qdr}')
            sector_points_distance.append(sector_points_dis * RLtools.constants.NM2KM)

            drift = bs.traf.hdg[ac_idx] - sector_points_qdr

            drift = RLtools.functions.bound_angle_positive_negative_180(drift)

            sector_points_cos_drift.append(np.cos(np.deg2rad(drift)))
            sector_points_sin_drift.append(np.sin(np.deg2rad(drift)))
        
        obstacle_radius = self.obstacle_radius
        ## Find the RLtools.constants.NUM_OBSTACLES closest obstacles to the ownship (restricted areas or weather cells)
        if self.weather_active:
            obstacle_centre_distance.append(weather_cell_centre_distance)
            obstacle_centre_cos_bearing.append(weather_cell_centre_cos_bearing)
            obstacle_centre_sin_bearing.append(weather_cell_centre_sin_bearing)
            obstacle_radius.append(self.weather_cell_radius)

            obstacle_centre_distance = np.array(obstacle_centre_distance)
            obstacle_centre_cos_bearing = np.array(obstacle_centre_cos_bearing)
            obstacle_centre_sin_bearing = np.array(obstacle_centre_sin_bearing)
            obstacle_radius = np.array(obstacle_radius)

            idx_sorted = np.argsort(obstacle_centre_distance)[:RLtools.constants.NUM_OBSTACLES]

            obstacle_centre_distance = obstacle_centre_distance[idx_sorted]
            obstacle_centre_cos_bearing = obstacle_centre_cos_bearing[idx_sorted]
            obstacle_centre_sin_bearing = obstacle_centre_sin_bearing[idx_sorted]
            obstacle_radius = obstacle_radius[idx_sorted]

        observation = {
                "intruder_distance": np.array(intruder_distance).reshape(-1)/self.waypoint_distance_max,
                "intruder_cos_difference_pos": np.array(intruder_cos_bearing).reshape(-1),
                "intruder_sin_difference_pos": np.array(intruder_sin_bearing).reshape(-1),
                "intruder_x_difference_speed": np.array(intruder_x_difference_speed).reshape(-1)/RLtools.constants.AC_SPD,
                "intruder_y_difference_speed": np.array(intruder_y_difference_speed).reshape(-1)/RLtools.constants.AC_SPD,
                "destination_waypoint_distance": np.array(destination_waypoint_distance).reshape(-1)/self.waypoint_distance_max,
                "destination_waypoint_cos_drift": np.array(destination_waypoint_cos_drift).reshape(-1),
                "destination_waypoint_sin_drift": np.array(destination_waypoint_sin_drift).reshape(-1),
                # observations on obstacles
                "restricted_area_radius": np.array(obstacle_radius).reshape(-1)/self.max_obstacle_radius,
                "restricted_area_distance": np.array(obstacle_centre_distance).reshape(-1)/self.waypoint_distance_max,
                "cos_difference_restricted_area_pos": np.array(obstacle_centre_cos_bearing).reshape(-1),
                "sin_difference_restricted_area_pos": np.array(obstacle_centre_sin_bearing).reshape(-1),
                # observations on sector polygon edges and points along the edges
                "sector_points_distance": np.array(sector_points_distance).reshape(-1)/self.waypoint_distance_max,
                "sector_points_cos_drift": np.array(sector_points_cos_drift).reshape(-1),
                "sector_points_sin_drift": np.array(sector_points_sin_drift).reshape(-1)
            }
        # if len(closest_intruders_idx) < RLtools.constants.NUM_INTRUDERS:
        #     print(f'observation: {observation}')

        return observation

    def _set_action(self, action, ac_idx):
        """
        Control each aircraft separately
        """
        # print(f'New action')
        dv = action[1] * RLtools.constants.D_SPEED
        dh = action[0] * RLtools.constants.D_HEADING

        id = traf.id[ac_idx]
        heading_new = RLtools.functions.bound_angle_positive_negative_180(traf.hdg[ac_idx] + dh)
        speed_new = (traf.cas[ac_idx] + dv) * RLtools.constants.MpS2Kt
        # stack.stack(f"ECHO Aircraft {id} - New heading: {heading_new} deg, New speed: {speed_new/RLtools.constants.MpS2Kt} m/s")
        stack.stack(f"HDG {id} {heading_new}")
        stack.stack(f"SPD {id} {speed_new}")
        # print(f'Action for aircraft {id} - traf.hdg: {traf.hdg[ac_idx]} -> {heading_new} with dh {dh}, traf.cas: {traf.cas[ac_idx]} m/s -> {speed_new/RLtools.constants.MpS2Kt} with dv {dv} m/s')

    @core.timed_function(name='weather_disturbance', dt=RLtools.constants.ACTION_FREQUENCY)
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
        NM2KM = RLtools.constants.NM2KM
        weather_cell_lifetime_min = 600  # minimum lifetime [s]
        shape_name = 'WEATHER_CELL'
        simt = bs.sim.simt
        # dt   = (simt - getattr(self, "weather_last_update_t", simt))
        # self.weather_last_update_t = simt
        dt=RLtools.constants.ACTION_FREQUENCY

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
                    new_lat, new_lon = RLtools.functions.get_point_at_distance(self.weather_cell_center_lat, self.weather_cell_center_lon, distance_km, self.weather_cell_heading)
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
                    vertex_lat, vertex_lon = RLtools.functions.get_point_at_distance(weather_cell_latitude_centre, weather_cell_longitude_centre, radius_jittered_km, ang_deg)
                    weather_cell_poly.append((vertex_lat, vertex_lon))
                # close polygon
                weather_cell_poly.append(weather_cell_poly[0])

                self.weather_cell_radius = max(weather_cell_radius)

                # Re-draw polygon (delete + poly keeps syntax simple)
                stack.process(f"DELETE {shape_name}")
                flat = ", ".join([f"{lat:.6f}, {lon:.6f}" for (lat, lon) in weather_cell_poly])
                stack.process(f"POLY {shape_name}, {flat}")
                stack.process(f"COLOR {shape_name}, BLUE")
                # stack.process(f"CIRCLE {shape_name}_bounding_circle, {self.weather_cell_center_lat}, {self.weather_cell_center_lon}, {self.weather_cell_radius/RLtools.constants.NM2KM}")
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
                    vertex_lat, vertex_lon = RLtools.functions.get_point_at_distance(weather_cell_latitude_centre, weather_cell_longitude_centre, radius_jittered_km, ang_deg)
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
                # stack.process(f"CIRCLE {shape_name}_bounding_circle, {self.weather_cell_center_lat}, {self.weather_cell_center_lon}, {self.weather_cell_radius/RLtools.constants.NM2KM}")
                # stack.process(f"COLOR {shape_name}_bounding_circle, YELLOW")

    def _generate_random_restricted_areas(self, NUM_OBSTACLES):
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

        self._generate_coordinates_centre_obstacles(num_obstacles = RLtools.constants.NUM_OBSTACLES)

        obstacle_dict = {}  # Initialize the dictionary to store obstacles for overlap checking

        for i in range(RLtools.constants.NUM_OBSTACLES):

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

        for i in range(NUM_OBSTACLES):
            overlap_list = []  # List to store overlap information
            # Check for overlaps with existing obstacles
            for j in range(NUM_OBSTACLES):
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
                        interpolated_points = RLtools.functions.interpolate_along_obstacle_vertices(obstacle_vertices[j][k], obstacle_vertices[j][0])
                    else:
                        interpolated_points = RLtools.functions.interpolate_along_obstacle_vertices(obstacle_vertices[j][k], obstacle_vertices[j][k+1])
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
        p = [RLtools.functions.random_point_on_circle(R) for _ in range(3)] # 3 random points to start building the polygon
        p = RLtools.functions.sort_points_clockwise(p)
        p_area = RLtools.functions.polygon_area(p)
        
        while p_area < OBSTACLE_AREA_RANGE[0]:
            p.append(RLtools.functions.random_point_on_circle(R))
            p = RLtools.functions.sort_points_clockwise(p)
            p_area = RLtools.functions.polygon_area(p)
        
        p = [RLtools.functions.nm_to_latlong(centre, point) for point in p] # Convert to lat/long coordinateS
        
        return p_area, p, R

    def _generate_coordinates_centre_obstacles(self, num_obstacles = RLtools.constants.NUM_OBSTACLES):
        self.obstacle_centre_lat = []
        self.obstacle_centre_lon = []
        OBSTACLE_DISTANCE_MIN = 20 # KM
        OBSTACLE_DISTANCE_MAX = 500 # KM

        for i in range(num_obstacles):
            obstacle_dis_from_reference = np.random.randint(OBSTACLE_DISTANCE_MIN, OBSTACLE_DISTANCE_MAX)
            obstacle_hdg_from_reference = np.random.randint(0, 360)
            # ac_idx = bs.traf.id2idx(acid)

            obstacle_centre_lat, obstacle_centre_lon = RLtools.functions.get_point_at_distance(latitude_bounds[0] + (latitude_bounds[1]-latitude_bounds[0])/2, longitude_bounds[0] + (longitude_bounds[1]-longitude_bounds[0])/2, obstacle_dis_from_reference, obstacle_hdg_from_reference)    
            self.obstacle_centre_lat.append(obstacle_centre_lat)
            self.obstacle_centre_lon.append(obstacle_centre_lon)



        