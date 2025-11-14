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

# N_AC = 20  # Number of aircraft in the randomised scenario
N_SCN = 10 # Number of testing iterations

sector_name = 'LISBON_FIR'
latitude_bounds = (31.4, 43.0)
longitude_bounds = (-18.3, -6.1)

# smaller bounds for testing stage
latitude_bounds = (33.0, 36.0)
longitude_bounds = (-18.0, -12.0)

def init_plugin():
    deploy_RL = DeployRL()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'DeployRL2', #'Deploy_RL', #
        # The type of this plugin.
        'plugin_type':     'sim',
        }
    stackfunctions = {
        'DEPLOY_RL': [
            'DEPLOY_RL [ENV_NAME] [ALGORITHM] [N_AC] [N_OBSTACLES]',
            '[txt] [txt] [int] [int]',
            deploy_RL.initialize_RL,
            'Initialises the RL deployment plugin with the specified environment and algorithm',
        ]}
    return config, stackfunctions

class DeployRL(core.Entity):  
    def __init__(self):
        super().__init__()
        self.scn_idx = 0


        import debug
        debug.light_blue(f'initialised deploy RL at {self.scn_idx}')
        self.first_initialization = True

    def reset(self):
        import debug
        debug.light_green(f'reset deploy RL at {self.scn_idx}')
        
        # for shape_name, _ in tools.areafilter.basic_shapes.items():
        #     print(f'Existing obstacle after reset: {shape_name}')


        # stack.process('pcall ai4realnet_deploy_RL/sector.scn;initialize_scenario;DTMULT 5000')


    def initialize_RL(self, env_name: str, algorithm: str, number_aircraft: int, number_obstacles: int):
        """
        Initialize a new random scenario with the specified number of aircraft and obstacles.

        Args:
            number_aircraft (int): Number of aircraft to generate in the scenario.
            number_obstacles (int): Number of random restricted areas to create.

        Example:
            INITIALIZE_SCENARIO 20 5
        """
        if self.first_initialization:
            self.env_name = env_name
            self.algorithm = algorithm
            self.number_obstacles = number_obstacles
            self.number_aircraft = number_aircraft
            self.model = SAC.load(f"bluesky/plugins/ai4realnet_deploy_RL_tools/models/{env_name}/{env_name}_{algorithm}/model", env=None)
            
            # logging
            self.log_buffer = []   # temporary storage
            self.csv_file = (f"output/ai4realnet_deploy_RL_{env_name}_log.csv")
            self.first_initialization = False
        
        stack.stack(f'initialize_scenario {number_aircraft} {number_obstacles}')
        stack.stack(f'perturbation weather on')
        stack.stack(f'perturbation volcanic on')
        stack.stack(f'SAVEIC test_{self.scn_idx}')
        stack.stack(f'DTMULT 5000')
        debug.light_blue(f'called save ic for {self.scn_idx}')
        self.initialise_observation_flag = True


    @core.timed_function(name='StaticObstacleSectorCR', dt=RLtools.constants.ACTION_FREQUENCY)
    def update(self):
        if self.first_initialization:
            return
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
            # for some reason, the weather and volcanic cell on the screen is not deleted when reset is issued 
            stack.process(f"DELETE WEATHER_CELL")
            stack.process(f"DELETE VOLCANIC_CELL")
            # bs.tools.areafilter.deleteArea('WEATHER_CELL')
            # bs.sim.step()
            stack.process('RESET')
            # stack.process('RESET')
            stack.process(f'deploy_RL {self.env_name} {self.algorithm} {self.number_aircraft} {self.number_obstacles}')

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
            self.obstacle_centre_lat = []
            self.obstacle_centre_lon = []
            self.obstacle_radius = []
            self.number_obstacles = 0
            for shape_name, shape in tools.areafilter.basic_shapes.items():
                if shape_name != 'LISBON_FIR' and shape_name != 'WEATHER_CELL' and shape_name != 'VOLCANIC_CELL':
                    self.number_obstacles += 1
                    # print(f'Processing obstacle: {shape_name}')
                    coordinates = shape.coordinates
                    coordinates = list(zip(coordinates[::2], coordinates[1::2]))

                    (lat_c, lon_c), radius = RLtools.functions.bounding_circle_geodesic(coordinates)
                    self.obstacle_centre_lat.append(lat_c)
                    self.obstacle_centre_lon.append(lon_c)
                    self.obstacle_radius.append(radius)
            
            self.max_obstacle_radius = max(self.obstacle_radius)

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

        for obs_idx in range(self.number_obstacles):
            
            obs_centre_qdr, obs_centre_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.obstacle_centre_lat[obs_idx], self.obstacle_centre_lon[obs_idx])
            obs_centre_dis = obs_centre_dis * RLtools.constants.NM2KM #KM        
            bearing = bs.traf.hdg[ac_idx] - obs_centre_qdr
            
            bearing = RLtools.functions.bound_angle_positive_negative_180(bearing)

            obstacle_centre_distance.append(obs_centre_dis)
            obstacle_centre_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            obstacle_centre_sin_bearing.append(np.sin(np.deg2rad(bearing)))

        # weather disturbances
        if bs.tools.areafilter.basic_shapes.get('WEATHER_CELL') is not None:
            coordinates = bs.tools.areafilter.basic_shapes['WEATHER_CELL'].coordinates
            coordinates = list(zip(coordinates[::2], coordinates[1::2]))
            (weather_cell_center_lat, weather_cell_center_lon), weather_cell_radius = RLtools.functions.bounding_circle_geodesic(coordinates)
            weather_cell_centre_qdr, weather_cell_centre_distance = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], weather_cell_center_lat, weather_cell_center_lon)
            weather_cell_centre_distance = weather_cell_centre_distance * RLtools.constants.NM2KM #KM            
            bearing = RLtools.functions.bound_angle_positive_negative_180(bs.traf.hdg[ac_idx] - weather_cell_centre_qdr)

            weather_cell_centre_cos_bearing = np.cos(np.deg2rad(bearing))
            weather_cell_centre_sin_bearing = np.sin(np.deg2rad(bearing))

        # volcanic disturbances
        if bs.tools.areafilter.basic_shapes.get('VOLCANIC_CELL') is not None:
            coordinates = bs.tools.areafilter.basic_shapes['VOLCANIC_CELL'].coordinates
            coordinates = list(zip(coordinates[::2], coordinates[1::2]))
            (volcanic_cell_center_lat, volcanic_cell_center_lon), volcanic_cell_radius = RLtools.functions.bounding_circle_geodesic(coordinates)
            volcanic_cell_centre_qdr, volcanic_cell_centre_distance = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], volcanic_cell_center_lat, volcanic_cell_center_lon)
            volcanic_cell_centre_distance = volcanic_cell_centre_distance * RLtools.constants.NM2KM #KM            
            bearing = RLtools.functions.bound_angle_positive_negative_180(bs.traf.hdg[ac_idx] - volcanic_cell_centre_qdr)

            volcanic_cell_centre_cos_bearing = np.cos(np.deg2rad(bearing))
            volcanic_cell_centre_sin_bearing = np.sin(np.deg2rad(bearing))


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
        # RLtools.constants.NUM_OBSTACLES depends on how many obstacles the AI has been trained with
        if bs.tools.areafilter.basic_shapes.get('WEATHER_CELL') is not None:
            obstacle_centre_distance.append(weather_cell_centre_distance)
            obstacle_centre_cos_bearing.append(weather_cell_centre_cos_bearing)
            obstacle_centre_sin_bearing.append(weather_cell_centre_sin_bearing)
            obstacle_radius.append(weather_cell_radius)
        
        if bs.tools.areafilter.basic_shapes.get('VOLCANIC_CELL') is not None:
            obstacle_centre_distance.append(volcanic_cell_centre_distance)
            obstacle_centre_cos_bearing.append(volcanic_cell_centre_cos_bearing)
            obstacle_centre_sin_bearing.append(volcanic_cell_centre_sin_bearing)
            obstacle_radius.append(volcanic_cell_radius)


        obstacle_centre_distance = np.array(obstacle_centre_distance)
        obstacle_centre_cos_bearing = np.array(obstacle_centre_cos_bearing)
        obstacle_centre_sin_bearing = np.array(obstacle_centre_sin_bearing)
        obstacle_radius = np.array(obstacle_radius)
        
        idx_sorted = np.argsort(obstacle_centre_distance)[:RLtools.constants.NUM_OBSTACLES]

        obstacle_centre_distance = obstacle_centre_distance[idx_sorted]
        obstacle_centre_cos_bearing = obstacle_centre_cos_bearing[idx_sorted]
        obstacle_centre_sin_bearing = obstacle_centre_sin_bearing[idx_sorted]
        obstacle_radius = obstacle_radius[idx_sorted]

        num_missing = RLtools.constants.NUM_OBSTACLES - obstacle_centre_distance.size
        if num_missing > 0:
            obstacle_centre_distance = np.pad(obstacle_centre_distance, (0, num_missing), 'constant', constant_values=0.0)
            obstacle_centre_cos_bearing = np.pad(obstacle_centre_cos_bearing, (0, num_missing), 'constant', constant_values=0.0)
            obstacle_centre_sin_bearing = np.pad(obstacle_centre_sin_bearing, (0, num_missing), 'constant', constant_values=0.0)
            obstacle_radius = np.pad(obstacle_radius, (0, num_missing), 'constant', constant_values=0.0)
        
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

