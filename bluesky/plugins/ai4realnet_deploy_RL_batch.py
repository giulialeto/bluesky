"""
    AI4REALNET -  Deliverable 1.4 BlueSky plugin for deploying RL-based model in batch scenarios
    Authors: Giulia Leto
    Date: Nov 2025
"""
from bluesky import core, stack, traf, tools, settings 
from bluesky.stack.simstack import readscn
import bluesky as bs
import bluesky.plugins.ai4realnet_deploy_RL_tools_batch as RLtools
from stable_baselines3 import SAC, TD3, PPO, DDPG
import numpy as np
import pandas as pd
import os, datetime
from pathlib import Path

# Global variables
PLUGIN_DIR = Path(__file__).resolve().parent
MODELS_DIR = PLUGIN_DIR / "ai4realnet_deploy_RL_models"

# print(f"Current directory is {os.getcwd()}.")

sector_name = 'LISBON_FIR'
save_dir = 'ai4realnet_deploy_RL_batch/generated_scenarios'

# Plugin initialization function
def init_plugin():
    deploy_RL = DeployRL()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'DeployRL_batch',
        # The type of this plugin.
        'plugin_type':     'sim',
        }

    return config

class DeployRL(core.Entity):  

    def __init__(self):
        super().__init__()

        self.start_next = True
        self.repeats = 0
        self.total = 0
        self.scentime = []
        self.scencmd = []
        self.start_updates = False
        self.max_sim_time = 3600  # seconds
        os.makedirs(f'scenario/{save_dir}', exist_ok=True)

    def reset(self):
        pass

    @stack.command
    def detached_batch(self, fname: str):
        """
        Load a batch scenario file, extract REPEATS line,
        and store the scenario commands locally.
        """
        # use the server batch file load function to populate self.scen_batch
        scentime = []
        scencmd = []
        for (cmdtime, cmd) in readscn(fname):
                scentime.append(cmdtime)
                scencmd.append(cmd)
                # print(f'[DeployRL] Read command: {cmd} at time {cmdtime}')

        idx = next(
            (i for i, cmd in enumerate(scencmd)
            if cmd.strip().lower().startswith('repeats')),
            None
        )

        if idx is None:
            # No repeats line found
            print(f"[DeployRL] No 'repeats' line found in scenario '{fname}'.")
            return

        cmd = scencmd.pop(idx)
        scentime.pop(idx)
        stack.process(cmd)

        # Remove plugin lines from scenario
        new_scentime = []
        new_scencmd  = []

        for t, c in zip(scentime, scencmd):
            # Skip plugin lines
            if c.strip().lower().startswith("plugin"):
                stack.process(c)  # process the plugin command
                continue
            new_scentime.append(t)
            new_scencmd.append(c)

        
        self.start_next = True


        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, cmd in enumerate(new_scencmd):
            line = cmd.strip().lower()

            # Parse: initialize_scenario NUM_AIRCRAFT, NUM_OBSTACLES
            if line.startswith("initialize_scenario"):
                # Remove command name
                args = line.replace("initialize_scenario", "").strip()
                parts = [p.strip() for p in args.split(",")]

                if len(parts) == 2:
                    self.number_aircraft = int(parts[0])
                    self.number_obstacles = int(parts[1])
                else:
                    print("[DeployRL] ERROR: initialize_scenario must have exactly 2 arguments.")

            # Parse: deploy_RL ENV_NAME ALGORITHM
            elif line.startswith("deploy_rl"):
                tokens = line.split()
                # tokens = ["deploy_rl", "staticobstacleenv-v0", "sac"]
                if len(tokens) == 3:
                    self.env_name = tokens[1]
                    self.algorithm = tokens[2]
                else:
                    print("[DeployRL] ERROR: deploy_RL must be: deploy_RL <env_name> <algorithm>")
                index_to_remove = i
                
        new_scencmd.pop(index_to_remove)  # remove deploy_RL line after processing
        new_scentime.pop(index_to_remove) # remove corresponding time entry
        
        self.scentime = new_scentime
        self.scencmd  = new_scencmd

        # Load the RL model 
        if self.algorithm.lower() in ('sac'):
            self.model = SAC.load(f"{MODELS_DIR}/{self.env_name}/{self.env_name}_{self.algorithm}/model", env=None)
        elif self.algorithm.lower() in ('td3'):
            self.model = TD3.load(f"{MODELS_DIR}/{self.env_name}/{self.env_name}_{self.algorithm}/model", env=None)
        elif self.algorithm.lower() in ('ppo'):
            self.model = PPO.load(f"{MODELS_DIR}/{self.env_name}/{self.env_name}_{self.algorithm}/model", env=None)
        elif self.algorithm.lower() in ('ddpg'):
            self.model = DDPG.load(f"{MODELS_DIR}/{self.env_name}/{self.env_name}_{self.algorithm}/model", env=None)

        # logging
        self.log_buffer = []   # temporary storage
        current_working_dir = os.getcwd()
       
        # print(f'Current working dir is: {current_working_dir}')

        if  Path(os.getcwd()).name == 'ai4realnet-orchestrators':
            
            marker_path = Path(f'{current_working_dir}/ai4realnet_orchestrators/atm/current_scenfile.txt')

            scenfile = marker_path.read_text().strip()
            stack.stack(f'ECHO Loaded scenario file: {scenfile}')
            scenario_id = Path(scenfile).stem
            stack.stack(f'ECHO Scenario base name: {scenario_id}')

            self.csv_file = (f"{current_working_dir}/ai4realnet_orchestrators/atm/output/{scenario_id}_log.csv")
        else:
            self.csv_file = (f"output/ai4realnet_deploy_RL_batch_{self.env_name}_{self.algorithm}_log.csv")
        
        # print(f'self.env_name: {self.env_name}, self.algorithm: {self.algorithm}, number_aircraft: {self.number_aircraft}, number_obstacles: {self.number_obstacles}')
        # print(f'self.csv_file: {self.csv_file}')

        if self.env_name in ('staticobstaclesectorcrenv-v0'):
            # print(f'initialise_RL for {self.env_name} at {self.scn_idx}')
            # Model parameters
            self.NUM_OBSTACLES = 5 #np.random.randint(1,5)
            self.NUM_INTRUDERS = 5
            self.AC_SPD = 150 # m/s
            self.D_HEADING = 45 #degrees
            self.D_SPEED = 20/3 # m/s

            self.ACTION_FREQUENCY = 10

            self.TOTAL_OBSERVATION_POINTS = 50 # Number of points to be observed along the sector polygon edges
            self.DISTANCE_MARGIN = 5 # km

        if self.env_name in ('staticobstaclesectorenv-v0'):
            # print(f'initialise_RL for {self.env_name} at {self.scn_idx}')
            # Model parameters
            self.NUM_OBSTACLES = 10 #np.random.randint(1,5)
            self.AC_SPD = 150 # m/s
            self.D_HEADING = 45 #degrees
            self.D_SPEED = 20/3 # m/s

            self.ACTION_FREQUENCY = 10

            self.TOTAL_OBSERVATION_POINTS = 50 # Number of points to be observed along the sector polygon edges
            self.DISTANCE_MARGIN = 5 # km
        if self.env_name in ('staticobstaclecrenv-v0'):
            # print(f'initialise_RL for {self.env_name} at {self.scn_idx}')
            # Model parameters
            self.NUM_OBSTACLES = 5 #np.random.randint(1,5)
            self.NUM_INTRUDERS = 5
            self.AC_SPD = 150 # m/s
            self.D_HEADING = 45 #degrees
            self.D_SPEED = 20/3 # m/s

            self.ACTION_FREQUENCY = 10
            self.DISTANCE_MARGIN = 5 # km

        if self.env_name in ('staticobstacleenv-v0'):
            # print(f'initialise_RL for {self.env_name} at {self.scn_idx}')
            # Model parameters
            self.NUM_OBSTACLES = 10 #np.random.randint(1,5)
            self.AC_SPD = 150 # m/s
            self.D_HEADING = 45 #degrees
            self.D_SPEED = 20/3 # m/s

            self.ACTION_FREQUENCY = 10
            self.DISTANCE_MARGIN = 5 # km

        self.start_updates = True
        stack.process(f'OP')
        stack.process(f'DTMULT 5000')


    @stack.command
    def end_scen(self):
        """
        Mark the current scenario as finished so the next one can start.
        """
        self.start_next = True

    @stack.command
    def repeats(self, repetitions: int):
        """
        Set how many times the detached batch scenario should be run.
        """
        self.total = self.repeats = repetitions
        
    @core.timed_function(name='update', dt=RLtools.constants.ACTION_FREQUENCY)
    def update(self):
        if self.start_updates == False:
            return
        
        if self.start_next and self.repeats >= 0:
            self.repeats -= 1
            self.scn_idx = self.total - self.repeats
            bs.sim.start_batch_scenario(f'batch_{self.scn_idx}', list(self.scentime), list(self.scencmd))
            stack.stack(f'SAVEIC {save_dir}/{self.timestamp}_{self.env_name}_{self.algorithm}_{self.number_aircraft}_{self.number_obstacles}_{self.scn_idx}')

            self.start_next = False
            self.initialise_observation_flag = True
            self.initial_observation_done = False
            stack.stack(f'ECHO Starting scenario batch_{self.scn_idx}, repeats left: {self.repeats}')
            print(f'Starting scenario batch_{self.scn_idx}, repeats left: {self.repeats}')

        for ac_idx, id in enumerate(traf.id):
            _, dest_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1])
            if dest_dis * RLtools.constants.NM2KM < self.DISTANCE_MARGIN:
                # stack.stack(f"ECHO Aircraft {id} has reached the destination waypoint.")
                # print(f"Aircraft {id} has reached the destination waypoint.")
                stack.process(f"DELETE {id}")
                continue  # skip action if the aircraft has reached its destination
            obs = self._get_obs(ac_idx)
            action, _ = self.model.predict(obs, deterministic=True)
            self._set_action(action, ac_idx)

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


        if (traf.id == [] or simt > self.max_sim_time) and (self.repeats > 0) and (self.initial_observation_done):
            print(f'Ending scenario batch_{self.scn_idx}')
            stack.process('END_SCEN')


        if (traf.id == [] or simt > self.max_sim_time) and (self.repeats == 0) and (self.initial_observation_done):
            print(f'Ending scenario batch_{self.scn_idx}')
            stack.process('QUIT')

    def _get_obs(self, ac_idx):
        """
        Observation is the normalized. Normalisation logic should be studied further
        """
        if self.initialise_observation_flag:
            waypoint_distances = []

            # Give aircraft initial heading
            for ac_idx, _ in enumerate(traf.id):
                _, initial_wpt_dist = tools.geo.kwikqdrdist(traf.lat[ac_idx], traf.lon[ac_idx], bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1])
                waypoint_distances.append(initial_wpt_dist)

            # Scaling factor for the distances in the observation vector
            self.waypoint_distance_max = max(waypoint_distances)

            if self.env_name in ('staticobstaclesectorcrenv-v0', 'staticobstaclesectorenv-v0'):
                sector = tools.areafilter.basic_shapes[sector_name]
                coordinates = sector.coordinates
                latitudes = coordinates[::2]
                longitudes = coordinates[1::2]
                self.sector_points = RLtools.functions.resample_closed_border(latitudes, longitudes, self.TOTAL_OBSERVATION_POINTS)

            # # Display in BlueSky the sector points contained in the observation 
            # print(f'sector points: {self.sector_points}, shape: {np.array(self.sector_points).shape}, type: {type(self.sector_points)}')
            # coordinates = ", ".join([f"{lat} {lon}" for lat, lon in self.sector_points])
            # stack.stack(f"POLY SECTOR_OBSERVATION, {coordinates}")
            self.obstacle_centre_lat = []
            self.obstacle_centre_lon = []
            self.obstacle_radius = []
            self.number_obstacles = 0
            for shape_name, shape in tools.areafilter.basic_shapes.items():
                if shape_name != sector_name and shape_name != 'WEATHER_CELL' and shape_name != 'VOLCANIC_CELL':
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
            self.initial_observation_done = True

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
            # print(f'Processing WEATHER_CELL obstacle for observation')
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
            # print(f'Processing VOLCANIC_CELL obstacle for observation')
            coordinates = bs.tools.areafilter.basic_shapes['VOLCANIC_CELL'].coordinates
            coordinates = list(zip(coordinates[::2], coordinates[1::2]))
            (volcanic_cell_center_lat, volcanic_cell_center_lon), volcanic_cell_radius = RLtools.functions.bounding_circle_geodesic(coordinates)
            volcanic_cell_centre_qdr, volcanic_cell_centre_distance = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], volcanic_cell_center_lat, volcanic_cell_center_lon)
            volcanic_cell_centre_distance = volcanic_cell_centre_distance * RLtools.constants.NM2KM #KM            
            bearing = RLtools.functions.bound_angle_positive_negative_180(bs.traf.hdg[ac_idx] - volcanic_cell_centre_qdr)

            volcanic_cell_centre_cos_bearing = np.cos(np.deg2rad(bearing))
            volcanic_cell_centre_sin_bearing = np.sin(np.deg2rad(bearing))
        
        obstacle_radius = self.obstacle_radius.copy()

        # Find the RLtools.constants.NUM_OBSTACLES closest obstacles to the ownship (restricted areas or weather cells)
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
        
        # print(f'self.NUM_OBS: {self.NUM_OBSTACLES}')
        # select the closest RLtools.constants.NUM_OBSTACLES obstacles
        idx_sorted = np.argsort(obstacle_centre_distance)[:self.NUM_OBSTACLES]
        
        obstacle_centre_distance = obstacle_centre_distance[idx_sorted]
        obstacle_centre_cos_bearing = obstacle_centre_cos_bearing[idx_sorted]
        obstacle_centre_sin_bearing = obstacle_centre_sin_bearing[idx_sorted]
        obstacle_radius = obstacle_radius[idx_sorted]

        # padding if less than RLtools.constants.NUM_OBSTACLES are present
        num_missing = self.NUM_OBSTACLES - obstacle_centre_distance.size
        if num_missing > 0:
            obstacle_centre_distance = np.pad(obstacle_centre_distance, (0, num_missing), 'constant', constant_values=0.0)
            obstacle_centre_cos_bearing = np.pad(obstacle_centre_cos_bearing, (0, num_missing), 'constant', constant_values=0.0)
            obstacle_centre_sin_bearing = np.pad(obstacle_centre_sin_bearing, (0, num_missing), 'constant', constant_values=0.0)
            obstacle_radius = np.pad(obstacle_radius, (0, num_missing), 'constant', constant_values=0.0)


        if self.env_name in ('staticobstaclesectorcrenv-v0', 'staticobstaclecrenv-v0'):
            # intruder observation
            intruders_lat = np.delete(bs.traf.lat, ac_idx)
            intruders_lon = np.delete(bs.traf.lon, ac_idx)
            
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], intruders_lat, intruders_lon)

            # index of the closes N_INTRUDERS intruders for arrays of intruders_lat, intruders_lon, int_qdr, int_dis, intruders_speed, intruders_heading: the ownship is excluded a priori in these
            closest_intruders_idx = np.argsort(int_dis)[:self.NUM_INTRUDERS]

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
            if len(closest_intruders_idx) < self.NUM_INTRUDERS:
                num_missing_intruders = self.NUM_INTRUDERS - len(closest_intruders_idx)
                intruder_distance = np.pad(intruder_distance, (0, num_missing_intruders), 'constant', constant_values=(self.waypoint_distance_max,))
                intruder_cos_bearing =  np.pad(intruder_cos_bearing, (0, num_missing_intruders), 'constant', constant_values=(0,))
                intruder_sin_bearing =  np.pad(intruder_sin_bearing, (0, num_missing_intruders), 'constant', constant_values=(0,))
                intruder_x_difference_speed =  np.pad(intruder_x_difference_speed, (0, num_missing_intruders), 'constant', constant_values=(0,))
                intruder_y_difference_speed =  np.pad(intruder_y_difference_speed, (0, num_missing_intruders), 'constant', constant_values=(0,))

        if self.env_name in ('staticobstaclesectorcrenv-v0', 'staticobstaclesectorenv-v0'):
            # sector polygon edges observation
            sector_points_distance = []
            sector_points_cos_drift = []
            sector_points_sin_drift = []

            # Calculate distance and bearing from the ownship to each of the sector points
            for point_index in range(len(self.sector_points)):
                sector_points_qdr, sector_points_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.sector_points[point_index,0],self.sector_points[point_index,1])
                # print(f'point_index: {point_index}, sector_points_dis: {sector_points_dis}, sector_points_qdr: {sector_points_qdr}')
                sector_points_distance.append(sector_points_dis * RLtools.constants.NM2KM)

                drift = bs.traf.hdg[ac_idx] - sector_points_qdr

                drift = RLtools.functions.bound_angle_positive_negative_180(drift)

                sector_points_cos_drift.append(np.cos(np.deg2rad(drift)))
                sector_points_sin_drift.append(np.sin(np.deg2rad(drift)))
        # print(f'self.obstacle_radius length is: {len(self.obstacle_radius)}')
        # if len(closest_intruders_idx) < RLtools.constants.NUM_INTRUDERS:
        #     print(f'observation: {observation}')
        if self.env_name in ('staticobstaclesectorcrenv-v0'):
            observation = {
                    "intruder_distance": np.array(intruder_distance).reshape(-1)/self.waypoint_distance_max,
                    "intruder_cos_difference_pos": np.array(intruder_cos_bearing).reshape(-1),
                    "intruder_sin_difference_pos": np.array(intruder_sin_bearing).reshape(-1),
                    "intruder_x_difference_speed": np.array(intruder_x_difference_speed).reshape(-1)/self.AC_SPD,
                    "intruder_y_difference_speed": np.array(intruder_y_difference_speed).reshape(-1)/self.AC_SPD,
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
        if self.env_name in ('staticobstaclesectorenv-v0'):
            observation = {
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
        if self.env_name in ('staticobstaclecrenv-v0'):
            observation = {
                    "intruder_distance": np.array(intruder_distance).reshape(-1)/self.waypoint_distance_max,
                    "intruder_cos_difference_pos": np.array(intruder_cos_bearing).reshape(-1),
                    "intruder_sin_difference_pos": np.array(intruder_sin_bearing).reshape(-1),
                    "intruder_x_difference_speed": np.array(intruder_x_difference_speed).reshape(-1)/self.AC_SPD,
                    "intruder_y_difference_speed": np.array(intruder_y_difference_speed).reshape(-1)/self.AC_SPD,
                    "destination_waypoint_distance": np.array(destination_waypoint_distance).reshape(-1)/self.waypoint_distance_max,
                    "destination_waypoint_cos_drift": np.array(destination_waypoint_cos_drift).reshape(-1),
                    "destination_waypoint_sin_drift": np.array(destination_waypoint_sin_drift).reshape(-1),
                    # observations on obstacles
                    "restricted_area_radius": np.array(obstacle_radius).reshape(-1)/self.max_obstacle_radius,
                    "restricted_area_distance": np.array(obstacle_centre_distance).reshape(-1)/self.waypoint_distance_max,
                    "cos_difference_restricted_area_pos": np.array(obstacle_centre_cos_bearing).reshape(-1),
                    "sin_difference_restricted_area_pos": np.array(obstacle_centre_sin_bearing).reshape(-1),
                }
        if self.env_name in ('staticobstacleenv-v0'):
            observation = {
                    "destination_waypoint_distance": np.array(destination_waypoint_distance).reshape(-1)/self.waypoint_distance_max,
                    "destination_waypoint_cos_drift": np.array(destination_waypoint_cos_drift).reshape(-1),
                    "destination_waypoint_sin_drift": np.array(destination_waypoint_sin_drift).reshape(-1),
                    # observations on obstacles
                    "restricted_area_radius": np.array(obstacle_radius).reshape(-1)/self.max_obstacle_radius,
                    "restricted_area_distance": np.array(obstacle_centre_distance).reshape(-1)/self.waypoint_distance_max,
                    "cos_difference_restricted_area_pos": np.array(obstacle_centre_cos_bearing).reshape(-1),
                    "sin_difference_restricted_area_pos": np.array(obstacle_centre_sin_bearing).reshape(-1),
                }
        # print(
        #     f'observation for aircraft {bs.traf.id[ac_idx]}: '
        #     f'{ {k: v for k, v in observation.items()} }'
        # )

        # print(
        #     f'observation shapes for aircraft {bs.traf.id[ac_idx]}: '
        #     f'{ {k: v.shape for k, v in observation.items()} }'
        # )        
        return observation

    def _set_action(self, action, ac_idx):
        """
        Control each aircraft separately
        """
        # print(f'New action')
        dv = action[1] * self.D_SPEED
        dh = action[0] * self.D_HEADING

        id = traf.id[ac_idx]
        heading_new = RLtools.functions.bound_angle_positive_negative_180(traf.hdg[ac_idx] + dh)
        speed_new = (traf.cas[ac_idx] + dv) * RLtools.constants.MpS2Kt
        # stack.stack(f"ECHO Aircraft {id} - New heading: {heading_new} deg, New speed: {speed_new/RLtools.constants.MpS2Kt} m/s")
        stack.stack(f"HDG {id} {heading_new}")
        stack.stack(f"SPD {id} {speed_new}")
        # print(f'Action for aircraft {id} - traf.hdg: {traf.hdg[ac_idx]} -> {heading_new} with dh {dh}, traf.cas: {traf.cas[ac_idx]} m/s -> {speed_new/RLtools.constants.MpS2Kt} with dv {dv} m/s')