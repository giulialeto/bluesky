"""
    AI4REALNET -  Deliverable 1.4 BlueSky plugin for deploying RL-based model in batch scenarios
    Authors: Giulia Leto
    Date: Nov 2025
"""
from multiprocessing.util import debug
from os import path

from bluesky import core, stack, traf, tools, settings 
from bluesky.stack.simstack import readscn
import bluesky as bs
import bluesky.plugins.ai4realnet_deploy_baseline_tools_batch as Baselinetools
from stable_baselines3 import SAC, TD3, PPO, DDPG
import numpy as np
import pandas as pd
import os, datetime
from pathlib import Path
import networkx as nx
import bluesky.plugins.ai4realnet_deploy_baseline_tools_batch.deterministic_path_planning as path_plan
from bluesky.tools.aero import kts
import bluesky.plugins.CRTools as CRT ### IF comparing with my model, the update step should be the same


# Global variables
PLUGIN_DIR = Path(__file__).resolve().parent
MODELS_DIR = PLUGIN_DIR / "ai4realnet_deploy_RL_models"

# print(f"Current directory is {os.getcwd()}.")

sector_name = 'LISBON_FIR'
save_dir = 'ai4realnet_deploy_baseline_batch/generated_scenarios'

# Plugin initialization function
def init_plugin():
    deploy_Baseline = DeployBaseline()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'DeployBaseline',
        # The type of this plugin.
        'plugin_type':     'sim',
        }

    return config

class DeployBaseline(core.Entity):  

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
        # read scenario file and extract commands and their times
        scentime = []
        scencmd = []
        for (cmdtime, cmd) in readscn(fname):
                scentime.append(cmdtime)
                scencmd.append(cmd)
                # print(f'[DeployBaseline] Read command: {cmd} at time {cmdtime}')

        # Find the index of the "repeats" line
        idx = next(
            (i for i, cmd in enumerate(scencmd)
            if cmd.strip().lower().startswith('repeats')),
            None
        )

        if idx is None:
            # No repeats line found
            print(f"[DeployBaseline] No 'repeats' line found in scenario '{fname}'.")
            return

        # Extract the number of repetitions from the "repeats" line
        cmd = scencmd.pop(idx)
        scentime.pop(idx)
        stack.process(cmd)

        # Remove plugin lines from scenario
        new_scentime = []
        new_scencmd  = []

        for plugin_time, plugin_cmd in zip(scentime, scencmd):
            # Skip plugin lines
            if plugin_cmd.strip().lower().startswith("plugin"):

                stack.process(plugin_cmd)  # process the plugin command
                continue
            new_scentime.append(plugin_time)
            new_scencmd.append(plugin_cmd)

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
                    print("[DeployBaseline] ERROR: initialize_scenario must have exactly 2 arguments.")

            # Parse: deploy_RL ENV_NAME ALGORITHM
            elif line.startswith("deploy_baseline"):
                tokens = line.split()
                # tokens = ["deploy_baseline", "staticobstacleenv-v0", "sac"]
                if len(tokens) == 3:
                    self.env_name = tokens[1]
                    self.algorithm = tokens[2]
                else:
                    print("[DeployBaseline] ERROR: deploy_baseline must be: deploy_baseline <env_name> <algorithm>")
 
                index_to_remove = i

        new_scencmd.pop(index_to_remove)  # remove deploy_Baseline line after processing
        new_scentime.pop(index_to_remove) # remove corresponding time entry
        
        self.scentime = new_scentime
        self.scencmd  = new_scencmd

        # Load the correct algorithn based on the scenario configuations
        # if self.algorithm.lower() in ('sac'):
        #     self.model = SAC.load(f"{MODELS_DIR}/{self.env_name}/{self.env_name}_{self.algorithm}/model", env=None)
        # elif self.algorithm.lower() in ('td3'):
        #     self.model = TD3.load(f"{MODELS_DIR}/{self.env_name}/{self.env_name}_{self.algorithm}/model", env=None)
        # elif self.algorithm.lower() in ('ppo'):
        #     self.model = PPO.load(f"{MODELS_DIR}/{self.env_name}/{self.env_name}_{self.algorithm}/model", env=None)
        # elif self.algorithm.lower() in ('ddpg'):
        #     self.model = DDPG.load(f"{MODELS_DIR}/{self.env_name}/{self.env_name}_{self.algorithm}/model", env=None)

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
            self.csv_file = (f"output/ai4realnet_deploy_baseline_batch_{self.env_name}_{self.algorithm}_log.csv")
        
        self.DISTANCE_MARGIN = 5 # km
        # print(f'self.env_name: {self.env_name}, self.algorithm: {self.algorithm}, number_aircraft: {self.number_aircraft}, number_obstacles: {self.number_obstacles}')
        # print(f'self.csv_file: {self.csv_file}')

        # if self.env_name in ('staticobstaclesectorcrenv-v0'):
        #     # print(f'initialise_RL for {self.env_name} at {self.scn_idx}')
        #     # Model parameters
        #     self.NUM_OBSTACLES = 5 #np.random.randint(1,5)
        #     self.NUM_INTRUDERS = 5
        #     self.AC_SPD = 150 # m/s
        #     self.D_HEADING = 45 #degrees
        #     self.D_SPEED = 20/3 # m/s

        #     self.ACTION_FREQUENCY = 10

        #     self.TOTAL_OBSERVATION_POINTS = 50 # Number of points to be observed along the sector polygon edges
        #     self.DISTANCE_MARGIN = 5 # km

        # if self.env_name in ('staticobstaclesectorenv-v0'):
        #     # print(f'initialise_RL for {self.env_name} at {self.scn_idx}')
        #     # Model parameters
        #     self.NUM_OBSTACLES = 10 #np.random.randint(1,5)
        #     self.AC_SPD = 150 # m/s
        #     self.D_HEADING = 45 #degrees
        #     self.D_SPEED = 20/3 # m/s

        #     self.ACTION_FREQUENCY = 10

        #     self.TOTAL_OBSERVATION_POINTS = 50 # Number of points to be observed along the sector polygon edges
        #     self.DISTANCE_MARGIN = 5 # km
        # if self.env_name in ('staticobstaclecrenv-v0'):
        #     # print(f'initialise_RL for {self.env_name} at {self.scn_idx}')
        #     # Model parameters
        #     self.NUM_OBSTACLES = 5 #np.random.randint(1,5)
        #     self.NUM_INTRUDERS = 5
        #     self.AC_SPD = 150 # m/s
        #     self.D_HEADING = 45 #degrees
        #     self.D_SPEED = 20/3 # m/s

        #     self.ACTION_FREQUENCY = 10
        #     self.DISTANCE_MARGIN = 5 # km

        # if self.env_name in ('staticobstacleenv-v0'):
        #     # print(f'initialise_RL for {self.env_name} at {self.scn_idx}')
        #     # Model parameters
        #     self.NUM_OBSTACLES = 10 #np.random.randint(1,5)
        #     self.AC_SPD = 150 # m/s
        #     self.D_HEADING = 45 #degrees
        #     self.D_SPEED = 20/3 # m/s

        #     self.ACTION_FREQUENCY = 10
        #     self.DISTANCE_MARGIN = 5 # km

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
        
    @core.timed_function(name='update', dt=CRT.constants.TIMESTEP)
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
            self.init_aircraft = self.number_aircraft
            self.initial_observation_done = False
            self.plan_path = True

            stack.stack(f'ECHO Starting scenario batch_{self.scn_idx}, repeats left: {self.repeats}')
            print(f'Starting scenario batch_{self.scn_idx}, repeats left: {self.repeats}')

        for ac_idx, id in enumerate(traf.id):
            _, dest_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1])
            if dest_dis * Baselinetools.constants.NM2KM < self.DISTANCE_MARGIN:
                # stack.stack(f"ECHO Aircraft {id} has reached the destination waypoint.")
                # print(f"Aircraft {id} has reached the destination waypoint.")
                stack.process(f"DELETE {id}")
                continue  # skip action if the aircraft has reached its destination
            obs = self._get_obs(ac_idx)

            # if self.plan_path:
            if self.algorithm.lower() not in ('rein_weston'):
                G = Baselinetools.functions.build_graph_from_edges(obs)
                start = (bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
                goal  = (bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1])

                path = Baselinetools.functions.astar_route(G, start, goal)
                # import debug
                # debug.pink(f"A* path: {path}")

            if self.algorithm.lower() in ('rein_weston'):
                # import debug
                # debug.pink(f'obs for path planning: {obs}')
                # self.planned_path = []
                self._path_planning(ac_idx)

            # if ac_idx == self.number_aircraft - 1:
            #     self.plan_path = False

            #### INSERT HERE PATH PLANNING ALGORITHM TO COMPUTE WAYPOINTS BASED ON THE GRAPH for the a* or other graph-based path planning
            # action = 
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

        if (traf.id == [] or simt > self.max_sim_time) and (self.repeats > 0) and (self.initial_observation_done):
            print(f'Ending scenario batch_{self.scn_idx}')
            stack.process('END_SCEN')


        if (traf.id == [] or simt > self.max_sim_time) and (self.repeats == 0) and (self.initial_observation_done):
            print(f'Ending scenario batch_{self.scn_idx}')
            stack.process('QUIT')

    def _path_planning(self, ac_idx):
        '''used for debugging'''
        # import pickle

        # # Saving the objects:
        # with open('de-bugging_obstacles/objs.pkl', 'wb') as f:
        #     obj0 = self.other_aircraft_names
        #     obj1 = bs.traf.lat
        #     obj2 = bs.traf.lon
        #     obj3 = bs.traf.alt
        #     obj4 = bs.traf.tas
        #     obj5 = self.wpt_lat
        #     obj6 = self.wpt_lon
        #     obj7 = self.obstacle_vertices
        #     pickle.dump([obj0, obj1, obj2, obj3, obj4, obj5, obj6, obj7], f)

        # # Getting back the objects:
        # with open('de-bugging_obstacles/objs_impossible_route_0.pkl', 'rb') as f:
            # obj0, obj1, obj2, obj3, obj4, obj5, obj6, obj7 = pickle.load(f)
            # obj7 = self._merge_overlapping_obstacles(obj7)
        '''END used for debugging'''

        merged_obstacles_vertices = path_plan.merge_overlapping_obstacles(self.obstacle_vertices)
        planned_path = path_plan.det_path_planning(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.alt[ac_idx], bs.traf.tas[ac_idx]/kts, bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1], merged_obstacles_vertices)

        '''used for debugging'''
        # ac_idx = bs.traf.id2idx(obj0[i])
        # planned_path_other_aircraft = path_plan.det_path_planning(obj1[ac_idx], obj2[ac_idx], obj3[ac_idx], obj4[ac_idx]/kts, obj5[i+1], obj6[i+1], obj7)
        '''END used for debugging'''

        # self.planned_path.append(planned_path)
        # remove previously added waypoints for this aircraft except the destination waypoint
        for element in traf.ap.route[ac_idx].wpname:
            if element != traf.ap.route[ac_idx].wpname[-1]:  # keep the destination waypoint
                stack.stack(f"DELWPT {bs.traf.id[ac_idx]} {element}")

        for element in planned_path[1:-1]:  # skip the first and last point as they correspond to the current position and the destination waypoint
            bs.stack.stack(f"ADDWPT {bs.traf.id[ac_idx]} {element[0]} {element[1]}")
            # debug.green(f"Added waypoint {element} for aircraft {bs.traf.id[ac_idx]}")
            
    def _get_obs(self, ac_idx):
        """
        Observation is the normalized. Normalisation logic should be studied further
      """
        if self.initialise_observation_flag:
            # this logic is used when we want to generate the graph to feed to the path planning algorithm only once. otherwise, if the initialization flag is removed, the new path can be computed at every update time step
            waypoint_distances = []

            self.init_aircraft -= 1

            # Give aircraft initial heading
            for ac_idx_for_heading, _ in enumerate(traf.id):
                _, initial_wpt_dist = tools.geo.kwikqdrdist(traf.lat[ac_idx_for_heading], traf.lon[ac_idx_for_heading], bs.traf.ap.route[ac_idx_for_heading].wplat[-1], bs.traf.ap.route[ac_idx_for_heading].wplon[-1])
                waypoint_distances.append(initial_wpt_dist)

            # # Scaling factor for the distances in the observation vector
            # self.waypoint_distance_max = max(waypoint_distances)

            # if self.env_name in ('staticobstaclesectorcrenv-v0', 'staticobstaclesectorenv-v0'):
            #     sector = tools.areafilter.basic_shapes[sector_name]
            #     coordinates = sector.coordinates
            #     latitudes = coordinates[::2]
            #     longitudes = coordinates[1::2]
            #     self.sector_points = RLtools.functions.resample_closed_border(latitudes, longitudes, self.TOTAL_OBSERVATION_POINTS)

            # # Display in BlueSky the sector points contained in the observation 
            # print(f'sector points: {self.sector_points}, shape: {np.array(self.sector_points).shape}, type: {type(self.sector_points)}')
            # coordinates = ", ".join([f"{lat} {lon}" for lat, lon in self.sector_points])
            # stack.stack(f"POLY SECTOR_OBSERVATION, {coordinates}")
            self.obstacle_vertices = []
            # self.obstacle_centre_lon = []
            # self.obstacle_radius = []

            self.number_obstacles = 0
            for shape_name, shape in tools.areafilter.basic_shapes.items():
                if shape_name != sector_name:
                # if shape_name != sector_name and shape_name != 'WEATHER_CELL' and shape_name != 'VOLCANIC_CELL':
                    self.number_obstacles += 1
                    # print(f'Processing obstacle: {shape_name}')
                    coordinates = shape.coordinates

                    self.edges = []
                    i_point = 0
                    for point in zip(coordinates[::2], coordinates[1::2]):

                        i_point += 1
                        # print(f'Processing sector point {i_point} with coordinates: {point}')
                        # print(f"Point: {point[0]}, {point[1]} type: {type(point[0])}, {type(point[1])}")
                        _, orig2node_distance = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], point[0], point[1])
                        self.edges.append(((bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]), point, 1, orig2node_distance))
                        edge = self.edges[-1]

                        # if self.init_aircraft == 9:
                        #     stack.stack(f"LINE {shape_name}_{ac_idx}_O_{i_point} {edge[0][0]} {edge[0][1]} {edge[1][0]} {edge[1][1]}")

                        _, dest2node_distance = bs.tools.geo.kwikqdrdist(bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1], point[0], point[1])
                        self.edges.append(((bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1]), point, 1, dest2node_distance))

                        # stack.echo(f'Obstacle {shape_name} orig pair coordinates: {self.edges}')
                        edge = self.edges[-1]
                        
                        # if self.init_aircraft == 9:
                        #     stack.stack(f"LINE {shape_name}_{ac_idx}_D_{i_point} {edge[0][0]} {edge[0][1]} {edge[1][0]} {edge[1][1]}")

                    coordinates = list(zip(coordinates[::2], coordinates[1::2]))
                    # debug.cyan(f'Obstacle {shape_name} coordinates (lat, lon): {coordinates}')

            #         (lat_c, lon_c), radius = RLtools.functions.bounding_circle_geodesic(coordinates)
                    self.obstacle_vertices.append(coordinates)
            #         self.obstacle_centre_lon.append(lon_c)
            #         self.obstacle_radius.append(radius)
            
            # self.max_obstacle_radius = max(self.obstacle_radius)
            if self.init_aircraft == 0:
                self.initialise_observation_flag = False
                self.initial_observation_done = True


        # destination waypoint
        # wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1])
    
        # destination_waypoint_distance = wpt_dis * RLtools.constants.NM2KM

        # drift = bs.traf.hdg[ac_idx] - wpt_qdr
        # destination_waypoint_drift = RLtools.functions.bound_angle_positive_negative_180(drift)

        # destination_waypoint_cos_drift = np.cos(np.deg2rad(destination_waypoint_drift))
        # destination_waypoint_sin_drift = np.sin(np.deg2rad(destination_waypoint_drift))


        ################# Insert here the logic for the construction of the graph for weather disturbances and volcanic disturbances.
        ################# Insert here the logic for the construction of the graph for the restricted areas in case we want to re-plan at every update time step
        # obstacles 
        # obstacle_centre_distance = []
        # obstacle_centre_cos_bearing = []
        # obstacle_centre_sin_bearing = []

        # for obs_idx in range(self.number_obstacles):
            
        #     obs_centre_qdr, obs_centre_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.obstacle_centre_lat[obs_idx], self.obstacle_centre_lon[obs_idx])
        #     obs_centre_dis = obs_centre_dis * RLtools.constants.NM2KM #KM        
        #     bearing = bs.traf.hdg[ac_idx] - obs_centre_qdr
            
        #     bearing = RLtools.functions.bound_angle_positive_negative_180(bearing)

        #     obstacle_centre_distance.append(obs_centre_dis)
        #     obstacle_centre_cos_bearing.append(np.cos(np.deg2rad(bearing)))
        #     obstacle_centre_sin_bearing.append(np.sin(np.deg2rad(bearing)))

        # weather disturbances
        if bs.tools.areafilter.basic_shapes.get('WEATHER_CELL') is not None:
            # print(f'Processing WEATHER_CELL obstacle for observation')
            coordinates = bs.tools.areafilter.basic_shapes['WEATHER_CELL'].coordinates
            # coordinates = list(zip(coordinates[::2], coordinates[1::2]))
            # (weather_cell_center_lat, weather_cell_center_lon), weather_cell_radius = RLtools.functions.bounding_circle_geodesic(coordinates)
            # weather_cell_centre_qdr, weather_cell_centre_distance = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], weather_cell_center_lat, weather_cell_center_lon)
            # weather_cell_centre_distance = weather_cell_centre_distance * RLtools.constants.NM2KM #KM            
            # bearing = RLtools.functions.bound_angle_positive_negative_180(bs.traf.hdg[ac_idx] - weather_cell_centre_qdr)

            # weather_cell_centre_cos_bearing = np.cos(np.deg2rad(bearing))
            # weather_cell_centre_sin_bearing = np.sin(np.deg2rad(bearing))

        # volcanic disturbances
        if bs.tools.areafilter.basic_shapes.get('VOLCANIC_CELL') is not None:
            # print(f'Processing VOLCANIC_CELL obstacle for observation')
            coordinates = bs.tools.areafilter.basic_shapes['VOLCANIC_CELL'].coordinates
            # coordinates = list(zip(coordinates[::2], coordinates[1::2]))
            # (volcanic_cell_center_lat, volcanic_cell_center_lon), volcanic_cell_radius = RLtools.functions.bounding_circle_geodesic(coordinates)
            # volcanic_cell_centre_qdr, volcanic_cell_centre_distance = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], volcanic_cell_center_lat, volcanic_cell_center_lon)
            # volcanic_cell_centre_distance = volcanic_cell_centre_distance * RLtools.constants.NM2KM #KM            
            # bearing = RLtools.functions.bound_angle_positive_negative_180(bs.traf.hdg[ac_idx] - volcanic_cell_centre_qdr)

            # volcanic_cell_centre_cos_bearing = np.cos(np.deg2rad(bearing))
            # volcanic_cell_centre_sin_bearing = np.sin(np.deg2rad(bearing))
        
        # obstacle_radius = self.obstacle_radius.copy()

        # Find the RLtools.constants.NUM_OBSTACLES closest obstacles to the ownship (restricted areas or weather cells)
        # RLtools.constants.NUM_OBSTACLES depends on how many obstacles the AI has been trained with
        if bs.tools.areafilter.basic_shapes.get('WEATHER_CELL') is not None:
            # obstacle_centre_distance.append(weather_cell_centre_distance)
            # obstacle_centre_cos_bearing.append(weather_cell_centre_cos_bearing)
            # obstacle_centre_sin_bearing.append(weather_cell_centre_sin_bearing)
            # obstacle_radius.append(weather_cell_radius)
            pass
        
        if bs.tools.areafilter.basic_shapes.get('VOLCANIC_CELL') is not None:
            # obstacle_centre_distance.append(volcanic_cell_centre_distance)
            # obstacle_centre_cos_bearing.append(volcanic_cell_centre_cos_bearing)
            # obstacle_centre_sin_bearing.append(volcanic_cell_centre_sin_bearing)
            # obstacle_radius.append(volcanic_cell_radius)
            pass

        # obstacle_centre_distance = np.array(obstacle_centre_distance)
        # obstacle_centre_cos_bearing = np.array(obstacle_centre_cos_bearing)
        # obstacle_centre_sin_bearing = np.array(obstacle_centre_sin_bearing)
        # obstacle_radius = np.array(obstacle_radius)
        
        # # print(f'self.NUM_OBS: {self.NUM_OBSTACLES}')
        # # select the closest RLtools.constants.NUM_OBSTACLES obstacles
        # idx_sorted = np.argsort(obstacle_centre_distance)[:self.NUM_OBSTACLES]
        
        # obstacle_centre_distance = obstacle_centre_distance[idx_sorted]
        # obstacle_centre_cos_bearing = obstacle_centre_cos_bearing[idx_sorted]
        # obstacle_centre_sin_bearing = obstacle_centre_sin_bearing[idx_sorted]
        # obstacle_radius = obstacle_radius[idx_sorted]

        # # padding if less than RLtools.constants.NUM_OBSTACLES are present
        # num_missing = self.NUM_OBSTACLES - obstacle_centre_distance.size
        # if num_missing > 0:
        #     obstacle_centre_distance = np.pad(obstacle_centre_distance, (0, num_missing), 'constant', constant_values=0.0)
        #     obstacle_centre_cos_bearing = np.pad(obstacle_centre_cos_bearing, (0, num_missing), 'constant', constant_values=0.0)
        #     obstacle_centre_sin_bearing = np.pad(obstacle_centre_sin_bearing, (0, num_missing), 'constant', constant_values=0.0)
        #     obstacle_radius = np.pad(obstacle_radius, (0, num_missing), 'constant', constant_values=0.0)


        # if self.env_name in ('staticobstaclesectorcrenv-v0', 'staticobstaclecrenv-v0'):
        #     # intruder observation
        #     intruders_lat = np.delete(bs.traf.lat, ac_idx)
        #     intruders_lon = np.delete(bs.traf.lon, ac_idx)
            
        #     int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], intruders_lat, intruders_lon)

        #     # index of the closes N_INTRUDERS intruders for arrays of intruders_lat, intruders_lon, int_qdr, int_dis, intruders_speed, intruders_heading: the ownship is excluded a priori in these
        #     closest_intruders_idx = np.argsort(int_dis)[:self.NUM_INTRUDERS]

        #     intruder_distance = int_dis[closest_intruders_idx] * RLtools.constants.NM2KM

        #     # relative heading
        #     bearing = bs.traf.hdg[ac_idx] - int_qdr[closest_intruders_idx]
        #     for bearing_idx in range(len(bearing)):
        #         bearing[bearing_idx] = RLtools.functions.bound_angle_positive_negative_180(bearing[bearing_idx])

        #     intruder_cos_bearing = np.cos(np.deg2rad(bearing))
        #     intruder_sin_bearing = np.sin(np.deg2rad(bearing))

        #     intruders_heading = np.delete(bs.traf.hdg, ac_idx)
        #     intruders_speed = np.delete(bs.traf.gs, ac_idx)
        #     heading_difference = bs.traf.hdg[ac_idx] - intruders_heading[closest_intruders_idx]

        #     # relative speed
        #     intruder_x_difference_speed = - np.cos(np.deg2rad(heading_difference)) * intruders_speed[closest_intruders_idx]
        #     intruder_y_difference_speed = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * intruders_speed[closest_intruders_idx]

        #     # padding the observation if less than NUM_INTRUDERS are present
        #     if len(closest_intruders_idx) < self.NUM_INTRUDERS:
        #         num_missing_intruders = self.NUM_INTRUDERS - len(closest_intruders_idx)
        #         intruder_distance = np.pad(intruder_distance, (0, num_missing_intruders), 'constant', constant_values=(self.waypoint_distance_max,))
        #         intruder_cos_bearing =  np.pad(intruder_cos_bearing, (0, num_missing_intruders), 'constant', constant_values=(0,))
        #         intruder_sin_bearing =  np.pad(intruder_sin_bearing, (0, num_missing_intruders), 'constant', constant_values=(0,))
        #         intruder_x_difference_speed =  np.pad(intruder_x_difference_speed, (0, num_missing_intruders), 'constant', constant_values=(0,))
        #         intruder_y_difference_speed =  np.pad(intruder_y_difference_speed, (0, num_missing_intruders), 'constant', constant_values=(0,))

        # if self.env_name in ('staticobstaclesectorcrenv-v0', 'staticobstaclesectorenv-v0'):
        #     # sector polygon edges observation
        #     sector_points_distance = []
        #     sector_points_cos_drift = []
        #     sector_points_sin_drift = []

        #     # Calculate distance and bearing from the ownship to each of the sector points
        #     for point_index in range(len(self.sector_points)):
        #         sector_points_qdr, sector_points_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.sector_points[point_index,0],self.sector_points[point_index,1])
        #         # print(f'point_index: {point_index}, sector_points_dis: {sector_points_dis}, sector_points_qdr: {sector_points_qdr}')
        #         sector_points_distance.append(sector_points_dis * RLtools.constants.NM2KM)

        #         drift = bs.traf.hdg[ac_idx] - sector_points_qdr

        #         drift = RLtools.functions.bound_angle_positive_negative_180(drift)

        #         sector_points_cos_drift.append(np.cos(np.deg2rad(drift)))
        #         sector_points_sin_drift.append(np.sin(np.deg2rad(drift)))
        # # print(f'self.obstacle_radius length is: {len(self.obstacle_radius)}')
        # # if len(closest_intruders_idx) < RLtools.constants.NUM_INTRUDERS:
        # #     print(f'observation: {observation}')
        # if self.env_name in ('staticobstaclesectorcrenv-v0'):
        #     observation = {
        #             "intruder_distance": np.array(intruder_distance).reshape(-1)/self.waypoint_distance_max,
        #             "intruder_cos_difference_pos": np.array(intruder_cos_bearing).reshape(-1),
        #             "intruder_sin_difference_pos": np.array(intruder_sin_bearing).reshape(-1),
        #             "intruder_x_difference_speed": np.array(intruder_x_difference_speed).reshape(-1)/self.AC_SPD,
        #             "intruder_y_difference_speed": np.array(intruder_y_difference_speed).reshape(-1)/self.AC_SPD,
        #             "destination_waypoint_distance": np.array(destination_waypoint_distance).reshape(-1)/self.waypoint_distance_max,
        #             "destination_waypoint_cos_drift": np.array(destination_waypoint_cos_drift).reshape(-1),
        #             "destination_waypoint_sin_drift": np.array(destination_waypoint_sin_drift).reshape(-1),
        #             # observations on obstacles
        #             "restricted_area_radius": np.array(obstacle_radius).reshape(-1)/self.max_obstacle_radius,
        #             "restricted_area_distance": np.array(obstacle_centre_distance).reshape(-1)/self.waypoint_distance_max,
        #             "cos_difference_restricted_area_pos": np.array(obstacle_centre_cos_bearing).reshape(-1),
        #             "sin_difference_restricted_area_pos": np.array(obstacle_centre_sin_bearing).reshape(-1),
        #             # observations on sector polygon edges and points along the edges
        #             "sector_points_distance": np.array(sector_points_distance).reshape(-1)/self.waypoint_distance_max,
        #             "sector_points_cos_drift": np.array(sector_points_cos_drift).reshape(-1),
        #             "sector_points_sin_drift": np.array(sector_points_sin_drift).reshape(-1)
        #         }
        # if self.env_name in ('staticobstaclesectorenv-v0'):
        #     observation = {
        #             "destination_waypoint_distance": np.array(destination_waypoint_distance).reshape(-1)/self.waypoint_distance_max,
        #             "destination_waypoint_cos_drift": np.array(destination_waypoint_cos_drift).reshape(-1),
        #             "destination_waypoint_sin_drift": np.array(destination_waypoint_sin_drift).reshape(-1),
        #             # observations on obstacles
        #             "restricted_area_radius": np.array(obstacle_radius).reshape(-1)/self.max_obstacle_radius,
        #             "restricted_area_distance": np.array(obstacle_centre_distance).reshape(-1)/self.waypoint_distance_max,
        #             "cos_difference_restricted_area_pos": np.array(obstacle_centre_cos_bearing).reshape(-1),
        #             "sin_difference_restricted_area_pos": np.array(obstacle_centre_sin_bearing).reshape(-1),
        #             # observations on sector polygon edges and points along the edges
        #             "sector_points_distance": np.array(sector_points_distance).reshape(-1)/self.waypoint_distance_max,
        #             "sector_points_cos_drift": np.array(sector_points_cos_drift).reshape(-1),
        #             "sector_points_sin_drift": np.array(sector_points_sin_drift).reshape(-1)
        #         }
        # if self.env_name in ('staticobstaclecrenv-v0'):
        #     observation = {
        #             "intruder_distance": np.array(intruder_distance).reshape(-1)/self.waypoint_distance_max,
        #             "intruder_cos_difference_pos": np.array(intruder_cos_bearing).reshape(-1),
        #             "intruder_sin_difference_pos": np.array(intruder_sin_bearing).reshape(-1),
        #             "intruder_x_difference_speed": np.array(intruder_x_difference_speed).reshape(-1)/self.AC_SPD,
        #             "intruder_y_difference_speed": np.array(intruder_y_difference_speed).reshape(-1)/self.AC_SPD,
        #             "destination_waypoint_distance": np.array(destination_waypoint_distance).reshape(-1)/self.waypoint_distance_max,
        #             "destination_waypoint_cos_drift": np.array(destination_waypoint_cos_drift).reshape(-1),
        #             "destination_waypoint_sin_drift": np.array(destination_waypoint_sin_drift).reshape(-1),
        #             # observations on obstacles
        #             "restricted_area_radius": np.array(obstacle_radius).reshape(-1)/self.max_obstacle_radius,
        #             "restricted_area_distance": np.array(obstacle_centre_distance).reshape(-1)/self.waypoint_distance_max,
        #             "cos_difference_restricted_area_pos": np.array(obstacle_centre_cos_bearing).reshape(-1),
        #             "sin_difference_restricted_area_pos": np.array(obstacle_centre_sin_bearing).reshape(-1),
        #         }
        # if self.env_name in ('staticobstacleenv-v0'):
        #     observation = {
        #             "destination_waypoint_distance": np.array(destination_waypoint_distance).reshape(-1)/self.waypoint_distance_max,
        #             "destination_waypoint_cos_drift": np.array(destination_waypoint_cos_drift).reshape(-1),
        #             "destination_waypoint_sin_drift": np.array(destination_waypoint_sin_drift).reshape(-1),
        #             # observations on obstacles
        #             "restricted_area_radius": np.array(obstacle_radius).reshape(-1)/self.max_obstacle_radius,
        #             "restricted_area_distance": np.array(obstacle_centre_distance).reshape(-1)/self.waypoint_distance_max,
        #             "cos_difference_restricted_area_pos": np.array(obstacle_centre_cos_bearing).reshape(-1),
        #             "sin_difference_restricted_area_pos": np.array(obstacle_centre_sin_bearing).reshape(-1),
        #         }
        # print(
        #     f'observation for aircraft {bs.traf.id[ac_idx]}: '
        #     f'{ {k: v for k, v in observation.items()} }'
        # )

        # print(
        #     f'observation shapes for aircraft {bs.traf.id[ac_idx]}: '
        #     f'{ {k: v.shape for k, v in observation.items()} }'
        # )        
        return self.edges

    def _set_action(self, action, ac_idx):
        """
        Control each aircraft separately
        """
        # print(f'New action')
        # dv = action[1] * self.D_SPEED
        # dh = action[0] * self.D_HEADING

        id = traf.id[ac_idx]
        # heading_new = RLtools.functions.bound_angle_positive_negative_180(traf.hdg[ac_idx] + dh)
        # speed_new = (traf.cas[ac_idx] + dv) * RLtools.constants.MpS2Kt
        # stack.stack(f"ECHO Aircraft {id} - New heading: {heading_new} deg, New speed: {speed_new/RLtools.constants.MpS2Kt} m/s")
        # stack.stack(f"HDG {id} {heading_new}")
        # stack.stack(f"SPD {id} {speed_new}")
        # print(f'Action for aircraft {id} - traf.hdg: {traf.hdg[ac_idx]} -> {heading_new} with dh {dh}, traf.cas: {traf.cas[ac_idx]} m/s -> {speed_new/RLtools.constants.MpS2Kt} with dv {dv} m/s')
        stack.stack(f'ADDWPT {id} {action}')