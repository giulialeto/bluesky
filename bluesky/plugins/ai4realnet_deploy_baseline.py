"""
    AI4REALNET -  Deliverable 1.4 BlueSky plugin for deploying RL-based model in batch scenarios
    Authors: Giulia Leto
    Date: Nov 2025
"""
from multiprocessing.util import debug
from os import path

from bluesky import core, stack, traf, tools, settings 
from bluesky.plugins.ai4realnet_deploy_baseline_tools_batch.functions import closest_point_on_polygon
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

        self.start_next = True # Flag to indicate whether to start the next scenario batch, set to True at the end of each scenario and to False after the first update loop of the next scenario batch
        self.repeats = 0 # Number of repetitions left for the current batch scenario
        self.total = 0 # Total number of repetitions, as specified in the scenario file
        self.scentime = []
        self.scencmd = []
        self.start_updates = False # Flag to indicate when to start the update loop, set to True after loading the scenario the first time, otherwise it would try to update the state before the aircraft are spawned
        self.max_sim_time = 3600  # seconds
        os.makedirs(f'scenario/{save_dir}', exist_ok=True)
        # self.add_buffer = False

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
        
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, cmd in enumerate(new_scencmd):
            line = cmd.strip().lower()

            # Parse: initialize_scenario NUM_AIRCRAFT, NUM_OBSTACLES
            if line.startswith("initialize_scenario"):
                # Remove command name
                args = line.replace("initialize_scenario", "").strip()
                parts = [p.strip() for p in args.split(",")]

                if len(parts) == 2:
                    self.inital_number_aircraft = int(parts[0])
                    self.initial_number_obstacles = int(parts[1])
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
        self.OBSTACLE_BUFFER_DISTANCE_NM = 5 # NM

        self.start_updates = True
        self.delete_buffer_weather = True
        self.delete_buffer_volcanic = True
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
        """ Update function called with frequency dt. """

        if self.start_updates == False:
            return
        
        self._buffer_added_this_step = False # flag to ensure the buffer is only added once per update step, not for each aircraft

        if self.start_next and self.repeats >= 0:
            self.repeats -= 1
            self.scn_idx = self.total - self.repeats
            bs.sim.start_batch_scenario(f'batch_{self.scn_idx}', list(self.scentime), list(self.scencmd))
            stack.stack(f'SAVEIC {save_dir}/{self.timestamp}_{self.env_name}_{self.algorithm}_{self.inital_number_aircraft}_{self.initial_number_obstacles}_{self.scn_idx}')

            self.start_next = False
            self.initialise_observation_flag = True # flag used to prevent the scenario from being quit when there are no aircraft in the scenario at the beginning, before the spawn command is executed

            stack.stack(f'ECHO Starting scenario batch_{self.scn_idx}, repeats left: {self.repeats}')
            print(f'Starting scenario batch_{self.scn_idx}, repeats left: {self.repeats}')

        # path planning for each aircraft
        for ac_idx, id in enumerate(traf.id):
            # stack.echo(f"Processing aircraft {id} at index {ac_idx}")
            # stack.echo(f"Current position of aircraft {id}: lat={bs.traf.lat[ac_idx]}, lon={bs.traf.lon[ac_idx]}, alt={bs.traf.alt[ac_idx]}, tas={bs.traf.tas[ac_idx]}, hdg={bs.traf.hdg[ac_idx]}")
            # check if the aircraft has reached its destination, if yes, delete it and skip the rest of the loop
            _, dest_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1])
            if dest_dis * Baselinetools.constants.NM2KM < self.DISTANCE_MARGIN:
                # stack.stack(f"ECHO Aircraft {id} has reached the destination waypoint.")
                stack.process(f"DELETE {id}")
                continue  # skip action if the aircraft has reached its destination

            # Observation of the obstacles. Since volcanic and weather disturbances are present, the obstacle varies over time.
            self._get_obs(ac_idx)

            # Path planning
            if self.algorithm.lower() not in ('rein_weston'):
                G = Baselinetools.functions.build_graph_from_edges(self.edges)
                start = (bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
                goal  = (bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1])

                path = Baselinetools.functions.astar_route(G, start, goal)
                # import debug
                # debug.pink(f"A* path: {path}")

            if self.algorithm.lower() in ('rein_weston'):
                self._path_planning(ac_idx)

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

        if (traf.id == [] or simt > self.max_sim_time) and (self.repeats > 0) and (self.initialise_observation_flag == False):
            print(f'Ending scenario batch_{self.scn_idx}')
            stack.process('END_SCEN')


        if (traf.id == [] or simt > self.max_sim_time) and (self.repeats == 0) and (self.initialise_observation_flag == False):
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

        # remove previously added waypoints for this aircraft except the destination waypoint
        for element in traf.ap.route[ac_idx].wpname:
            if element != traf.ap.route[ac_idx].wpname[-1]:  # keep the destination waypoint
                stack.stack(f"DELWPT {bs.traf.id[ac_idx]} {element}")

        for element in planned_path[1:-1]:  # skip the first and last point as they correspond to the current position and the destination waypoint
        # for element in planned_path[1:]:  # skip the first and last point as they correspond to the current position and the destination waypoint
            bs.stack.stack(f"ADDWPT {bs.traf.id[ac_idx]} {element[0]} {element[1]}")
            
    def _get_obs(self, ac_idx):
        """
        Observation is the normalized. Normalisation logic should be studied further
        """

        self.obstacle_vertices = []
        self.shape_name = []

        self.current_number_obstacles = 0
        
        self.edges = []

        for shape_name, shape in tools.areafilter.basic_shapes.items():
            # process coordinates for all the areas in basic shapes except the one corresponding to the sector (restricted areas + volcanic/weather disturbances)
            if shape_name != sector_name and not shape_name.startswith("BUFFER_"):
                self.current_number_obstacles += 1
                coordinates = shape.coordinates

                for point in zip(coordinates[::2], coordinates[1::2]):

                    _, orig2node_distance = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], point[0], point[1])
                    self.edges.append(((bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]), point, 1, orig2node_distance))

                    _, dest2node_distance = bs.tools.geo.kwikqdrdist(bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1], point[0], point[1])
                    self.edges.append(((bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1]), point, 1, dest2node_distance))
                    
                    # if self.init_aircraft == 9:
                    #     stack.stack(f"LINE {shape_name}_{ac_idx}_D_{i_point} {edge[0][0]} {edge[0][1]} {edge[1][0]} {edge[1][1]}")

                coordinates = list(zip(coordinates[::2], coordinates[1::2]))

                self.shape_name.append(shape_name)
                self.obstacle_vertices.append(coordinates)
            if shape_name.startswith("BUFFER_WEATHER_CELL"):
                self.delete_buffer_weather = True
            if shape_name.startswith("BUFFER_VOLCANIC_CELL"):
                self.delete_buffer_volcanic = True
        if self.add_buffer and not self._buffer_added_this_step: # only need to add the buffer once per update, not for each aircraft
            self._add_buffer()
        self.obstacle_vertices = self.buffered_obstacle_vertices

        if self.initialise_observation_flag:
            self.initialise_observation_flag = False

    def _add_buffer(self):
        """ Add a safety buffer around the obstacles by expanding their vertices outward by a specified distance. """
        
        # add buffer around obstacles at every time step, since the observation of the obstacles is updated at every time step to account for the volcanic and weather disturbances, which might appear/disappear/move during the scenario
        self.buffered_obstacle_vertices = Baselinetools.functions.buffer_obstacles_nm(self.obstacle_vertices, self.OBSTACLE_BUFFER_DISTANCE_NM)

        # Delete the buffered obstacles from the previous time step in case there is no more volcanic/weather disturbance in the scenario. 
        if self.delete_buffer_weather:
            if not any(shape_name.startswith("WEATHER_CELL") for shape_name, _ in tools.areafilter.basic_shapes.items()):
                stack.process(f'DELETE BUFFER_WEATHER_CELL')
                self.delete_buffer_weather = False
        if self.delete_buffer_volcanic:
            if not any(shape_name.startswith("VOLCANIC_CELL") for shape_name, _ in tools.areafilter.basic_shapes.items()):
                stack.process(f'DELETE BUFFER_VOLCANIC_CELL')
                self.delete_buffer_volcanic = False

        # Adding the buffered obstacles to BlueSky as POLY areas (ONLY ONCE for the restricted areas, continuously for the volcanic and weather cells)
        for i, polygon in enumerate(self.buffered_obstacle_vertices):

            # Make a mutable copy
            p = list(polygon)

            # Flatten [(lat, lon), (lat, lon), ...] -> [lat, lon, lat, lon, ...]
            points = [coord for latlon in p for coord in latlon]

            poly_name = f"BUFFER_{self.shape_name[i]}"
            # Draw polygon in BlueSky
            if self.shape_name[i].startswith("RESTRICTED_AREA"):
                if self.initialise_observation_flag:
                    stack.process(f"POLY {poly_name}, " + ", ".join(map(str, points)))
                    stack.process(f"COLOR {poly_name}, YELLOW")
            else:
                stack.process(f"POLY {poly_name}, " + ", ".join(map(str, points)))
                stack.process(f"COLOR {poly_name}, YELLOW")

            if self.initialise_observation_flag:
                for ac_idx, id in enumerate(traf.id):
                    if bs.tools.areafilter.checkInside(poly_name, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.alt[ac_idx]):
                        # if the aircraft is inside an obstacle, find the closest point on the obstacle polygon and move the aircraft to that point
                        (new_lat, new_lon), _ = Baselinetools.functions.closest_point_on_polygon((bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]), polygon, Baselinetools.constants.SAFETY_MARGIN_BUFFER)

                        ## save aircraft data 
                        lat_before = bs.traf.lat[ac_idx]
                        lon_before = bs.traf.lon[ac_idx]
                        alt_before = bs.traf.alt[ac_idx]
                        tas_before = bs.traf.tas[ac_idx]
                        hdg_before = bs.traf.hdg[ac_idx]
                        dest_lat_before = bs.traf.ap.route[ac_idx].wplat[-1]
                        dest_lon_before = bs.traf.ap.route[ac_idx].wplon[-1] 

                        bs.stack.process(f"CIRCLE BUFFER_{id},{lat_before} {lon_before}, 5")
                        bs.stack.process(f"COLOUR BUFFER_{id},255,165,0")
                        bs.stack.process(f"DEL {id}")
                        bs.stack.process(f'CRE {id}, A320, {new_lat} {new_lon} {hdg_before} {alt_before} {tas_before}')
                        bs.stack.process(f'DEST {id} {dest_lat_before} {dest_lon_before}')

                    if bs.tools.areafilter.checkInside(poly_name, bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1], bs.traf.alt[ac_idx]):
                        # if the aircraft is inside an obstacle, find the closest point on the obstacle polygon and move the aircraft to that point
                        (new_lat_dest, new_lon_dest), _ = Baselinetools.functions.closest_point_on_polygon((bs.traf.ap.route[ac_idx].wplat[-1], bs.traf.ap.route[ac_idx].wplon[-1]), polygon, Baselinetools.constants.SAFETY_MARGIN_BUFFER)

                        ## save aircraft data 
                        lat_before = bs.traf.lat[ac_idx]
                        lon_before = bs.traf.lon[ac_idx]
                        alt_before = bs.traf.alt[ac_idx]
                        tas_before = bs.traf.tas[ac_idx]
                        hdg_before = bs.traf.hdg[ac_idx]
                        dest_lat_before = bs.traf.ap.route[ac_idx].wplat[-1]
                        dest_lon_before = bs.traf.ap.route[ac_idx].wplon[-1] 
                        
                        bs.stack.process(f"CIRCLE BUFFER_{id},{dest_lat_before} {dest_lon_before}, 5")
                        bs.stack.process(f"COLOUR BUFFER_{id},255,165,0")
                        bs.stack.process(f"DEL {id}")
                        bs.stack.process(f'CRE {id}, A320, {lat_before} {lon_before} {hdg_before} {alt_before} {tas_before}')
                        bs.stack.process(f'DEST {id} {new_lat_dest} {new_lon_dest}')


                    bs.stack.process(f"CIRCLE BUFFER_{id}_dest_test,{bs.traf.ap.route[ac_idx].wplat[-1]} {bs.traf.ap.route[ac_idx].wplon[-1]}, 5")
                    bs.stack.process(f"COLOUR BUFFER_{id}_dest_test,0,255,0")

                        # bs.stack.process(f"MOVE {id} {lat} {lon}")

                # if bs.tools.areafilter.checkInside(poly_name, traf.ap.route[ac_idx].wplat[-1], traf.ap.route[ac_idx].wplon[-1], bs.traf.alt[ac_idx]):
                #     # if the destination waypoint is inside an obstacle, find the closest point on the obstacle polygon and move the waypoint to that point
                #     (lat, lon), _ = Baselinetools.functions.closest_point_on_polygon((traf.ap.route[ac_idx].wplat[-1], traf.ap.route[ac_idx].wplon[-1]), polygon)

                #     traf.ap.route[ac_idx].wplat[-1] = lat
                #     traf.ap.route[ac_idx].wplon[-1] = lon
                    # bs.stack.process(f"DELWPT {bs.traf.id[ac_idx]} {traf.ap.route[ac_idx].wplat[-1]} {traf.ap.route[ac_idx].wplon[-1]}")
                    # bs.stack.process(f"ADDWPT {bs.traf.id[ac_idx]} {lat} {lon}")
        self._buffer_added_this_step = True

    def _set_action(self, action, ac_idx):
        """
        Control each aircraft separately
        """

        stack.stack(f'ADDWPT {traf.id[ac_idx]} {action}')


    @stack.command(name='BUFFER')
    def set_active(self, state: str = 'OFF'):
        """Activate or deactivate buffer around obstacles.

        Args:
            state (str): 'ON' or 'OFF'.
        """
        if state.upper() == 'ON':
            self.add_buffer = True
        elif state.upper() == 'OFF':
            self.add_buffer = False


    @stack.command(name='START_NEXT')
    def start_next_scenario(self):
        """Start the next scenario in the batch.
        """
        self.start_next = True