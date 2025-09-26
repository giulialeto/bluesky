"""
    AI4REALNET - BlueSky plugin for deploying RL-based model
    ENV: StaticObstacleEnv-v0
    Authors: Giulia Leto
"""
from bluesky import core, stack, traf, tools, settings 
from stable_baselines3 import SAC
import numpy as np
import bluesky.plugins.ai4realnet_deploy_RL_tools.StaticObstacleEnv as RLtools
import bluesky as bs
import pandas as pd

algorithm = 'SAC'
env_name = 'StaticObstacleEnv-v0'

def init_plugin():
    StaticObstacle = DeployRL()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'DEPLOY_RL_StaticObstacleEnv',
        # The type of this plugin.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

class DeployRL(core.Entity):  
    def __init__(self):
        super().__init__()
        self.model = SAC.load(f"bluesky/plugins/ai4realnet_deploy_RL_tools/models/{env_name}/{env_name}_{algorithm}/model", env=None)
        
        # logging
        self.log_buffer = []   # temporary storage
        self.csv_file = (f"output/ai4realnet_deploy_RL_{env_name}_log.csv")

        # get number of aircraft in the simulation
        n_ac = len(bs.traf.id)

        # get destination waypoint of all aircraft in the simulation
        destination_list = bs.traf.ap.dest
        self.destination_coordinates = np.array([tuple(map(float, wpt.split(','))) for wpt in destination_list], dtype=np.float64)
        
        waypoint_distances = []
        # Give aircraft initial heading
        for ac_idx_aircraft in range(n_ac):
            initial_wpt_qdr, initial_wpt_dist = tools.geo.kwikqdrdist(traf.lat[ac_idx_aircraft], traf.lon[ac_idx_aircraft], self.destination_coordinates[ac_idx_aircraft][0], self.destination_coordinates[ac_idx_aircraft][1])
            bs.traf.hdg[ac_idx_aircraft] = initial_wpt_qdr
            waypoint_distances.append(initial_wpt_dist)

        # Scaling factor for the distances in the observation vector
        self.waypoint_distance_max = max(waypoint_distances)

        # Find observation points for static obstacles
        self.obstacle_center_lat = []
        self.obstacle_center_lon = []
        self.obstacle_radius = []
        # area_nm2 = []

        for shape_name, shape in tools.areafilter.basic_shapes.items():
            # print(f'Restricted area name: {shape_name}')
            if shape_name != 'LISBON_FIR':
                coordinates = shape.coordinates
                latitudes = coordinates[::2]
                longitudes = coordinates[1::2]
                (center_latlon, radius_m) = RLtools.functions.bounding_circle_geodesic(list(zip(latitudes, longitudes)))
                ''' Debugging'''
                self.obstacle_center_lat.append(center_latlon[0])
                self.obstacle_center_lon.append(center_latlon[1])
                self.obstacle_radius.append(radius_m)
                # Using the area for the observation?
                # area_nm2.append(RLtools.functions.poly_area_nm2(latitudes, longitudes))

                # Add circles around the restricted area for visualization of the observation
                # print(f'Obstacle {shape_name} center: {center_latlon}, radius: {radius_m}')
                # stack.stack(f"CIRCLE {shape_name}_bounding_circle, {center_latlon[0]}, {center_latlon[1]}, {radius_m/RLtools.constants.NM2KM/1000}")
                # stack.stack(f"COLOR {shape_name}_bounding_circle, YELLOW")

        # Scaling factor for the radius in the observation vector
        self.max_obstacle_radius = max(self.obstacle_radius)
        # self.max_obstacle_area_nm2 = max(area_nm2)

    @core.timed_function(name='StaticObstacle', dt=RLtools.constants.ACTION_FREQUENCY)
    def update(self):
        # controlling each aircraft separately
        for id in traf.id:
            ac_idx = traf.id2idx(id)
            obs = self._get_obs(ac_idx)
            ''' debugging code used to save and compare the observation vector with the one in the training'''
            # import pickle
            # with open('bluesky/plugins/ai4realnet_deploy_RL_tools/observation_test.pkl', 'rb') as f:
            #     obs = pickle.load(f)
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

    def _get_obs(self, ac_idx):
        """
        Observation is the normalized. Normalisation logic should be studied further
        """

        # destination waypoint
        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.destination_coordinates[ac_idx, 0], self.destination_coordinates[ac_idx, 1])
    
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
            obs_centre_qdr, obs_centre_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.obstacle_center_lat[obs_idx], self.obstacle_center_lon[obs_idx])
            obs_centre_dis = obs_centre_dis * RLtools.constants.NM2KM #KM        

            bearing = bs.traf.hdg[ac_idx] - obs_centre_qdr
            
            bearing = RLtools.functions.bound_angle_positive_negative_180(bearing)

            obstacle_centre_distance.append(obs_centre_dis)
            obstacle_centre_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            obstacle_centre_sin_bearing.append(np.sin(np.deg2rad(bearing)))
        
        observation = {
                "destination_waypoint_distance": np.array(destination_waypoint_distance).reshape(-1)/self.waypoint_distance_max,
                "destination_waypoint_cos_drift": np.array(destination_waypoint_cos_drift).reshape(-1),
                "destination_waypoint_sin_drift": np.array(destination_waypoint_sin_drift).reshape(-1),
                # observations on obstacles
                "restricted_area_radius": np.array(self.obstacle_radius).reshape(-1)/self.max_obstacle_radius,
                "restricted_area_distance": np.array(obstacle_centre_distance).reshape(-1)/self.waypoint_distance_max,
                "cos_difference_restricted_area_pos": np.array(obstacle_centre_cos_bearing).reshape(-1),
                "sin_difference_restricted_area_pos": np.array(obstacle_centre_sin_bearing).reshape(-1),
            }
        
        return observation

    def _set_action(self, action, ac_idx):
        """
        Control each aircraft separately
        """
        # print(f'New action')
        # Randomised action for testing
        # action = np.random.uniform(-1, 1, size=2)

        dv = action[1] * RLtools.constants.D_SPEED
        dh = action[0] * RLtools.constants.D_HEADING

        id = traf.id[ac_idx]
        heading_new = RLtools.functions.bound_angle_positive_negative_180(traf.hdg[ac_idx] + dh)
        speed_new = (traf.cas[ac_idx] + dv) * RLtools.constants.MpS2Kt
        # stack.stack(f"ECHO Aircraft {id} - New heading: {heading_new} deg, New speed: {speed_new/RLtools.constants.MpS2Kt} m/s")
        stack.stack(f"HDG {id} {heading_new}")
        stack.stack(f"SPD {id} {speed_new}")

        # print(f'Action for aircraft {id} - traf.hdg: {traf.hdg[ac_idx]} -> {heading_new} with dh {dh}, traf.cas: {traf.cas[ac_idx]} m/s -> {speed_new/RLtools.constants.MpS2Kt} with dv {dv} m/s')
