"""
    AI4REALNET - BlueSky plugin for deploying RL-based model
    Authors: Giulia Leto
"""
from bluesky import core, stack, traf, tools, settings 
from stable_baselines3 import SAC
import numpy as np
import bluesky.plugins.ai4realnet_deploy_RL_tools.StaticObstacleSectorEnv as RLtools
import bluesky as bs

import debug

# TODO make this such that you can select the algorithm in settings
# settings.set_variable_defaults(algorithm='SAC')
algorithm = 'SAC'
env_name = 'StaticObstacleSectorEnv'
def init_plugin():
    StaticObstacleSectorCR = DeployRL()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'DEPLOY_RL_StaticObstacleSectorEnv',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

class DeployRL(core.Entity):  
    def __init__(self):
        super().__init__()
        self.model = SAC.load(f"bluesky/plugins/ai4realnet_deploy_RL_tools/models/{env_name}-v0/{env_name}-v0_{algorithm}/model", env=None)

        # number of aircraft in the simulation is set by the training conditions?
        # get number of aircraft in the simulation
        n_ac = len(bs.traf.id)

        # get destination waypoint of all aircraft in the simulation
        destination_list = bs.traf.ap.dest

        self.destination_coordinates = np.array([tuple(map(float, wpt.split(','))) for wpt in destination_list], dtype=np.float64)

        # Give aircraft initial heading
        for ac_idx_aircraft in range(n_ac):
            
            initial_wpt_qdr, _ = tools.geo.kwikqdrdist(traf.lat[ac_idx_aircraft], traf.lon[ac_idx_aircraft], self.destination_coordinates[ac_idx_aircraft][0], self.destination_coordinates[ac_idx_aircraft][1])
            bs.traf.hdg[ac_idx_aircraft] = initial_wpt_qdr

        self.obstacle_center_lat = []
        self.obstacle_center_lon = []
        self.obstacle_radius = []
        # Find observation points for static obstacles
        for shape_name, shape in tools.areafilter.basic_shapes.items():
            # print(f'shape_name: {shape_name}')
            if shape_name != 'LISBON_FIR':
                coordinates = shape.coordinates
                latitudes = coordinates[::2]
                longitudes = coordinates[1::2]
                (center_latlon, radius_m) = RLtools.functions.bounding_circle_geodesic(list(zip(latitudes, longitudes)))
                ''' Debugging'''
                self.obstacle_center_lat.append(center_latlon[0])
                self.obstacle_center_lon.append(center_latlon[1])
                self.obstacle_radius.append(radius_m)
                # print(f'Obstacle {shape_name} center: {center_latlon}, radius: {radius_m}')
                # stack.stack(f"CIRCLE {shape_name}_bounding_circle, {center_latlon[0]}, {center_latlon[1]}, {radius_m/RLtools.constants.NM2KM/1000}")
                # stack.stack(f"COLOR {shape_name}_bounding_circle, YELLOW")
        
        # Find observation points for sector
        sector = tools.areafilter.basic_shapes['LISBON_FIR']
        coordinates = sector.coordinates
        latitudes = coordinates[::2]
        longitudes = coordinates[1::2]
        self.sector_points = RLtools.functions.resample_closed_border(latitudes, longitudes, RLtools.constants.TOTAL_OBSERVATION_POINTS)
        ''' Debugging'''
        # print(f'sector points: {self.sector_points}, shape: {np.array(self.sector_points).shape}, type: {type(self.sector_points)}')
        # coordinates = ", ".join([f"{lat} {lon}" for lat, lon in self.sector_points])
        # stack.stack(f"POLY SECTOR_OBSERVATION, {coordinates}")


    @core.timed_function(name='StaticObstacleSector', dt=RLtools.constants.ACTION_FREQUENCY)
    def update(self):
        for id in traf.id:
            ac_idx = traf.id2idx(id)
            obs = self._get_obs(ac_idx)
            # clipped_obs = {key: np.clip(arr, -1.2, 1.2) for key, arr in obs.items()}
            action, _ = self.model.predict(obs, deterministic=True)
            self._set_action(action, ac_idx)
    #     obs = self._get_obs()
    # #     clipped_obs = {key: np.clip(arr, -1.2, 1.2) for key, arr in obs.items()}
    #     action, _ = self.model.predict(obs, deterministic=True)
    #     self._set_action(action)

    def _get_obs(self, ac_idx):
        # # """
        # # Observation is the normalized x and y coordinate of the aircraft
        # # """

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
            # print(f'obs_idx: {obs_idx}')
            obs_centre_qdr, obs_centre_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.obstacle_center_lat[obs_idx], self.obstacle_center_lon[obs_idx])
            obs_centre_dis = obs_centre_dis * RLtools.constants.NM2KM #KM        
            # debug.red(f'obs_idx: {obs_idx}, obs_centre_dis: {obs_centre_dis}, obs_centre_qdr: {obs_centre_qdr}')
            bearing = bs.traf.hdg[ac_idx] - obs_centre_qdr
            
            bearing = RLtools.functions.bound_angle_positive_negative_180(bearing)

            obstacle_centre_distance.append(obs_centre_dis)
            obstacle_centre_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            obstacle_centre_sin_bearing.append(np.sin(np.deg2rad(bearing)))

        # sector polygon edges observation
        sector_points_distance = []
        sector_points_cos_drift = []
        sector_points_sin_drift = []

        # Calculate distance and bearing from the ownship to each of the sector points
        for point_index in range(len(self.sector_points)):
            # print(f'point_index: {point_index}')
            sector_points_qdr, sector_points_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.sector_points[point_index,0],self.sector_points[point_index,1])
            # debug.red(f'point_index: {point_index}, sector_points_dis: {sector_points_dis}, sector_points_qdr: {sector_points_qdr}')
            sector_points_distance.append(sector_points_dis * RLtools.constants.NM2KM)

            drift = bs.traf.hdg[ac_idx] - sector_points_qdr

            drift = RLtools.functions.bound_angle_positive_negative_180(drift)

            sector_points_cos_drift.append(np.cos(np.deg2rad(drift)))
            sector_points_sin_drift.append(np.sin(np.deg2rad(drift)))
        
        observation = {
                "destination_waypoint_distance": np.array(destination_waypoint_distance).reshape(-1)/RLtools.constants.WAYPOINT_DISTANCE_MAX,
                "destination_waypoint_cos_drift": np.array(destination_waypoint_cos_drift).reshape(-1),
                "destination_waypoint_sin_drift": np.array(destination_waypoint_sin_drift).reshape(-1),
                # observations on obstacles
                "restricted_area_radius": np.array(self.obstacle_radius).reshape(-1)/(RLtools.constants.OBSTACLE_AREA_RANGE[0]),
                "restricted_area_distance": np.array(obstacle_centre_distance).reshape(-1)/RLtools.constants.WAYPOINT_DISTANCE_MAX,
                "cos_difference_restricted_area_pos": np.array(obstacle_centre_cos_bearing).reshape(-1),
                "sin_difference_restricted_area_pos": np.array(obstacle_centre_sin_bearing).reshape(-1),
                # observations on sector polygon edges and points along the edges
                "sector_points_distance": np.array(sector_points_distance).reshape(-1)/RLtools.constants.WAYPOINT_DISTANCE_MAX,
                "sector_points_cos_drift": np.array(sector_points_cos_drift).reshape(-1),
                "sector_points_sin_drift": np.array(sector_points_sin_drift).reshape(-1)
            }

        return observation

    def _set_action(self, action, ac_idx):
        # """
        # Centralised control for multiple aircraft
        # """
        # print(f'New action')
        dv = action[1] * RLtools.constants.D_SPEED
        dh = action[0] * RLtools.constants.D_HEADING

        id = traf.id[ac_idx]
        heading_new = RLtools.functions.bound_angle_positive_negative_180(traf.hdg[ac_idx] + dh)
        speed_new = (traf.tas[ac_idx] + dv) * RLtools.constants.MpS2Kt
        print(f'Aircraft {id} - New heading: {heading_new}, New speed: {speed_new}')
        # if not baseline_test:
        stack.stack(f"HDG {id} {heading_new}")
        stack.stack(f"SPD {id} {speed_new}")

        print(f'Action for aircraft {id} - traf.hdg: {traf.hdg[ac_idx]} -> {heading_new} with dh {dh}, traf.tas: {traf.tas[ac_idx]} m/s -> {speed_new/RLtools.constants.MpS2Kt} with dv {dv} m/s')
