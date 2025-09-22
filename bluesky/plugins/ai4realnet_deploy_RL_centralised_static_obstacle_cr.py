"""
    AI4REALNET - BlueSky plugin for deploying RL-based model
    Authors: Giulia Leto
"""
from bluesky import core, stack, traf, tools, settings 
from stable_baselines3 import SAC
import numpy as np
import bluesky.plugins.ai4realnet_deploy_RL_tools.CentralisedStaticObstacleCREnv as RLtools
import bluesky as bs

# TODO make this such that you can select the algorithm in settings
# settings.set_variable_defaults(algorithm='SAC')
algorithm = 'SAC'
env_name = 'CentralisedStaticObstacleCREnv'
def init_plugin():
    CentralisedStaticObstacleCR = DeployRL()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'DEPLOY_RL_CentralisedStaticObstacleCREnv',
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

    @core.timed_function(name='CentralisedStaticObstacleSectorCR', dt=RLtools.constants.ACTION_FREQUENCY)
    def update(self):
        # for id in traf.id:
        #     idx = traf.id2idx(id)
        #     obs = self._get_obs(idx)
        #     clipped_obs = {key: np.clip(arr, -1.2, 1.2) for key, arr in obs.items()}
        #     action, _ = self.model.predict(clipped_obs, deterministic=True)
        #     self._set_action(action,idx)
        # pass
        obs = self._get_obs()
    #     clipped_obs = {key: np.clip(arr, -1.2, 1.2) for key, arr in obs.items()}
        action, _ = self.model.predict(obs, deterministic=True)
        self._set_action(action)

    def _get_obs(self):
        """
        Observation is the normalized x and y coordinate of the aircraft
        """

        # intruder observation
        intruder_distance = []
        intruder_cos_bearing = []
        intruder_sin_bearing = []
        intruder_x_difference_speed = []
        intruder_y_difference_speed = []

        for i in range(len(traf.id)):
            intruders_lat = np.delete(bs.traf.lat, i)
            intruders_lon = np.delete(bs.traf.lon, i)
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[i], bs.traf.lon[i], intruders_lat, intruders_lon)
            
            intruder_distance.append(int_dis * RLtools.constants.NM2KM)

            bearing = bs.traf.hdg[i] - int_qdr
            for bearing_idx in range(len(bearing)):
                bearing[bearing_idx] = RLtools.functions.bound_angle_positive_negative_180(bearing[bearing_idx])

            intruder_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            intruder_sin_bearing.append(np.sin(np.deg2rad(bearing)))

            intruders_heading = np.delete(bs.traf.hdg, i)
            intruders_speed = np.delete(bs.traf.gs, i)
            heading_difference = bs.traf.hdg[i] - intruders_heading
            x_dif = - np.cos(np.deg2rad(heading_difference)) * intruders_speed
            y_dif = bs.traf.gs[i] - np.sin(np.deg2rad(heading_difference)) * intruders_speed
            intruder_x_difference_speed.append(x_dif)
            intruder_y_difference_speed.append(y_dif)

        # destination waypoint
        destination_waypoint_distance = []
        destination_waypoint_cos_drift = []
        destination_waypoint_sin_drift = []
        destination_waypoint_drift = []

        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat, bs.traf.lon, self.destination_coordinates[:, 0], self.destination_coordinates[:, 1])
    
        destination_waypoint_distance.append(wpt_dis * RLtools.constants.NM2KM)

        drift = bs.traf.hdg - wpt_qdr
        for drift_idx in range(len(drift)):
            drift[drift_idx] = RLtools.functions.bound_angle_positive_negative_180(drift[drift_idx])

        destination_waypoint_drift.append(drift)
        destination_waypoint_cos_drift.append(np.cos(np.deg2rad(drift)))
        destination_waypoint_sin_drift.append(np.sin(np.deg2rad(drift)))

        # obstacles 
        obstacle_centre_distance = []
        obstacle_centre_cos_bearing = []
        obstacle_centre_sin_bearing = []

        for obs_idx in range(RLtools.constants.NUM_OBSTACLES):
            # print(f'obs_idx: {obs_idx}')
            obs_centre_qdr, obs_centre_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat, bs.traf.lon, self.obstacle_center_lat[obs_idx], self.obstacle_center_lon[obs_idx])
            obs_centre_dis = obs_centre_dis * RLtools.constants.NM2KM #KM        

            bearing = bs.traf.hdg - obs_centre_qdr
            # bearing = fn.bound_angle_positive_negative_180(bearing)
            for bearing_idx in range(len(bearing)):
                bearing[bearing_idx] = RLtools.functions.bound_angle_positive_negative_180(bearing[bearing_idx])

            obstacle_centre_distance.append(obs_centre_dis)
            obstacle_centre_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            obstacle_centre_sin_bearing.append(np.sin(np.deg2rad(bearing)))
        
        observation = {
                "intruder_distance": np.array(intruder_distance).reshape(-1)/RLtools.constants.WAYPOINT_DISTANCE_MAX,
                "intruder_cos_difference_pos": np.array(intruder_cos_bearing).reshape(-1),
                "intruder_sin_difference_pos": np.array(intruder_sin_bearing).reshape(-1),
                "intruder_x_difference_speed": np.array(intruder_x_difference_speed).reshape(-1)/RLtools.constants.AC_SPD,
                "intruder_y_difference_speed": np.array(intruder_y_difference_speed).reshape(-1)/RLtools.constants.AC_SPD,
                "destination_waypoint_distance": np.array(destination_waypoint_distance).reshape(-1)/RLtools.constants.WAYPOINT_DISTANCE_MAX,
                "destination_waypoint_cos_drift": np.array(destination_waypoint_cos_drift).reshape(-1),
                "destination_waypoint_sin_drift": np.array(destination_waypoint_sin_drift).reshape(-1),
                # observations on obstacles
                "restricted_area_radius": np.array(self.obstacle_radius).reshape(-1)/(RLtools.constants.OBSTACLE_AREA_RANGE[0]),
                "restricted_area_distance": np.array(obstacle_centre_distance).reshape(-1)/RLtools.constants.WAYPOINT_DISTANCE_MAX,
                "cos_difference_restricted_area_pos": np.array(obstacle_centre_cos_bearing).reshape(-1),
                "sin_difference_restricted_area_pos": np.array(obstacle_centre_sin_bearing).reshape(-1),
            }

        return observation

    def _set_action(self, action):
        # """
        # Centralised control for multiple aircraft
        # """
        print(f'New action')
        for i in range(len(traf.id)):
            action_index = i
            dv = action[action_index*2+1] * RLtools.constants.D_SPEED
            dh = action[action_index*2] * RLtools.constants.D_HEADING

            id = traf.id[i]
            heading_new = RLtools.functions.bound_angle_positive_negative_180(traf.hdg[i] + dh)
            speed_new = (traf.tas[i] + dv) * RLtools.constants.MpS2Kt
            print(f'dv: {dv}, dh: {dh}')
            print(f'Aircraft {id} - New heading: {heading_new}, New speed: {speed_new}')
            # if not baseline_test:
            stack.stack(f"HDG {id} {heading_new}")
            stack.stack(f"SPD {id} {speed_new}")
