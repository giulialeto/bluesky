"""
    AI4REALNET - BlueSky plugin for deploying RL-based model
    ENV: StaticObstacleCREnv-v0
    Authors: Giulia Leto
"""
from bluesky import core, stack, traf, tools, settings 
from stable_baselines3 import SAC
import numpy as np
import bluesky.plugins.ai4realnet_deploy_RL_tools.StaticObstacleCREnv as RLtools
import bluesky as bs

algorithm = 'SAC'
env_name = 'StaticObstacleCREnv-v0'

def init_plugin():
    StaticObstacleCR = DeployRL()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'DEPLOY_RL_StaticObstacleCREnv',
        # The type of this plugin.
        'plugin_type':     'sim',
        }
    return config

class DeployRL(core.Entity):  
    def __init__(self):
        super().__init__()
        self.model = SAC.load(f"bluesky/plugins/ai4realnet_deploy_RL_tools/models/{env_name}/{env_name}_{algorithm}/model", env=None)

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
        for shape_name, shape in tools.areafilter.basic_shapes.items():
            # print(f'Restricted area name: {shape_name}')
            if shape_name != 'LISBON_FIR':
                coordinates = shape.coordinates
                latitudes = coordinates[::2]
                longitudes = coordinates[1::2]
                (center_latlon, radius_m) = RLtools.functions.bounding_circle_geodesic(list(zip(latitudes, longitudes)))
                self.obstacle_center_lat.append(center_latlon[0])
                self.obstacle_center_lon.append(center_latlon[1])
                self.obstacle_radius.append(radius_m)
                
                # Add circles around the restricted area for visualization of the observation
                # print(f'Obstacle {shape_name} center: {center_latlon}, radius: {radius_m}')
                # stack.stack(f"CIRCLE {shape_name}_bounding_circle, {center_latlon[0]}, {center_latlon[1]}, {radius_m/RLtools.constants.NM2KM/1000}")
                # stack.stack(f"COLOR {shape_name}_bounding_circle, YELLOW")
        
        # Scaling factor for the radius in the observation vector
        self.max_obstacle_radius = max(self.obstacle_radius)

    @core.timed_function(name='StaticObstacleCR', dt=RLtools.constants.ACTION_FREQUENCY)
    def update(self):
        # controlling each aircraft separately
        for id in traf.id:
            ac_idx = traf.id2idx(id)
            obs = self._get_obs(ac_idx)
            action, _ = self.model.predict(obs, deterministic=True)
            self._set_action(action, ac_idx)

    def _get_obs(self, ac_idx):
        """
        Observation is the normalized. Normalisation logic should be studied further
        """
        # intruder observation
        intruders_lat = np.delete(bs.traf.lat, ac_idx)
        intruders_lon = np.delete(bs.traf.lon, ac_idx)
        int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], intruders_lat, intruders_lon)

        # index of the closes N_INTRUDERS intruders for arrays of intruders_lat, intruders_lon, int_qdr, int_dis, intruders_speed, intruders_heading: the ownship is excluded a priori in these
        closest_intruders_idx = np.argsort(int_dis)[:RLtools.constants.NUM_INTRUDERS]

        intruder_distance = int_dis[closest_intruders_idx] * RLtools.constants.NM2KM

        bearing = bs.traf.hdg[ac_idx] - int_qdr[closest_intruders_idx]
        for bearing_idx in range(len(bearing)):
            bearing[bearing_idx] = RLtools.functions.bound_angle_positive_negative_180(bearing[bearing_idx])

        intruder_cos_bearing = np.cos(np.deg2rad(bearing))
        intruder_sin_bearing = np.sin(np.deg2rad(bearing))

        intruders_heading = np.delete(bs.traf.hdg, ac_idx)
        intruders_speed = np.delete(bs.traf.gs, ac_idx)
        heading_difference = bs.traf.hdg[ac_idx] - intruders_heading[closest_intruders_idx]
        intruder_x_difference_speed = - np.cos(np.deg2rad(heading_difference)) * intruders_speed[closest_intruders_idx]
        intruder_y_difference_speed = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * intruders_speed[closest_intruders_idx]

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
            # print(f'obs_idx: {obs_idx}, obs_centre_dis: {obs_centre_dis}, obs_centre_qdr: {obs_centre_qdr}')
            bearing = bs.traf.hdg[ac_idx] - obs_centre_qdr
            
            bearing = RLtools.functions.bound_angle_positive_negative_180(bearing)

            obstacle_centre_distance.append(obs_centre_dis)
            obstacle_centre_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            obstacle_centre_sin_bearing.append(np.sin(np.deg2rad(bearing)))
        
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
        dv = action[1] * RLtools.constants.D_SPEED
        dh = action[0] * RLtools.constants.D_HEADING

        id = traf.id[ac_idx]
        heading_new = RLtools.functions.bound_angle_positive_negative_180(traf.hdg[ac_idx] + dh)
        speed_new = (traf.cas[ac_idx] + dv) * RLtools.constants.MpS2Kt
        # stack.stack(f"ECHO Aircraft {id} - New heading: {heading_new} deg, New speed: {speed_new/RLtools.constants.MpS2Kt} m/s")
        stack.stack(f"HDG {id} {heading_new}")
        stack.stack(f"SPD {id} {speed_new}")

        # print(f'Action for aircraft {id} - traf.hdg: {traf.hdg[ac_idx]} -> {heading_new} with dh {dh}, traf.cas: {traf.cas[ac_idx]} m/s -> {speed_new/RLtools.constants.MpS2Kt} with dv {dv} m/s')
