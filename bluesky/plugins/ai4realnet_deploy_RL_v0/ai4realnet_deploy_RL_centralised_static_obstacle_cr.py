"""
    AI4REALNET - BlueSky plugin for deploying RL-based model
    Env: CentralisedStaticObstacleCREnv-v0
    Authors: Giulia Leto
"""
from bluesky import core, stack, traf, tools, settings 
from stable_baselines3 import SAC
import numpy as np
import bluesky.plugins.ai4realnet_deploy_RL_tools_v0.CentralisedStaticObstacleCREnv as RLtools
import bluesky as bs
import pandas as pd

algorithm = 'SAC'
env_name = 'CentralisedStaticObstacleCREnv-v0'

def init_plugin():
    CentralisedStaticObstacleCR = DeployRL()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'DEPLOY_RL_CentralisedStaticObstacleCREnv',
        # The type of this plugin.
        'plugin_type':     'sim',
        }
    return config

class DeployRL(core.Entity):  
    def __init__(self):
        super().__init__()
        self.model = SAC.load(f"bluesky/plugins/ai4realnet_deploy_RL_models/{env_name}/{env_name}_{algorithm}/model", env=None)

        # logging
        self.log_buffer = []   # temporary storage
        self.csv_file = (f"output/ai4realnet_deploy_RL_{env_name}_log.csv")

        # number of aircraft in the simulation is set by the training conditions
        # get number of aircraft in the simulation
        n_ac = len(bs.traf.id)

        # get destination waypoint of all aircraft in the simulation
        destination_list = bs.traf.ap.dest

        destination_coordinates = np.array([tuple(map(float, wpt.split(','))) for wpt in destination_list], dtype=np.float64)

        waypoint_distances = []
        # Give aircraft initial heading
        for ac_idx_aircraft in range(n_ac):
            initial_wpt_qdr, initial_wpt_dist = tools.geo.kwikqdrdist(traf.lat[ac_idx_aircraft], traf.lon[ac_idx_aircraft], destination_coordinates[ac_idx_aircraft][0], destination_coordinates[ac_idx_aircraft][1])
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

    @core.timed_function(name='CentralisedStaticObstacleCR', dt=RLtools.constants.ACTION_FREQUENCY)
    def update(self):
        # controlling all the aircraft of the scenario together (centralised). 
        # Size of the scenarios (number of aircraft) is therefore limited by the 
        # number of aircraft on which the agent was trained. 
        # Alternatively, we can control N aircraft at a time in a centralised manner, 
        # but is that any better than single agent?
        obs = self._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        self._set_action(action)

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

    def _get_obs(self):
        """
        Observation is the normalized. Normalisation logic should be studied further
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

        # wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat, bs.traf.lon, bs.traf.ap.route[:].wplat[-1], bs.traf.ap.route[:].wplon[-1])
        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(
            bs.traf.lat,
            bs.traf.lon,
            np.array([r.wplat[-1] for r in bs.traf.ap.route]),
            np.array([r.wplon[-1] for r in bs.traf.ap.route])
        )

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
            obs_centre_qdr, obs_centre_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat, bs.traf.lon, self.obstacle_center_lat[obs_idx], self.obstacle_center_lon[obs_idx])
            obs_centre_dis = obs_centre_dis * RLtools.constants.NM2KM #KM        

            bearing = bs.traf.hdg - obs_centre_qdr
            for bearing_idx in range(len(bearing)):
                bearing[bearing_idx] = RLtools.functions.bound_angle_positive_negative_180(bearing[bearing_idx])

            obstacle_centre_distance.append(obs_centre_dis)
            obstacle_centre_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            obstacle_centre_sin_bearing.append(np.sin(np.deg2rad(bearing)))
        
        observation = {
                "intruder_distance": np.array(intruder_distance).reshape(-1)/self.waypoint_distance_max,
                "intruder_cos_difference_pos": np.array(intruder_cos_bearing).reshape(-1),
                "intruder_sin_difference_pos": np.array(intruder_sin_bearing).reshape(-1),
                "intruder_x_difference_speed": np.array(intruder_x_difference_speed).reshape(-1)/RLtools.constants.AC_SPD,
                "intruder_y_difference_speed": np.array(intruder_y_difference_speed).reshape(-1)/RLtools.constants.AC_SPD,
                # observation destination waypoints
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

    def _set_action(self, action):
        """
        Centralised control for multiple aircraft
        """
        # print(f'New action')
        for i in range(len(traf.id)):
            action_index = i
            dv = action[action_index*2+1] * RLtools.constants.D_SPEED
            dh = action[action_index*2] * RLtools.constants.D_HEADING

            id = traf.id[i]
            heading_new = RLtools.functions.bound_angle_positive_negative_180(traf.hdg[i] + dh)
            speed_new = (traf.cas[i] + dv) * RLtools.constants.MpS2Kt
            # stack.stack(f"ECHO Aircraft {id} - New heading: {heading_new} deg, New speed: {speed_new/RLtools.constants.MpS2Kt} m/s")
            stack.stack(f"HDG {id} {heading_new}")
            stack.stack(f"SPD {id} {speed_new}")

            # print(f'Action for aircraft {id} - traf.hdg: {traf.hdg[ac_idx]} -> {heading_new} with dh {dh}, traf.cas: {traf.cas[ac_idx]} m/s -> {speed_new/RLtools.constants.MpS2Kt} with dv {dv} m/s')
