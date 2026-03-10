from bluesky import core, stack, traf, tools, settings 
# from stable_baselines3 import SAC
import numpy as np
import bluesky.plugins.CRTools as CRT
import bluesky.plugins.CommonTools.functions as fn
from bluesky.plugins.ai4realnet_deploy_baseline_tools_batch.constants import NM2KM, MpS2Kt
import torch

MIN_DIST_CORR = False
MIN_DIST = 45 #NM

def init_plugin():
    cr = CR()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'CR',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

class CR(core.Entity):  
    def __init__(self):
        super().__init__()
        self.model = CRT.actor.MultiHeadAdditiveActorBasic(q_dim = 7,
                                                            kv_dim = 7,
                                                            out_dim = 2,
                                                            num_heads = 5,
                                                            dropout_rate=0)
        self.model.load_state_dict(torch.load("bluesky/plugins/CRTools/weights/actor.pt"))
        self.model.set_test(True)
        with traf.settrafarrays():
            traf.target_heading = np.array([])
        self.active = False

    def reset(self):
        self.active = False

    @stack.command(name='CR')
    def set_active(self, state: str):
        """Activate or deactivate CR.

        Args:
            state (str): 'ON' or 'OFF'.
        """
        if state.upper() == 'ON':
            self.active = True
        elif state.upper() == 'OFF':
            self.active = False

    @core.timed_function(name='CR', dt=CRT.constants.TIMESTEP)
    def update(self):
        if not self.active:
            return
        if len(traf.id) > 0:
            observations = self._get_obs()
            obs_array = np.array(list(observations.values()))

            action = self.model(torch.FloatTensor(np.array([obs_array])))
            action = np.array(action[0].detach().numpy())
            act_array = np.clip(action, -1, 1)

            for id, action in zip(traf.id,act_array[0]):
                idx = traf.id2idx(id)
                if MIN_DIST_CORR:
                    _, dist = tools.geo.kwikqdrdist_matrix(traf.lat[idx],traf.lon[idx],traf.lat,traf.lon)
                    # remove itself from the distance array
                    dist = dist[dist>0]
                    # 
                    if (dist.size == 0 or min(dist) > MIN_DIST):
                        stack.stack(f"HDG {id} {traf.target_heading[idx]}")
                    else:
                        self._set_action(action,idx)
                else:
                    self._set_action(action,idx)
        else:
            pass
    
    def create(self, n=1):
        super().create(n)
        traf.target_heading[-n:] = traf.hdg[-n:]

    def _get_obs(self):
        """
        Observation is the normalized x and y coordinate of the aircraft
        """
        obs = []

        for id in traf.id:
            ac_idx = traf.id2idx(id)
            
            # Get and decompose agent aircaft drift
            ac_hdg = traf.hdg[ac_idx]
    
            idx_next_wp = traf.ap.route[ac_idx].iactwp
            target_hdg, _ = tools.geo.kwikqdrdist(traf.lat[ac_idx], traf.lon[ac_idx], traf.ap.route[ac_idx].wplat[idx_next_wp], traf.ap.route[ac_idx].wplon[idx_next_wp])
            import debug
            debug.blue(f'Current heading for {id} is {ac_hdg}')
            debug.pink(f'Target heading for {id} is {target_hdg}')
            debug.green(f'traf.lat[ac_idx]: {traf.lat[ac_idx]}, traf.lon[ac_idx]: {traf.lon[ac_idx]}')
            debug.red(f'traf.ap.route[ac_idx].wplat: {traf.ap.route[ac_idx].wplat}, traf.ap.route[ac_idx].wplon: {traf.ap.route[ac_idx].wplon}')
            debug.black(f'traf.ap.route[ac_idx].wpname[idx_next_wp]: {traf.ap.route[ac_idx].wpname[idx_next_wp]}')

            drift = ac_hdg - target_hdg
            drift = fn.bound_angle_positive_negative_180(drift)
            cos_drift = np.cos(np.deg2rad(drift))
            sin_drift = np.sin(np.deg2rad(drift))
        
            # Get agent aircraft airspeed, m/s
            airspeed_cas = traf.cas[ac_idx]
            airspeed_gs = traf.gs[ac_idx]

            # Get speedlimit information
            altitude = traf.alt[ac_idx]
            cas_min, cas_max = fn.get_speed_limits_at_altitude(altitude)

            vx = np.cos(np.deg2rad(ac_hdg)) * traf.gs[ac_idx]
            vy = np.sin(np.deg2rad(ac_hdg)) * traf.gs[ac_idx]

            # Get destination information
            ac_dest_lat = traf.ap.route[ac_idx].wplat[-1]
            ac_dest_lon = traf.ap.route[ac_idx].wplon[-1]

            ac_loc = fn.latlong_to_nm(np.array([ac_dest_lat, ac_dest_lon]), np.array([traf.lat[ac_idx], traf.lon[ac_idx]])) * NM2KM * 1000 # Two-step conversion lat/long -> NM -> m
            x = ac_loc[0]
            y = ac_loc[1]

            observation = {
                "cos(drift)": np.array([cos_drift]),
                "sin(drift)": np.array([sin_drift]),
                "airspeed_cas": np.array([(airspeed_cas-150)/50]),
                "airspeed_gs": np.array([(airspeed_gs-150)/50]),
                "altitude": np.array([(altitude-1500)/3000]),
                "cas_min": np.array([(cas_min-150)/50]),
                "cas_max": np.array([(cas_max-150)/50]),
                "x": np.array([x/50000]),
                "y": np.array([y/50000]),
                "vx": np.array([vx/150]),
                "vy": np.array([vy/150])
            }

            obs.append(np.concatenate(list(observation.values())))

        observations = {
            a: o
            for a, o in zip(traf.id, obs)
        }

        return observations
    
    def _set_action(self, action, idx):
        dh = action[0] * CRT.constants.D_HEADING
        dv = action[1] * CRT.constants.D_VELOCITY
        heading_new = fn.bound_angle_positive_negative_180(traf.hdg[idx] + dh)
        speed_new = (traf.cas[idx] + dv)
        
        # limit speed based on altitude
        altitude = traf.alt[idx]
        speed_new = fn.get_speed_at_altitude(altitude,speed_new) * MpS2Kt

        # import code
        # code.interact(local=locals())

        id = traf.id[idx]
        stack.stack(f"HDG {id} {heading_new}")
        stack.stack(f"SPD {id} {speed_new}")