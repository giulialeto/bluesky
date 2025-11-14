"""
    AI4REALNET - BlueSky plugin for deploying RL-based model
    ENV: StaticObstacleSectorCREnv-v0
    Authors: Giulia Leto
"""
from bluesky import core, stack, traf, tools, settings 
import numpy as np
import bluesky as bs
import pandas as pd
from bluesky.network.publisher import state_publisher, StatePublisher
from bluesky.plugins.ai4realnet_deploy_RL_tools import functions, constants_general

import debug

# Global data
N_AC = 20  # Number of aircraft in the randomised scenario
N_OBSTACLES = 5 

sector_name = 'LISBON_FIR'
latitude_bounds = (31.4, 43.0)
longitude_bounds = (-18.3, -6.1)

# smaller bounds for testing stage
latitude_bounds = (33.0, 36.0)
longitude_bounds = (-18.0, -12.0)

perturbation_generator = None

def init_plugin():
    global perturbation_generator
    perturbation_generator = PerturbationGenerator()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'disturbance_generator',
        # The type of this plugin.
        'plugin_type':     'sim',
        }
    
    stackfunctions = {
        'PERTURBATION': [
            'PERTURBATION [WEATHER/VOLCANIC] [ON/OFF]',
            '[txt] [onoff]',
            perturbation_generator.add_perturbations,
            'Generates a random scenario inside of Lisbon FIR',
        ]
    }

    return config, stackfunctions
    # return config

class PerturbationGenerator(core.Entity):  
    def __init__(self):
        super().__init__()
        self.weather_perturbation_flag = False
        self.volcanic_perturbation_flag = False
        
        # logging
        import debug
        debug.light_blue(f'initialised perturbations')
    
    def reset(self):
        self.weather_perturbation_flag = False
        self.volcanic_perturbation_flag = False
        # self.sample_obstacle = True           
        # stack.process('pcall ai4realnet_deploy_RL/sector.scn;initialize_scenario;DTMULT 5000')
        # stack.stack('initialize_scenario 20 5')
        # self.initialise_observation_flag = True

    # @stack.command(name ='PERTURBATION', annotations = 'PERTURBATION [WEATHER/DISTURBANCE] [ON/OFF]', aliases = ('ADD_PERTURBATION'))
    def add_perturbations(self, perturbation_type: str, flag: str = 'ON'):
        """
        Enable or disable weather or volcanic perturbations.

        Args:
            perturbation_type (str): One of 'WEATHER' or 'VOLCANIC'.
            flag (str): 'ON' to enable, 'OFF' to disable. Gets converted to boolean.
        """

        import debug
        debug.light_blue(f'Adding perturbation: {perturbation_type}, type {type(perturbation_type)} set to {flag} type {type(flag)}')
        
        if perturbation_type == 'WEATHER':
            self.weather_perturbation_flag = flag
            self.weather_active = False  # Ensure weather disturbance starts inactive
        elif perturbation_type == 'VOLCANIC':
            self.volcanic_perturbation_flag = flag
            self.volcanic_active = False  # Ensure volcanic disturbance starts inactive
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

    @core.timed_function(name='update_disturbance', dt=constants_general.ACTION_FREQUENCY)
    def update_disturbance(self):
        """
        Implements ATM 1: Weather perturbations as adverse, drifting storm cells.

        Spawning:
            - With probability pweather (each call), spawn a new disturbance if none is active.
            - Lifetime is sampled from an exponential distribution with mean qweather (Poisson).

        Evolution:
            - Radius follows a triangular profile: base -> peak -> base over lifetime.
            - Center drifts with constant (sampled) heading/speed.
            - Shape is a polygon with per-vertex jitter (stable over the cell lifetime).

        Deletion:
            - When simt >= start + lifetime, the cell is removed.
        """
        if self.weather_perturbation_flag:
            pweather = 0.03                      # spawn probability per tick
            qweather = 1200               # mean lifetime [s] for exponential
            min_r_nm = 6          # small seed radius
            peak_r_nm = 28        # peak radius
        
            # jitter   = 0.15 # per-vertex shape jitter factor, 15%
            vmin, vmax = (5.0, 100.0)  # speed range [kt]
            NM2KM = constants_general.NM2KM
            weather_cell_lifetime_min = 600  # minimum lifetime [s]
            shape_name = 'WEATHER_CELL'
            simt = bs.sim.simt
            # dt   = (simt - getattr(self, "weather_last_update_t", simt))
            # self.weather_last_update_t = simt
            dt=constants_general.ACTION_FREQUENCY

            # States
            if not hasattr(self, "weather_active"):
                self.weather_active = False
                self.weather_shape_jitter = None  # per-vertex scale factors (stable)
                self.weather_angles_deg = None

            if self.weather_active:

                if (simt - self.weather_disturbance_start) >= self.weather_cell_lifetime:
                    # Delete weather disturbance (expired)
                    stack.process(f"DELETE {shape_name}")
                    self.weather_active = False
                    self.weather_shape_jitter = None
                    self.weather_angles_deg = None
                else:
                    # Evolve the active disturbance
                    age_s   = simt - self.weather_disturbance_start
                    tau     = age_s / max(1e-6, self.weather_cell_lifetime) # [0,1] progress through lifetime
                    # Triangular growth-shrink radius profile
                    if tau <= 0.5:
                        radius_nm = min_r_nm + 2.0 * tau * (peak_r_nm - min_r_nm)
                    else:
                        radius_nm = peak_r_nm - 2.0 * (tau - 0.5) * (peak_r_nm - min_r_nm)

                    # Drift center since last update
                    if self.weather_cell_speed > 0.0:
                        # distance in NM = speed[kt]*dt[h]
                        distance_km = self.weather_cell_speed * (dt / 3600.0) * NM2KM
                        new_lat, new_lon = functions.get_point_at_distance(self.weather_cell_center_lat, self.weather_cell_center_lon, distance_km, self.weather_cell_heading)
                        self.weather_cell_center_lat = new_lat
                        self.weather_cell_center_lon = new_lon

                    
                    # Build polygon with stable jitter
                    if self.weather_shape_jitter is None:
                        # Stable per-vertex jitter and angles for the lifetime of the cell
                        jitter = 0.25
                        self.weather_angles_deg = np.linspace(0.0, 360.0, num=self.n_verts, endpoint=False)
                        self.weather_shape_jitter = 1.0 + np.random.uniform(-jitter, jitter, size=self.n_verts)
                        import debug
                        debug.pink(f'Is this ever triggered?')
                    else:
                        jitter = 0.05
                        self.weather_shape_jitter = 1.0 + np.random.uniform(-jitter, jitter, size=self.n_verts)

                    weather_cell_latitude_centre, weather_cell_longitude_centre = self.weather_cell_center_lat, self.weather_cell_center_lon
                    weather_cell_poly = []
                    weather_cell_radius = []
                    for ang_deg, jit in zip(self.weather_angles_deg, self.weather_shape_jitter):
                        radius_jittered_km = radius_nm * float(jit) * NM2KM
                        weather_cell_radius.append(radius_jittered_km)
                        vertex_lat, vertex_lon = functions.get_point_at_distance(weather_cell_latitude_centre, weather_cell_longitude_centre, radius_jittered_km, ang_deg)
                        weather_cell_poly.append((vertex_lat, vertex_lon))
                    # close polygon
                    weather_cell_poly.append(weather_cell_poly[0])
                    self.weather_cell_radius = max(weather_cell_radius)

                    # Re-draw polygon (delete + poly keeps syntax simple)
                    stack.process(f"DELETE {shape_name}")
                    flat = ", ".join([f"{lat:.6f}, {lon:.6f}" for (lat, lon) in weather_cell_poly])
                    stack.process(f"POLY {shape_name}, {flat}")
                    stack.process(f"COLOR {shape_name}, BLUE")
                    # stack.process(f"CIRCLE {shape_name}_bounding_circle, {self.weather_cell_center_lat}, {self.weather_cell_center_lon}, {self.weather_cell_radius/constants_general.NM2KM}")
                    # stack.process(f"COLOR {shape_name}_bounding_circle, YELLOW")

            else: # No active cell: sample a new one with probability pweather

                # If there is no weather distrubance active, sample with probality pweather
                weather_disturbance_flag = (np.random.rand() < pweather) and (not self.weather_active)

                if weather_disturbance_flag:
                    self.n_verts  = np.random.randint(10, 16)            # polygon vertex count
                    # Sample lifetime ~ Exponential(mean=qweather)
                    # (Poisson process waiting time)
                    weather_cell_lifetime = float(np.random.exponential(scale=max(qweather, 1e-6)))
                    # Minimum lifetime guard
                    weather_cell_lifetime = max(weather_cell_lifetime, weather_cell_lifetime_min)

                    # Random center within sector bounds
                    weather_cell_latitude_centre, weather_cell_longitude_centre = np.random.uniform(latitude_bounds[0], latitude_bounds[1]), np.random.uniform(longitude_bounds[0], longitude_bounds[1])

                    # Sample stable drift and shape
                    self.weather_cell_heading = int(np.random.randint(0, 360)) # deg
                    self.weather_cell_speed = float(np.random.uniform(vmin, vmax)) # kts

                    # Jitter the vertices of each weather cell
                    jitter = 0.25
                    self.weather_angles_deg = np.linspace(0.0, 360.0, num=self.n_verts, endpoint=False)
                    self.weather_shape_jitter = 1.0 + np.random.uniform(-jitter, jitter, size=self.n_verts)

                    # Start the weather disturbance with a small radius
                    seed_radius_nm = float(np.random.uniform(0.5 * min_r_nm, min_r_nm))

                    # Draw initial polygon
                    weather_cell_poly = []
                    weather_cell_radius = []
                    for ang_deg, jit in zip(self.weather_angles_deg, self.weather_shape_jitter):
                        radius_jittered_km = seed_radius_nm * float(jit) * NM2KM
                        weather_cell_radius.append(radius_jittered_km)
                        vertex_lat, vertex_lon = functions.get_point_at_distance(weather_cell_latitude_centre, weather_cell_longitude_centre, radius_jittered_km, ang_deg)
                        weather_cell_poly.append((vertex_lat, vertex_lon))
                    # close polygon
                    weather_cell_poly.append(weather_cell_poly[0])

                    flattened_polygon = ", ".join([f"{lat:.6f}, {lon:.6f}" for (lat, lon) in weather_cell_poly])
                    stack.process(f"POLY {shape_name}, {flattened_polygon}")
                    stack.process(f"COLOR {shape_name}, BLUE")

                    # Persist state
                    self.weather_active = True
                    self.weather_cell_center_lat = float(weather_cell_latitude_centre)
                    self.weather_cell_center_lon = float(weather_cell_longitude_centre)
                    self.weather_disturbance_start = float(simt)
                    self.weather_cell_lifetime = float(weather_cell_lifetime)
                    self.weather_cell_radius = max(weather_cell_radius)
                    # stack.process(f"CIRCLE {shape_name}_bounding_circle, {self.weather_cell_center_lat}, {self.weather_cell_center_lon}, {self.weather_cell_radius/constants_general.NM2KM}")
                    # stack.process(f"COLOR {shape_name}_bounding_circle, YELLOW")
        else:
            if hasattr(self, "weather_active"):
                # Remove any active weather disturbance if the flag is turned off
                stack.process(f"DELETE WEATHER_CELL")
                self.weather_active = False
                self.weather_shape_jitter = None

        if self.volcanic_perturbation_flag:
            pvolcanic = 0.0005                      # spawn probability per tick
            qvolcanic = 1200               # mean lifetime [s] for exponential
            min_r_nm = 6          # small seed radius
            peak_r_nm = 80        # peak radius
        
            # jitter   = 0.15 # per-vertex shape jitter factor, 15%
            # vmin, vmax = (5.0, 100.0)  # speed range [kt]
            NM2KM = constants_general.NM2KM
            volcanic_cell_lifetime_min = 12000  # minimum lifetime [s]
            shape_name = 'VOLCANIC_CELL'
            simt = bs.sim.simt
            # dt   = (simt - getattr(self, "weather_last_update_t", simt))
            # self.weather_last_update_t = simt
            dt=constants_general.ACTION_FREQUENCY

            # States
            if not hasattr(self, "volcanic_active"):
                self.volcanic_active = False
                self.volcanic_shape_jitter = None  # per-vertex scale factors (stable)
                self.volcanic_angles_deg = None

            if self.volcanic_active:

                if (simt - self.volcanic_disturbance_start) >= self.volcanic_cell_lifetime:
                    # Delete weather disturbance (expired)
                    stack.process(f"DELETE {shape_name}")
                    self.volcanic_active = False
                    self.volcanic_shape_jitter = None
                    self.volcanic_angles_deg = None
                else:
                    # Evolve the active disturbance
                    age_s   = simt - self.volcanic_disturbance_start
                    tau     = age_s / max(1e-6, self.volcanic_cell_lifetime) # [0,1] progress through lifetime
                    # Triangular growth-shrink radius profile
                    if tau <= 0.5:
                        radius_nm = min_r_nm + 2.0 * tau * (peak_r_nm - min_r_nm)
                    else:
                        radius_nm = peak_r_nm - 2.0 * (tau - 0.5) * (peak_r_nm - min_r_nm)
                    
                    # Build polygon with stable jitter
                    if self.volcanic_shape_jitter is None:
                        # Stable per-vertex jitter and angles for the lifetime of the cell
                        jitter = 0.1
                        self.volcanic_angles_deg = np.linspace(0.0, 360.0, num=self.volcanic_n_verts, endpoint=False)
                        self.volcanic_shape_jitter = 1.0 + np.random.uniform(-jitter, jitter, size=self.volcanic_n_verts)
                        import debug
                        debug.pink(f'Is this ever triggered?')
                    else:
                        jitter = 0.05
                        self.volcanic_shape_jitter = 1.0 + np.random.uniform(-jitter, jitter, size=self.volcanic_n_verts)

                    volcanic_cell_latitude_centre, volcanic_cell_longitude_centre = self.volcanic_cell_center_lat, self.volcanic_cell_center_lon
                    volcanic_cell_poly = []
                    volcanic_cell_radius = []
                    for ang_deg, jit in zip(self.volcanic_angles_deg, self.volcanic_shape_jitter):
                        radius_jittered_km = radius_nm * float(jit) * NM2KM
                        volcanic_cell_radius.append(radius_jittered_km)
                        vertex_lat, vertex_lon = functions.get_point_at_distance(volcanic_cell_latitude_centre, volcanic_cell_longitude_centre, radius_jittered_km, ang_deg)
                        volcanic_cell_poly.append((vertex_lat, vertex_lon))
                    # close polygon
                    volcanic_cell_poly.append(volcanic_cell_poly[0])
                    self.volcanic_cell_radius = max(volcanic_cell_radius)

                    # Re-draw polygon (delete + poly keeps syntax simple)
                    stack.process(f"DELETE {shape_name}")
                    flat = ", ".join([f"{lat:.6f}, {lon:.6f}" for (lat, lon) in volcanic_cell_poly])
                    stack.process(f"POLY {shape_name}, {flat}")
                    stack.process(f"COLOR {shape_name}, GREEN")
                    # stack.process(f"CIRCLE {shape_name}_bounding_circle, {self.weather_cell_center_lat}, {self.weather_cell_center_lon}, {self.weather_cell_radius/constants_general.NM2KM}")
                    # stack.process(f"COLOR {shape_name}_bounding_circle, YELLOW")

            else: # No active cell: sample a new one with probability pweather

                # If there is no weather distrubance active, sample with probality pweather
                volcanic_disturbance_flag = (np.random.rand() < pvolcanic) and (not self.volcanic_active)

                if volcanic_disturbance_flag:
                    self.volcanic_n_verts  = np.random.randint(10, 16)            # polygon vertex count
                    # Sample lifetime ~ Exponential(mean=qweather)
                    # (Poisson process waiting time)
                    volcanic_cell_lifetime = float(np.random.exponential(scale=max(qvolcanic, 1e-6)))
                    # Minimum lifetime guard
                    volcanic_cell_lifetime = max(volcanic_cell_lifetime, volcanic_cell_lifetime_min)

                    # Random center within sector bounds
                    volcanic_cell_latitude_centre, volcanic_cell_longitude_centre = np.random.uniform(latitude_bounds[0], latitude_bounds[1]), np.random.uniform(longitude_bounds[0], longitude_bounds[1])

                    # Jitter the vertices of each weather cell
                    jitter = 0.25
                    self.volcanic_angles_deg = np.linspace(0.0, 360.0, num=self.volcanic_n_verts, endpoint=False)
                    self.volcanic_shape_jitter = 1.0 + np.random.uniform(-jitter, jitter, size=self.volcanic_n_verts)

                    # Start the weather disturbance with a small radius
                    seed_radius_nm = float(np.random.uniform(0.5 * min_r_nm, min_r_nm))

                    # Draw initial polygon
                    volcanic_cell_poly = []
                    volcanic_cell_radius = []
                    for ang_deg, jit in zip(self.volcanic_angles_deg, self.volcanic_shape_jitter):
                        radius_jittered_km = seed_radius_nm * float(jit) * NM2KM
                        volcanic_cell_radius.append(radius_jittered_km)
                        vertex_lat, vertex_lon = functions.get_point_at_distance(volcanic_cell_latitude_centre, volcanic_cell_longitude_centre, radius_jittered_km, ang_deg)
                        volcanic_cell_poly.append((vertex_lat, vertex_lon))
                    # close polygon
                    volcanic_cell_poly.append(volcanic_cell_poly[0])

                    flattened_polygon = ", ".join([f"{lat:.6f}, {lon:.6f}" for (lat, lon) in volcanic_cell_poly])
                    stack.process(f"POLY {shape_name}, {flattened_polygon}")
                    stack.process(f"COLOR {shape_name}, GREEN")

                    # Persist state
                    self.volcanic_active = True
                    self.volcanic_cell_center_lat = float(volcanic_cell_latitude_centre)
                    self.volcanic_cell_center_lon = float(volcanic_cell_longitude_centre)
                    self.volcanic_disturbance_start = float(simt)
                    self.volcanic_cell_lifetime = float(volcanic_cell_lifetime)
                    self.volcanic_cell_radius = max(volcanic_cell_radius)
                    # stack.process(f"CIRCLE {shape_name}_bounding_circle, {self.weather_cell_center_lat}, {self.weather_cell_center_lon}, {self.weather_cell_radius/constants_general.NM2KM}")
                    # stack.process(f"COLOR {shape_name}_bounding_circle, YELLOW")
        else:
            if hasattr(self, "volcanic_active"):
                # Remove any active weather disturbance if the flag is turned off
                stack.process(f"DELETE VOLCANIC_CELL")
                self.volcanic_active = False
                self.volcanic_shape_jitter = None
