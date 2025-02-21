"""
    ATM Assignment 2025
    Authors: Giulia Leto, Jakob Smretschnig
"""
from random import randint
import numpy as np

# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf, tools  #, settings, navdb, sim, scr, tools
from bluesky.tools import areafilter

# Import assignment_src dependencies
from bluesky.plugins.assignment_src.atco_workload import *
from bluesky.plugins.assignment_src.csr_avoidance import *

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our assignment_src entity
    assignment = Assignment()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'ASSIGNMENT',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
    }

    # init_plugin() should always return a configuration dict.
    return config


class Assignment(core.Entity):
    ''' Example new entity object for BlueSky. '''
    def __init__(self):
        super().__init__()

    # -------------------------------------------------------------------------------
    #   Periodically timed functions for ATCO workload
    # -------------------------------------------------------------------------------
    # TODO

    # -------------------------------------------------------------------------------
    #   Periodically timed functions for CSR avoidance
    # -------------------------------------------------------------------------------
    @core.timed_function(name='create_random_CSRs', dt=10000)
    def add_CSRs(self):
        poly_id = randint(1, 1000)
        coords = generate_random_polygon()
        coords_str = " ".join(coords.apply(lambda row: f"{row.lat},{row.lon}", axis=1))
        stack.stack(f"POLY csr{poly_id} {coords_str}")
        stack.stack(f'COLOR csr{poly_id} green')

    @core.timed_function(name='check_intersection', dt=1000)
    def check_ac_intersect_csr(self):
        # Check if the aircraft with flight_id 257379367 is within any polygon
        demo_id = "257379367"
        ac_idx = traf.id2idx(demo_id)
        if ac_idx != -1:
            ac_route = traf.ap.route[ac_idx]
            ac_coords = {
                "lat": ac_route.wplat[ac_route.iactwp],
                "lon": ac_route.wplon[ac_route.iactwp],
            }
            for shape_name, shape in areafilter.basic_shapes.items():
                if areafilter.checkInside(shape_name, ac_coords["lat"], ac_coords["lon"], 0):
                    print(f"Aircraft {demo_id} intersects with {shape_name}")
