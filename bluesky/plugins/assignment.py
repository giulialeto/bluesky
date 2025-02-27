"""
    ATM Assignment 2025
    Authors: Giulia Leto, Jakob Smretschnig
"""
from random import randint
import numpy as np

# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf, tools, sim  #, settings, navdb, sim, scr, tools
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
        self.polygon_id_count = 0
        self.box_id_count = 0

    # -------------------------------------------------------------------------------
    #   Periodically timed functions for ATCO workload
    # -------------------------------------------------------------------------------
    # TODO

    # -------------------------------------------------------------------------------
    #   Periodically timed functions for CSR (climate sensitive region) avoidance
    # -------------------------------------------------------------------------------
    @core.timed_function(name='create_random_CSRs', dt=7200)
    def add_CSRs(self):
        # TODO get ERA5 data for 2022/12/01 and determine actual CSRs
        create = {
            "polygon": False,
            "box": True
        }
        if create["polygon"]:
            coords = generate_random_polygon()
            coords_str = " ".join(coords.apply(lambda row: f"{row.lat},{row.lon}", axis=1))
            stack.stack(f"POLY csr{self.polygon_id_count} {coords_str}")
            stack.stack(f'COLOR csr{self.polygon_id_count} green')
            self.polygon_id_count += 1
        if create["box"]:
            coords = generate_random_rectangle()
            stack.stack(f"BOX csr{self.box_id_count} {coords[0][0]},{coords[0][1]} {coords[1][0]},{coords[1][1]}")
            stack.stack(f'COLOR csr{self.box_id_count} green')
            self.box_id_count += 1


    @core.timed_function(name='check_intersection', dt=60)
    def check_ac_intersect_csr(self):
        all_aircrafts = traf.id
        for ac_idx, ac_id in enumerate(all_aircrafts):
            ac_route = traf.ap.route[ac_idx]  # get aircraft trajectory

            # do not intervene when aircraft is heading to destination
            if ac_route.iactwp + 1 == len(ac_route.wpname):
                return

            # get one waypoint ahead of the one the aircraft is heading to, to still be able to diverge on time
            ac_coords = {
                "lat": ac_route.wplat[ac_route.iactwp + 1],
                "lon": ac_route.wplon[ac_route.iactwp + 1],
            }

            # check for intersections of that waypoint with CSRs and eventually reroute
            for shape_name, shape in dict(filter(lambda s: "CSR" in s[0], areafilter.basic_shapes.items())).items():
                if areafilter.checkInside(shape_name, ac_coords["lat"], ac_coords["lon"], 0):
                    print(f"Aircraft {ac_id} will intersect with {shape_name} soon! Trying to reroute...")
                    success = reroute_using_potential_field(ac_id, ac_route, shape, shape_name, plot=False)
                    print(f"Rerouting {'successful' if success else 'failed'} for aircraft {ac_id}.")
