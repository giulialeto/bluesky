"""
    AI4REALNET -  Scripted automation (workload and re-routing) plugin for BlueSky
    Authors: Giulia Leto, Jakob Smretschnig
"""
from random import randint
import numpy as np

# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf, tools, sim  #, settings, navdb, scr, tools
from bluesky.tools import areafilter

# Import dependencies
from bluesky.plugins.ai4realnet_scripted_automation_src.atco_workload import *
from bluesky.plugins.ai4realnet_scripted_automation_src.csr_avoidance import *
from bluesky.plugins.ai4realnet_scripted_automation_src.scripted_automation_sectors import *

# Global variables
workload_evaluation_dt = 15*60
sector_list = ['N', 'C', 'S', 'D', 'V', 'M']
feasible_sector_combinations = [
    'D', 'N', 'C', 'S', 'V', 'M', 'DN', 'DV', 'NC', 'CS', 'CV', 'SV', 'VM', 
    'DNC', 'DNV', 'DCV', 'DSV', 'DVM', 'NCS', 'NCV', 'CSV', 'CVM', 'SVM', 
    'DNCS', 'DNCV', 'DNSV', 'DNVM', 'DCSV', 'DCVM', 'DSVM', 'NCSV', 'NCVM', 
    'CSVM', 'DNCSV', 'DNCVM', 'DNSVM', 'DCSVM', 'NCSVM', 'DNCSVM'
]
coloring = {
    "sector": "white",
    "CSR": "0,255,0"
}

#--------------------------------------------------------------------------------
#  All sector combinations coordinated (LISBON FIR)
#--------------------------------------------------------------------------------

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our scripted_automation entity
    ai4realnet_scripted_automation = ScriptedAutomation()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'AI4REALNET_SCRIPTED_AUTOMATION',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
    }
    
    # init_plugin() should always return a configuration dict.
    return config

class ScriptedAutomation(core.Entity):
    ''' Example new entity object for BlueSky. '''
    def __init__(self):
        super().__init__()
        # Variables for ATCO

        # Variables for CSR
        self.polygon_id_count = 0
        self.box_id_count = 0
        self.create_random_CSRs = False
        self.reroute_around_CSRs = True
        self.plot_potential_fields = False
        self.max_amount_sectors = 3  # Depends on the availability of ATCos
        self.max_aircraft_allowed = 10  # Soft constraint for the aircraft
        self.sectors = pd.DataFrame(columns=[f"sector_{i}" for i in range(1, self.max_amount_sectors + 1)] + ["from", "to"])
        self.sectors_occupancy = pd.DataFrame(columns=[f"sector_{i}" for i in range(1, self.max_amount_sectors + 1)] + [f"ac_count_sector_{i}" for i in range(1, self.max_amount_sectors + 1)] + ["UTC"])

        # Init hardcoded obstacles for DEMO
        # POLYGONS
        # stack.stack("POLY CSR_POLY_HC1 42.373533,-9.207331 41.574781,-9.787984 41.167417,-8.267728 41.295217,-5.828984 42.261707,-5.955672 42.525296,-7.792648")
        # stack.stack(f'COLOR CSR_POLY_HC1 {coloring["CSR"]}')
        #
        # stack.stack("POLY CSR_POLY_HC2 39.998794,-9.69147 39.109431,-9.940113 39.219455,-8.081215 39.485347,-6.115756 40.062975,-6.103916 39.952951,-7.939134 40.145493,-9.229706")
        # stack.stack(f'COLOR CSR_POLY_HC2 {coloring["CSR"]}')
        #
        # stack.stack("POLY CSR_POLY_HC3 46.069824,-3.833122 44.847438,-5.102157 43.943935,-3.350888 44.067946,-0.53363 45.467489,0.253172 46.441854,-1.625")
        # stack.stack(f'COLOR CSR_POLY_HC3 {coloring["CSR"]}')

        # BOXES
        # stack.stack("BOX CSR_BOX_HC1 46.419242,-4.647985 44.151525,-0.072488")
        # stack.stack(f'COLOR CSR_BOX_HC1 {coloring["CSR"]}')
        #
        # stack.stack("BOX CSR_BOX_HC2 42.326925,-9.727867 41.710363,-5.836893")
        # stack.stack(f'COLOR CSR_BOX_HC2 {coloring["CSR"]}')
        #
        # stack.stack("BOX CSR_BOX_HC3 40.033143,-9.83595 39.459697,-6.593472")
        # stack.stack(f'COLOR CSR_BOX_HC3 {coloring["CSR"]}')

    # -------------------------------------------------------------------------------
    #   Stack commands for sector opening plans based on ATCO workload
    # -------------------------------------------------------------------------------

    @stack.command(name="AVAILABLE_ATCO")
    def change_available_ATCO(self, available_atco: int):
        self.max_amount_sectors = available_atco

    @stack.command(name="ALLOWED_AIRCRAFT")
    def change_allowed_aircraft(self, allowed_aircraft: int):
        self.max_aircraft_allowed = allowed_aircraft

    # -------------------------------------------------------------------------------
    #   Periodically timed functions for ATCO workload
    # -------------------------------------------------------------------------------
    @core.timed_function(name='sector_opening', dt=workload_evaluation_dt)
    def sector_count(self):
        # Get current location of all aircraft
        current_location_df = get_current_location(traf)
        # Get the next waypoint of all aircraft
        future_waypoints_df = get_future_location(traf, sim.simt, workload_evaluation_dt)
        
        
        # Get the sector count based on current location and future location of the aircraft
        sector_count = get_sector_count(sector_list, current_location_df, future_waypoints_df)
        # TODO: swap this function with some other metric, resulting in a df numbers assigned to each sector, with higher number representing higher complexity. For example, dynamic density metric

        # as long as no aircraft have been spawned and there are no aircraft in the airspace, go for FIR (else statement)
        if not sector_count.empty and sector_count.sum() != 0:
            # type(sector_count)
            # Get feasible sector combinations (depends only on the available ATCOs, encoded in max_amount_sectors)
            # TODO update live during the simulation the max_amount_sectors through stack commands
            feasible_sector_combinations_with_ATCO_available = get_feasible_sector_combinations_with_ATCO_available(feasible_sector_combinations, self.max_amount_sectors)
            # Get the number of aircraft in each feasible sector combination
            sector_count_feasible_sector_combinations_with_ATCO_available, ac_count_columns = sector_count_in_feasible_combinations(feasible_sector_combinations_with_ATCO_available, sector_count)
            # Select the best grouping of sectors
            selected_sectors = select_best_grouping(sector_count_feasible_sector_combinations_with_ATCO_available, ac_count_columns, self.max_aircraft_allowed)
            # yellow(f'sector_count_feasible_sector_combinations_with_ATCO_available {sector_count_feasible_sector_combinations_with_ATCO_available}')
            # green(f'select_best_grouping {selected_sectors}')
            stack.stack(f"ECHO {' '.join(selected_sectors.columns)}")
            stack.stack(f"ECHO {' '.join(selected_sectors.iloc[0].astype(str))}") 
            self.sectors_occupancy = pd.concat([self.sectors_occupancy, selected_sectors.iloc[0].to_frame().T], ignore_index=True)
            self.sectors_occupancy.iloc[-1, self.sectors_occupancy.columns.get_loc('UTC')] = sim.utc
            print(self.sectors_occupancy)

            for r in range(1, self.max_amount_sectors + 1):
                selected_sectors.drop(columns=f'ac_count_sector_{r}', inplace=True)

            # using a stack command, create POLY and color them
            def _plot_sectors():
                for index, row in selected_sectors.iterrows():
                    for sector in row.dropna():  # Drop NaN values to only process valid sector names
                        stack_name = f"{sector}_stack"
                        if stack_name in globals():  # Check if the variable exists
                            stack.stack(globals()[
                                            stack_name])  # Pass the actual variable content to stack.stack()
                        # color the active sectors
                        stack.stack(f"COLOR {sector} {coloring['sector']}")

            # if self.sectors.empty:
            #     # add the FIR
            #     _plot_sectors()
            #     # store sector history in a pd.DataFrame
            #     self.sectors = pd.concat([self.sectors, selected_sectors.iloc[0].to_frame().T], ignore_index=True)
            #     self.sectors.iloc[-1, self.sectors.columns.get_loc('from')] = sim.simt
            # else:
            if not self.sectors.empty:
                # check if the last sector setting is the same as the current one
                last_setting = self.sectors.iloc[-1][self.sectors.filter(regex="sector", axis=1).columns].to_dict()
                new_setting = selected_sectors.iloc[0].to_dict()
                if new_setting != last_setting:
                    # Re-set active sectors:
                    # Delete all sectors which are not in the basic sector list or CSRs, color the basic sectors black
                    for area in dict(filter(lambda s: "CSR" not in s[0], areafilter.basic_shapes.items())):
                        if area not in sector_list:
                            stack.stack(f"DEL {area}")
                        else:
                            stack.stack(f"COLOR {area} black")
                    # plot the new sectors
                    _plot_sectors()

                    # store sector history in a pd.DataFrame
                    self.sectors = pd.concat([self.sectors, selected_sectors.iloc[0].to_frame().T], ignore_index=True)
                    self.sectors.iloc[-2, self.sectors.columns.get_loc('to')] = sim.simt
                    self.sectors.iloc[-1, self.sectors.columns.get_loc('from')] = sim.simt
                    
                    # # Print header
                    # stack.stack(f"ECHO {' '.join(self.sectors.columns)}")

                    # # Print each row on a separate line
                    # for _, row in self.sectors.iterrows():
                    #     stack.stack(f"ECHO {' '.join(map(str, row.values))}")


            print(self.sectors)
        else: # empty traffic object, go for FIR
            header = []
            sector_combinations = []

            header = [f"sector_{r}" for r in range(1, self.max_amount_sectors + 1)]
            
            sector_combinations = ['DNCSVM'] + [None] * (self.max_amount_sectors - 1)

            selected_combinations_df = pd.DataFrame([sector_combinations], columns=header)

            # print in console the chosen sector and the amount of aircraft in each sector
            # header_selected_sectors.extend([f"ac_count_sector_{r}" for r in range(1, self.max_amount_sectors + 1)])
            print(header)
            header_selected_sectors = header + [f"ac_count_sector_{r}" for r in range(1, self.max_amount_sectors + 1)]

            sector_combinations_count = ['0'] * self.max_amount_sectors
            selected_sectors = sector_combinations + sector_combinations_count

            selected_sectors = pd.DataFrame([selected_sectors], columns=header_selected_sectors)


            if "DNCSVM" not in areafilter.basic_shapes.keys():
                stack.stack(globals()["DNCSVM_stack"])
                stack.stack(f"COLOR DNCSVM {coloring['sector']}")
                # store sector history in a pd.DataFrame
                self.sectors = pd.concat([self.sectors, selected_combinations_df.iloc[0].to_frame().T], ignore_index=True)
                self.sectors.iloc[-1, self.sectors.columns.get_loc('from')] = 0
                                
                # Print in console
                stack.stack(f"ECHO {' '.join(selected_sectors.columns)}")
                stack.stack(f"ECHO {' '.join(selected_sectors.iloc[0].astype(str))}")

            self.sectors_occupancy = pd.concat([self.sectors_occupancy, selected_sectors.iloc[0].to_frame().T], ignore_index=True)
            self.sectors_occupancy.iloc[-1, self.sectors_occupancy.columns.get_loc('UTC')] = sim.utc
            print(self.sectors_occupancy)
            print(sim.utc)
    # -------------------------------------------------------------------------------
    #   Stack commands for CSR (climate sensitive region) avoidance
    # -------------------------------------------------------------------------------
    @stack.command(name="CREATE_CSR")
    def create_CSRs(self, enable: "bool"):
        print(f"CSR Creation: {enable}")
        self.create_random_CSRs = enable


    @stack.command(name="AVOID_CSR")
    def avoid_CSRs(self, enable: "bool"):
        print(f"CSR Avoidance: {enable}")
        self.reroute_around_CSRs = enable


    @stack.command(name="PLOT_POTFIELD")
    def plot_potfields(self, enable: "bool"):
        print(f"Plot potential fields: {enable}")
        self.plot_potential_fields = enable


    @stack.command(name="ADD_BOX")
    def add_hardcoded_box(self, box_id: int):
        if box_id == 1:
            stack.stack("BOX CSR_BOX_HC1 46.419242,-4.647985 44.151525,-0.072488")
            stack.stack(f'COLOR CSR_BOX_HC1 {coloring["CSR"]}')
        elif box_id == 2:
            stack.stack("BOX CSR_BOX_HC2 42.326925,-9.727867 41.710363,-5.836893")
            stack.stack(f'COLOR CSR_BOX_HC2 {coloring["CSR"]}')
        elif box_id == 3:
            stack.stack("BOX CSR_BOX_HC3 40.033143,-9.83595 39.459697,-6.593472")
            stack.stack(f'COLOR CSR_BOX_HC3 {coloring["CSR"]}')


    # -------------------------------------------------------------------------------
    #   Periodically timed functions for CSR (climate sensitive region) avoidance
    # -------------------------------------------------------------------------------
    @core.timed_function(name='create_random_CSRs', dt=7200)
    def add_CSRs(self):
        if not self.create_random_CSRs:
            return
        # TODO get ERA5 data for 2022/12/01 and determine actual CSRs
        create = {
            "polygon": False,
            "box": True
        }
        if create["polygon"]:
            coords = generate_random_polygon(lon_range=(-12, -6))
            coords_str = " ".join(coords.apply(lambda row: f"{row.lat},{row.lon}", axis=1))
            stack.stack(f"POLY CSR_POLY_{self.polygon_id_count} {coords_str}")
            stack.stack(f'COLOR CSR_POLY_{self.polygon_id_count} {coloring["CSR"]}')
            self.polygon_id_count += 1
        if create["box"]:
            coords = generate_random_rectangle()
            stack.stack(f"BOX CSR_BOX_{self.box_id_count} {coords[0][0]},{coords[0][1]} {coords[1][0]},{coords[1][1]}")
            stack.stack(f'COLOR CSR_BOX_{self.box_id_count} {coloring["CSR"]}')
            self.box_id_count += 1


    @core.timed_function(name='check_intersection', dt=60)
    def check_ac_intersect_csr(self):
        if not self.reroute_around_CSRs:
            return
        for ac_id in traf.id: # all_aircrafts:
            ac_idx = traf.id2idx(ac_id)
            ac_route = traf.ap.route[ac_idx]  # get aircraft trajectory

            # do not intervene when aircraft is heading to destination
            if ac_route.iactwp + 1 == len(ac_route.wpname):
                continue

            # get one waypoint ahead of the one the aircraft is heading to, to still be able to diverge on time
            ac_coords = {
                "lat": ac_route.wplat[ac_route.iactwp + 1],
                "lon": ac_route.wplon[ac_route.iactwp + 1],
            }

            # check for intersections of that waypoint with CSRs and eventually reroute
            for shape_name, shape in dict(filter(lambda s: "CSR" in s[0], areafilter.basic_shapes.items())).items():
                if areafilter.checkInside(shape_name, ac_coords["lat"], ac_coords["lon"], 0):
                    print(f"Aircraft {ac_id} will intersect with {shape_name} soon! Trying to reroute...")
                    success = reroute_using_potential_field(ac_id, ac_route, shape, shape_name, plot=self.plot_potential_fields)
                    print(f"Rerouting {'successful' if success else 'failed'} for aircraft {ac_id}.")
