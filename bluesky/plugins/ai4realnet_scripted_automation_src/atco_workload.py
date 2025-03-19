import numpy as np
import pandas as pd
from bluesky import sim
from bluesky.tools import areafilter
from itertools import combinations

def get_current_location(traf):
    """
    Gets the current location of all aircraft

    Args:
        traf (object): BlueSky traffic object containing all aircraft and their routes.

    Returns:
        current_location_df (pd.DataFrame): contains columns ['ac_id', 'wpname', 'lat', 'lon', 'alt'] with the current location of all aircraft.
    """
    current_location = []

    for ac_id in traf.id:  # Iterate through all aircraft
        ac_idx = traf.id2idx(ac_id)

        current_lat = traf.lat[ac_idx]  # Get latitude of the aircraft
        current_lon = traf.lon[ac_idx]  # Get longitude of the aircraft
        current_alt = traf.alt[ac_idx]  # Get altitude of the aircraft

        current_location.append((ac_id, 'current_location', current_lat, current_lon, current_alt))
    
    # Convert list to Pandas DataFrame
    current_location_df = pd.DataFrame(current_location, columns=['ac_id', 'wpname', 'lat', 'lon', 'alt'])

    return current_location_df

def get_future_location(traf, current_time, time_horizon):
    """
    Saves the active waypoint of the aircraft when they fall within the given time horizon.
    
    Args:
        traf (object): BlueSky traffic object containing all aircraft and their routes.
        current_time (float): Current simulation time in seconds.
        time_horizon (int): Time window in seconds to check for waypoint updates.

    Returns:
        future_waypoints_df (pd.DataFrame): contains columns ['ac_id', 'wpname', 'lat', 'lon', 'alt'] with the future location of all aircraft, if the active waypoint will be reached within the time horizon.
    """
    future_waypoints = []

    for ac_id in traf.id:  # Iterate through all aircraft

        ac_idx = traf.id2idx(ac_id)
        route = traf.ap.route[ac_idx]  # Get the route of the aircraft
        iactwp = route.iactwp  # Get index of the currently active waypoint
        
        # Check the next waypoint (active waypoint)
        next_wprta = route.wprta[iactwp]
        # yellow(f"current_time {current_time}")
        # yellow(f"next_wprta {next_wprta}")
        # yellow(f"current_time + time_horizon {current_time + time_horizon}")

        if next_wprta <= (current_time + time_horizon):
            # If the next waypoint is within the time horizon, substitute it
            wpname = route.wpname[iactwp]
            wplat = route.wplat[iactwp]
            wplon = route.wplon[iactwp]
            wpalt = route.wpalt[iactwp]
            future_waypoints.append((ac_id, f"next_{wpname}", wplat, wplon, wpalt))
            # green(f"ac_id {ac_id} wpname {wpname}")
        # else:
            # red(f"ac_id {ac_id} next_wprta {next_wprta} not in time horizon")
    
    # Convert list to Pandas DataFrame
    future_waypoints_df = pd.DataFrame(future_waypoints, columns=['ac_id', 'wpname', 'lat', 'lon', 'alt'])

    return future_waypoints_df

def get_departing_aircraft(scen_commands_df, current_time, time_horizon):

    print(current_time)

    departing_aircraft_df = scen_commands_df[(scen_commands_df[0] >= current_time) & (scen_commands_df[0] <= time_horizon) & scen_commands_df[1].str.contains('CRE', na=False)]

    # parse 
    return departing_aircraft_df

def get_sector_count(sector_list, current_location_df, future_waypoints_df):
    """
    Count the number of aircraft in each sector.
    
    Args:
        sector_list (list): List of area names to check.
        current_location_df (pd.DataFrame): DataFrame with columns ['ac_id', 'wpname', 'lat', 'lon', 'alt'] with the current location of all aircraft.
        future_waypoints_df (pd.DataFrame): DataFrame with columns ['ac_id', 'wpname', 'lat', 'lon', 'alt'] with the future location of all aircraft, if the active waypoint will be reached within the time horizon.

    Returns:
        sector_count (pd.Series): Contains with the number of aircraft in each sector.
    """
    if future_waypoints_df.empty and current_location_df.empty:
        return pd.DataFrame()
    sector_count_mask = pd.concat([future_waypoints_df, current_location_df], ignore_index=True)

    sector_count = {}

    if not sector_count_mask.empty:

        lats = sector_count_mask['lat'].to_numpy()
        lons = sector_count_mask['lon'].to_numpy()
        alts = sector_count_mask['alt'].to_numpy()

        for sector in sector_list:
            # Check for all waypoints at once using numpy arrays
            sector_count_mask[sector] = areafilter.checkInside(sector, lats, lons, alts)
        
        # Group by the normalized ac_id and take the logical OR (so that if any entry is True, it's counted as True)
        sector_count_mask_grouped = sector_count_mask[['ac_id'] + sector_list].groupby('ac_id', as_index=False).any()
        
        # print(sector_count_mask)
        # yellow(sector_count_mask_grouped)
        # Count the number of aircraft in each sector
        sector_count = sector_count_mask_grouped[sector_list].sum()
        print(f"\n{sim.utc}")
        print(sector_count)
    return sector_count

def get_feasible_sector_combinations_with_ATCO_available(feasible_sector_combinations, max_amount_sectors):
    """
    Generate all feasible combinations of sectors with a given maximum amount of sectors open at the same time (which is the number of air traffic controllers available at the same time).
    
    Args:
        feasible_sector_combinations (list): List of feasible sector combinations.
        max_amount_sectors (int): Maximum number of sectors open at the same time, which is the number of air traffic controllers available at the same time.
    
    Returns:
        filtered_combinations (pd.DataFrame): Contains all feasible combinations of sectors with a given maximum amount of sectors.
    """
    # Generate all combinations with 1 to max_amount_sectors elements
    all_combinations = []
    header = []
    for r in range(1, max_amount_sectors + 1):
        all_combinations.extend(combinations(feasible_sector_combinations, r))
        header.append(f"sector_{r}")
    
    # Create DataFrame
    combinations_df = pd.DataFrame(all_combinations, columns=header)
    
    # Concatenate all elements in the row into a single string
    combinations_df['concatenate'] = combinations_df.apply(lambda row: ''.join(row.dropna().astype(str)), axis=1)
    
    # First filter: Keep only rows where all characters in 'concatenate' are unique
    filtered_combinations = combinations_df[combinations_df['concatenate'].apply(lambda x: len(set(x)) == len(x))]
    
    # Second filter: Keep only rows where 'concatenate' contains exactly 6 characters and re-index the DataFrame
    filtered_combinations = filtered_combinations[filtered_combinations['concatenate'].apply(lambda x: len(x) == 6)].reset_index(drop=True)
    
    # Drop the 'concatenate' column
    filtered_combinations = filtered_combinations.drop(columns='concatenate')

    return filtered_combinations

# Function to compute the sum of sector counts for each element in a row, which is defined as the sum of the sector counts of the characters in the name of the combinated sectors
def compute_ac_count(row, sector_count):
    return [sum(sector_count.get(letter, 0) for letter in str(element) if element is not None) for element in row]

def sector_count_in_feasible_combinations(filtered_combinations, sector_count):
    """
    Count the number of aircraft in each feasible sector combination.

    Args:
        filtered_combinations (pd.DataFrame): Contains all feasible combinations of sectors with a given maximum amount of sectors.
        sector_count (pd.Series): Contains with the number of aircraft in each sector.

    Returns:
        filtered_combinations (pd.DataFrame): Contains all feasible combinations of sectors with a given maximum amount of sectors, with the number of aircraft in each sector combination.
        ac_count_columns (list): List of column names containing the number of aircraft in each sector combination
    """

    # Generate column names
    ac_count_columns = [f"ac_count_sector_{i+1}" for i in range(filtered_combinations.shape[1])]

    # Apply the function row-wise and create new columns for each sector combination containing the combined sector count
    ac_counts = filtered_combinations.apply(lambda row: compute_ac_count(row, sector_count), axis=1, result_type='expand')

    # Assign correct column names to the new DataFrame
    ac_counts.columns = ac_count_columns

    # Merge the new columns with the original filtered_combinations DataFrame
    filtered_combinations = pd.concat([filtered_combinations, ac_counts], axis=1)

    return filtered_combinations, ac_count_columns

def select_best_grouping(filtered_combinations, ac_count_columns, max_aircraft_allowed):
    """
    Select the best grouping of sectors based on the number of aircraft in each sector combination.

    Args:
        filtered_combinations (pd.DataFrame): Contains all feasible combinations of sectors with a given maximum amount of sectors.
        ac_count_columns (list): List of column names containing the number of aircraft in each sector combination.
        max_aircraft_allowed (int): Soft constraint suggesting the preferable maximum number of aircraft allowed in one sector.

    Returns:
        selected_sectors (pd.DataFrame): Contains the selected sector combination to be scheduled.
    """
    # Calculate the maximum value across the ac_count_columns for each row
    filtered_combinations['max_ac_count'] = filtered_combinations[ac_count_columns].max(axis=1)
    # red(f"filtered_combinations {filtered_combinations}")

    # Find the maximum value of the aircraft count in the DataFrame
    max_max_ac_count = filtered_combinations['max_ac_count'].max()

    # If the maximum value of the aircraft count is less than or equal to the maximum number of aircraft allowed,
    # we will prioritize minimizing the amount of sectors open
    if max_max_ac_count <= max_aircraft_allowed:
        # Filter the DataFrame to keep only rows where the maximum aircraft count is the highest found
        selected_sectors = filtered_combinations[filtered_combinations['max_ac_count'] == max_max_ac_count].drop(columns=['max_ac_count'])
        return selected_sectors

    filtered_combinations["All_Less_Than_allowed_aircraft"] = filtered_combinations[ac_count_columns].lt(max_aircraft_allowed).all(axis=1)

    if filtered_combinations["All_Less_Than_allowed_aircraft"].any():
        # Filter the DataFrame to keep only rows where the maximum aircraft count is the highest found
        selected_sectors = filtered_combinations[filtered_combinations['All_Less_Than_allowed_aircraft'] == True].drop(columns=['All_Less_Than_allowed_aircraft'])
        
        max_max_ac_count_selected_sectoors = selected_sectors['max_ac_count'].max()

        selected_sectors = selected_sectors[selected_sectors['max_ac_count'] == max_max_ac_count_selected_sectoors].drop(columns=['max_ac_count'])

        # Count the number of None values in each row
        none_counts = selected_sectors.isna().sum(axis=1)

        # If there are None values, select the row(s) with the most None values, which is the combination with the least amount of sectors open
        if none_counts.max() > 0:
            selected_sectors = selected_sectors[none_counts == none_counts.max()]
            # If there are multiple rows with the most None values, select any of them
            selected_sectors = selected_sectors.iloc[:1]
        else:
            # If there are no None values, select any of the available rows
            selected_sectors = selected_sectors.iloc[:1]
        return selected_sectors
    
    filtered_combinations = filtered_combinations.drop(columns=['All_Less_Than_allowed_aircraft'])

    # If the maximum value of the aircraft count is greater than the maximum number of aircraft allowed,
    # we will prioritize sectors in which the maximum aircraft count is at its lowest, and where the aircraft count in the other sectors is balanced
    
    # Find the minimum value of the maximum aircraft count in the DataFrame
    min_max_ac_count = filtered_combinations['max_ac_count'].min()

    # Filter the DataFrame to keep only rows where the maximum aircraft count is the lowest found
    selected_sectors = filtered_combinations[filtered_combinations['max_ac_count'] == min_max_ac_count].drop(columns=['max_ac_count'])
    
    # find the minium number of aircraft in the sectors 
    selected_sectors['min_ac_count'] = selected_sectors[ac_count_columns].min(axis=1)

    # Select the combinations of sectors in which the minimum number of aircraft per sector is the highest
    max_min_ac_count = selected_sectors['min_ac_count'].max()
    selected_sectors = selected_sectors[selected_sectors['min_ac_count'] == max_min_ac_count].drop(columns=['min_ac_count'])

    # Select any sectors that meet the criteria
    selected_sectors = selected_sectors.iloc[:1]

    # if min_max_ac_count > max_aircraft_allowed:
    #     red(f"WARNING! Aircraft count is {min_max_ac_count}: exceeds max_aircraft_allowed {max_aircraft_allowed}")

    return selected_sectors

