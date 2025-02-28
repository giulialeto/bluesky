import numpy as np
import pandas as pd
from bluesky.tools import areafilter
from itertools import combinations

def get_current_location(traf):
    """
    Collects active waypoints for all aircraft and updates them if the next waypoint 
    will be reached within the given time horizon.

    Parameters:
    traf (object): BlueSky traffic object containing all aircraft and their routes.

    Returns:
    list: A list of tuples (ac_id, wpname, wplat, wplon) with the active or updated waypoints.
    """
    current_laction = []

    # import code
    # code.interact(local=locals())

    for ac_id in traf.id:  # Iterate through all aircraft
        ac_idx = traf.id2idx(ac_id)

        current_lat = traf.lat[ac_idx]  # Get latitude of the aircraft
        current_lon = traf.lon[ac_idx]  # Get longitude of the aircraft
        current_alt = traf.alt[ac_idx]  # Get altitude of the aircraft

        current_laction.append((ac_id, 'current_location', current_lat, current_lon, current_alt))
    
    # Convert list to Pandas DataFrame
    current_location_df = pd.DataFrame(current_laction, columns=['ac_id', 'wpname', 'lat', 'lon', 'alt'])

    return current_location_df

def get_future_location(traf, current_time, time_horizon):
    """
    Collects active waypoints for all aircraft and updates them if the next waypoint 
    will be reached within the given time horizon.

    Parameters:
    traf (object): BlueSky traffic object containing all aircraft and their routes.
    current_time (float): Current simulation time in seconds.
    time_horizon (int, optional): Time window in seconds to check for waypoint updates.

    Returns:
    list: A list of tuples (ac_id, wpname, wplat, wplon) with the active or updated waypoints.
    """
    future_waypoints = []

    # import code
    # code.interact(local=locals())

    for ac_id in traf.id:  # Iterate through all aircraft
        ac_idx = traf.id2idx(ac_id)

        route = traf.ap.route[ac_idx]  # Get the route of the aircraft
        iactwp = route.iactwp  # Get index of the currently active waypoint
        
        # if iactwp is None or iactwp < 0 or iactwp >= len(route.wpname) - 1:
        #     continue  # Skip if no active waypoint or it's the last one

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

def get_sector_count(sector_list, current_location_df, future_waypoints_df):
    """
    Applies a mask for multiple areas and creates separate binary columns for each area.

    Parameters:
    df (pd.DataFrame): DataFrame with columns ['ac_id', 'wpname', 'wplat', 'wplon', 'wprta'].
    sector_list (list): List of area names to check.

    Returns:
    pd.DataFrame: Updated DataFrame with additional columns for each area in sector_list.
    """

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
        print(sector_count)
        # import code
        # code.interact(local=locals())
    return sector_count


# Function to check if a selection of sectors covers all areas exactly once
def get_feasible_sector_combinations_with_ATCO_available(feasible_sector_combinations, max_amount_sectors):
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
    
    # Filter rows where 'concatenate' contains exactly 6 characters and re-index the DataFrame
    filtered_combinations = filtered_combinations[filtered_combinations['concatenate'].apply(lambda x: len(x) == 6)].reset_index(drop=True)
    filtered_combinations = filtered_combinations.drop(columns='concatenate')

    return filtered_combinations



# Function to compute the sum of sector counts for each element in a row
def compute_ac_count(row, sector_count):
    return [sum(sector_count.get(letter, 0) for letter in str(element) if element is not None) for element in row]

def sector_count_in_feasible_combinations(filtered_combinations, sector_count):
    # Generate column names
    ac_count_columns = [f"ac_count_sector_{i+1}" for i in range(filtered_combinations.shape[1])]

    # Apply the function row-wise and create new columns
    ac_counts = filtered_combinations.apply(lambda row: compute_ac_count(row, sector_count), axis=1, result_type='expand')

    # Assign correct column names to the new DataFrame
    ac_counts.columns = ac_count_columns

    # Merge the new columns with the original filtered_combinations DataFrame
    filtered_combinations = pd.concat([filtered_combinations, ac_counts], axis=1)

    return filtered_combinations, ac_count_columns

def select_best_grouping(filtered_combinations, ac_count_columns, max_aircraft_allowed):
    # Calculate the maximum value across the ac_count_sector_ columns for each row
    # import code
    # code.interact(local=locals())

    filtered_combinations['max_ac_count'] = filtered_combinations[ac_count_columns].max(axis=1)
    # red(f"filtered_combinations {filtered_combinations}")
    # Find the lowest maximum value
    max_max_ac_count = filtered_combinations['max_ac_count'].max()
    if max_max_ac_count <= max_aircraft_allowed:
        selected_sectors = filtered_combinations[filtered_combinations['max_ac_count'] == max_max_ac_count].drop(columns=['max_ac_count'])

        # Count the number of None values in each row
        none_counts = selected_sectors.isna().sum(axis=1)

        # If there are None values, select the row(s) with the most None values
        if none_counts.max() > 0:
            selected_sectors = selected_sectors[none_counts == none_counts.max()]
            selected_sectors = selected_sectors.iloc[:1]
        else:
            # If there are no None values, select any of the available rows
            # selected_sectors = selected_sectors.sample(n=1, random_state=42)
            selected_sectors = selected_sectors.iloc[:1]
        return selected_sectors
    
    min_max_ac_count = filtered_combinations['max_ac_count'].min()

    # Filter the DataFrame to keep only rows where the maximum aircraft count is the lowest found
    selected_sectors = filtered_combinations[filtered_combinations['max_ac_count'] == min_max_ac_count].drop(columns=['max_ac_count'])
    
    selected_sectors['min_ac_count'] = selected_sectors[ac_count_columns].min(axis=1)
    max_min_ac_count = selected_sectors['min_ac_count'].max()
    selected_sectors = selected_sectors[selected_sectors['min_ac_count'] == max_min_ac_count].drop(columns=['min_ac_count'])

    selected_sectors = selected_sectors.iloc[:1]

    # if min_max_ac_count > max_aircraft_allowed:
    #     red(f"WARNING! Aircraft count is {min_max_ac_count}: exceeds max_aircraft_allowed {max_aircraft_allowed}")

    return selected_sectors

