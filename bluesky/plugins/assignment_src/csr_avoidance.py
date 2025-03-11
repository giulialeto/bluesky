import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from bluesky import stack
from bluesky.tools import areafilter

# from larp-fork
# pip install -e git+https://github.com/jsmretschnig/Larp.git#egg=Larp
import larp.io as lpio
import larp.quad as quad
import larp.network as network

# -------------------------------------------------------------------------------
#   Generate random shapes
# -------------------------------------------------------------------------------
def generate_random_polygon(num_vertices=5, lat_range=(35, 42), lon_range=(-15, -6)):
    center_lat = np.random.randint(lat_range[0], lat_range[1])
    center_lon = np.random.randint(lon_range[0], lon_range[1])

    size_lat = np.random.uniform(2, 4)
    size_lon = np.random.uniform(2, 4)

    lat_coords = np.random.uniform(center_lat - size_lat / 2, center_lat + size_lat / 2, num_vertices).round(5)
    lon_coords = np.random.uniform(center_lon - size_lon / 2, center_lon + size_lon / 2, num_vertices).round(5)

    points = np.column_stack((lat_coords, lon_coords))

    # Step 2: Sort points in counter-clockwise order
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_points = points[np.argsort(angles)]
    sorted_points = np.vstack([sorted_points, sorted_points[0]])  # close the polygon

    return pd.DataFrame(sorted_points, columns=['lat', 'lon'])


def generate_random_rectangle(lat_range=(35, 42), lon_range=(-15, -6)):
    uppper_left_lat = np.random.uniform(lat_range[0], lat_range[1])
    uppper_left_lon = np.random.uniform(lon_range[0], lon_range[1])

    size_lat = np.random.uniform(1, 3)
    size_lon = np.random.uniform(1, 3)

    return np.array([
        [uppper_left_lat, uppper_left_lon],
        [uppper_left_lat - size_lat, uppper_left_lon + size_lon]
    ]).round(5)


# -------------------------------------------------------------------------------
#   Path finding using Potential Fields
# -------------------------------------------------------------------------------
def _potential_field(ac_id, start, end, fc, plot=False):
    """ Path finding algorithm to determine the new waypoints around the obstacles using potential fields.
        Based on https://github.com/wzjoriv/Larp/blob/main/presentation.ipynb.
        Citation: Rivera, Josue N., and Dengfeng Sun. "Multi-Scale Cell Decomposition for Path Planning using
        Restrictive Routing Potential Fields." arXiv preprint arXiv:2408.02786 (2024).
        Args:
            ac_id: Aircraft ID
            start (tuple): (longitude, latitude)
            end (tuple): (longitude, latitude)
            fc (dict): JSON FeatureCollection with the obstacles to avoid.
            plot (boolean): Whether to plot the potential field.
        Returns:
            np.ndarray | None: Array of coordinates connecting start and end and avoiding the obstacle. If no route found, return None.
    """
    field = lpio.fromGeoJSON(fc)
    field.size += 1  # 1 decimal degree ~= 11.1km

    edges = [0.1, 0.2, 0.4, 0.6, 0.8]
    quadtree = quad.QuadTree(
        field,
        minimum_length_limit=0.05,  # 0.05 seems to be a good trade-off in terms of resolution
        maximum_length_limit=10,
        edge_bounds=edges
    )
    quadtree.build()

    routing_network = network.RoutingNetwork(quadtree=quadtree, build_network=True)

    k = 1.2
    route = routing_network.find_route(start, end, scale_tranform=lambda x: 1 / (k * (1.0 - x + 1e-10)), alg="A*", penalty=10.0)
    print(f"Route found: {route is not None}")

    if route is None:
        # always plot
        plt.figure(dpi=150)
        display = field.to_image(resolution=400)
        plt.imshow(display, cmap='jet', extent=field.get_extent(), alpha=0.8)
        plt.colorbar().set_ticks([0.0] + edges + [1.0])
        plt.plot(start[0], start[1], 'w^')  # start
        plt.plot(end[0], end[1], 'wv')  # end
        plt.title(f"Aircraft {ac_id} | no route found")
        plt.show()

        return None

    route_path = network.RoutingNetwork.route_to_lines_collection(start, end, route, remapped=True)

    if plot:
        routes_lines = routing_network.to_routes_lines_collection()

        plt.figure(dpi=150)
        display = field.to_image(resolution=400)
        plt.imshow(display, cmap='jet', extent=field.get_extent(), alpha=0.8)
        plt.colorbar().set_ticks([0.0] + edges + [1.0])
        plt.plot(*routes_lines, color="#fff", alpha=1.0, linewidth=0.5)
        plt.plot(route_path[:, 0], route_path[:, 1], color="#0f0", alpha=1.0, linewidth=1.0)
        plt.plot(route_path[0, 0], route_path[0, 1], 'w^', markersize=10.0, markeredgewidth=1.5)  # start
        plt.plot(route_path[-1, 0], route_path[-1, 1], 'wv')  # end
        plt.title(f"Aircraft {ac_id}")
        plt.show()

    return route_path


def _remove_consecutive_duplicates(coords, tol=1e-5):
    if len(coords) == 0:
        return coords

    filtered_coords = [coords[0]]  # Start with the first coordinate

    for i in range(1, len(coords)):
        prev_lat, prev_lon = filtered_coords[-1]
        curr_lat, curr_lon = coords[i]

        # Use np.isclose to compare floating points with a tolerance
        if not (np.isclose(curr_lat, prev_lat, atol=tol) or np.isclose(curr_lon, prev_lon, atol=tol)):
            filtered_coords.append(coords[i])

    return np.array(filtered_coords)  # Convert back to ndarray for consistency


def reroute_using_potential_field(ac_id, ac_route, shape, shape_name, plot=False):
    """ Find a route around an obstacle using potential fields.
        Args:
            ac_id: Aircraft ID
            ac_route: Original aircraft trajectory
            shape: Geometric obstacle to avoid
            shape_name (str): Name of shape
        Returns:
            - Adds the new waypoints of the alternative trajectory to the route.
            - Deletes the old waypoints going through the CSR from the route.
            success (boolean): Rerouting succeeded.
    """
    # Find the first waypoint after the CSR
    upcoming_traj_coords = np.column_stack((ac_route.wplat, ac_route.wplon))[ac_route.iactwp:]
    idx_first_outside, coords_first_outside = next(((i, x) for i, x in enumerate(upcoming_traj_coords[1:]) if
                                                    not areafilter.checkInside(shape_name, x[0], x[1], 0)),
                                                   (None, None))

    # Define start and end coordinates for the path finding algorithm
    start = (ac_route.wplon[ac_route.iactwp], ac_route.wplat[ac_route.iactwp])  # lon, lat
    if coords_first_outside is None:
        return False
    end = (coords_first_outside[1], coords_first_outside[0])  # lon, lat

    repulsion = 0.01

    # Define the obstacle in a format readable by the path finding algorithm
    if "POLY" in shape_name:
        coords = shape.coordinates
        coords = [[coords[i + 1], coords[i]] for i in range(0, len(coords), 2)]
        feature = { "type": "Feature", "properties": {}, "geometry": {
            "type": "MultiLineString",
            "coordinates": [coords],
            "repulsion": [[repulsion, 0], [0, repulsion]]
        }}
    elif "BOX" in shape_name:
        feature = { "type": "Feature", "properties": {}, "geometry": {
            "type": "Rectangle",
            "coordinates": [[shape.coordinates[1], shape.coordinates[0]],
                            [shape.coordinates[3], shape.coordinates[2]]],
            "repulsion": [[repulsion, 0], [0, repulsion]]
        }}

    to_avoid = {
        "type": "FeatureCollection",
        "name": "reroute",
        "crs": None,
        "features": [
            feature
        ]
    }

    # Try to find a path around the obstacle.
    # If no path is found, try setting the 2nd, 3rd, ... waypoint after the CSR as end coordinate
    try_count = 0
    while (path := _potential_field(ac_id, start, end, to_avoid, plot)) is None:
        print("Try again to find a route around the obstacle.")
        idx_first_outside += 1
        try_count += 1
        if idx_first_outside >= len(upcoming_traj_coords) - 1 or try_count == 5:  # make sure to always keep destination
            print("Couldn't find any route around the obstacle.")
            return False
        coords_first_outside = upcoming_traj_coords[idx_first_outside]
        end = (coords_first_outside[1], coords_first_outside[0])

    path = path[1:-1]  # do not duplicate start and end coordinates

    # Determine waypoints to delete as they intersect the CSR
    first_wpt_to_delete = ac_route.iactwp + 1
    wpts_to_delete = ac_route.wpname[first_wpt_to_delete:first_wpt_to_delete + idx_first_outside]

    # Add the new waypoints to the trajectory
    wpt_before = ac_route.wpname[ac_route.iactwp]  # the starting wpt
    path = _remove_consecutive_duplicates(path)
    for index, wpt in enumerate(path[::-1]):
        stack.stack(f"ADDWPT {ac_id} {wpt[1]},{wpt[0]} ,,, {wpt_before}")  # need 3 commas to specify the last arg

    # Delete the old waypoints from the route
    for wp in wpts_to_delete:
        stack.stack(f"DELWPT {ac_id} {wp}")

    stack.stack(f"LNAV {ac_id} ON")
    stack.stack(f"VNAV {ac_id} ON")
    return True
