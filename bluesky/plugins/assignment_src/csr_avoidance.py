import numpy as np
import pandas as pd


def generate_random_polygon(num_vertices=5, lat_range=(28, 45), lon_range=(-40, -6)):
    center_lat = np.random.randint(lat_range[0], lat_range[1])
    center_lon = np.random.randint(lon_range[0], lon_range[1])

    size_lat = np.random.uniform(1, 4)
    size_lon = np.random.uniform(1, 4)

    lat_coords = np.random.uniform(center_lat - size_lat / 2, center_lat + size_lat / 2, num_vertices).round(5)
    lon_coords = np.random.uniform(center_lon - size_lon / 2, center_lon + size_lon / 2, num_vertices).round(5)

    points = np.column_stack((lat_coords, lon_coords))

    # Step 2: Sort points in counter-clockwise order
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_points = points[np.argsort(angles)]

    return pd.DataFrame(sorted_points, columns=['lat', 'lon'])
