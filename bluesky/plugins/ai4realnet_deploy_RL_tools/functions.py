'''
Functions for the CentralisedStaticObstacleSectorCREnv-0 environment.
'''

import numpy as np
import bluesky as bs
from math import radians, sin, cos, sqrt, atan2
from statistics import mean
from typing import List

def bound_angle_positive_negative_180(angle_deg: float) -> float:
    """ maps any angle in degrees to the [-180,180] interval 
    Parameters
    __________
    angle_deg: float
        angle that needs to be mapped (in degrees)
    
    Returns
    __________
    angle_deg: float
        input angle mapped to the interval [-180,180] (in degrees)
    """

    if angle_deg > 180:
        return -(360 - angle_deg)
    elif angle_deg < -180:
        return (360 + angle_deg)
    else:
        return angle_deg

def get_point_at_distance(lat1, lon1, d, bearing, R=6371):
    """
    lat: latitude of the reference point, in degrees
    lon: longitude of the referemce point, in degrees
    d: target distance from the reference point, in km
    bearing: (true) heading, in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from the reference point, in degrees
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    a = np.radians(bearing)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d/R) + np.cos(lat1) * np.sin(d/R) * np.cos(a))
    lon2 = lon1 + np.arctan2(
        np.sin(a) * np.sin(d/R) * np.cos(lat1),
        np.cos(d/R) - np.sin(lat1) * np.sin(lat2)
    )
    return np.degrees(lat2), np.degrees(lon2)

def random_point_on_circle(radius: float) -> np.array:
    """ Get a random point on a circle circumference with given radius
    Parameters
    __________
    radius: float
        radius for the circle
    
    Returns
    __________
    point: np.array
        randomly sampled point
    """
    alpha = 2 * np.pi * np.random.uniform(0., 1.)
    x = radius * np.cos(alpha)
    y = radius * np.sin(alpha)
    return np.array([x, y])

def sort_points_clockwise(vertices: np.array) -> np.array:
    """ Sort the points in clockwise order
    Parameters
    __________
    vertices: np.array
        array of points
    
    Returns
    __________
    sorted_vertices: np.array
        sorted array of points
    """
    sorted_vertices = [vertices[i] for i in np.argsort([np.arctan2(v[1], v[0]) for v in vertices])]

    return sorted_vertices   

def polygon_area(vertices: np.array) -> float:
    """ Calculate the area of a polygon given the vertices
    Parameters
    __________
    vertices: np.array
        array of vertices of the polygon
    
    Returns
    __________
    area: float
        area of the polygon
    """
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]  # Wrap around to the first vertex
        area += x1 * y2 - y1 * x2
    area = np.abs(area) / 2.0
    return area

def nm_to_latlong(center: np.array, point: np.array) -> np.array:
    """ Convert a point in nm to lat/long coordinates
    Parameters
    __________
    center: np.array
        center point of the conversion
    point: np.array
        point to be converted
    
    Returns
    __________
    latlong: np.array
        converted point in lat/long coordinates
    """
    lat = center[0] + (point[0] / 60)
    lon = center[1] + (point[1] / (60 * np.cos(np.radians(center[0]))))
    return np.array([lat, lon])

def latlong_to_nm(center: np.array, point: np.array) -> np.array:
    """ Convert a point in lat/long coordinates to nm
    Parameters
    __________
    center: np.array
        center point of the conversion
    point: np.array
        point to be converted
    
    Returns
    __________
    nm: np.array
        converted point in nm
    """
    x = (point[0] - center[0]) * 60
    y = (point[1] - center[1]) * 60 * np.cos(np.radians(center[0]))
    return np.array([x, y])

def euclidean_distance(point1: np.array, point2: np.array) -> float:
    """ Calculate the euclidean distance between two points
    Parameters
    __________
    point1: np.array
        [x, y] of the first point
    point2: np.array
        [x, y] of the second point
        
    Returns
    __________
    distance: float
        euclidean distance between the two points
    """
    return np.sqrt(np.sum((point2 - point1)**2))

def get_hdg(point1: np.array, point2: np.array) -> float:
    """ Calculate the heading from point1 to point2
    Parameters
    __________
    point1: np.array
        [lat, lon] of the first point 
    point2: np.array
        [lat, lon] of the second point
    
    Returns
    __________
    hdg: float
        heading from point1 to point2
    """
    
    lat1, lon1 = np.radians(point1)
    lat2, lon2 = np.radians(point2)
    
    delta_lon = lon2 - lon1
    
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    
    hdg = np.degrees(np.arctan2(x, y))
    
    hdg = (hdg + 360) % 360 # Convert back to [0, 360] interval
    
    return hdg

def haversine_m(lat1, lon1, lat2, lon2):
    """
    Returns distance in meters between two WGS84 lat/lon points.
    """
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def bounding_circle_geodesic(coords):
    """
    Compute a simple enclosing circle for lat/lon points by using a centroid-like
    center and radius = max geodesic distance to points.

    Args:
        coords (list[tuple]): [(lat, lon), ...]

    Returns:
        center (tuple): (lat_center, lon_center)
        radius_m (float): Circle radius in meters
    """
    # Use arithmetic mean as a robust, fast “center” (good enough for small regions)
    lat_c = mean(lat for lat, _ in coords)
    lon_c = mean(lon for _, lon in coords)

    radius = max(haversine_m(lat_c, lon_c, lat, lon) for lat, lon in coords)
    return (lat_c, lon_c), radius

def resample_closed_border(latitudes, longitudes, n_points):
    """
    Resample a closed border polyline into approximately equidistant points (great-circle nm).

    Args:
        latitudes (list[float]): Vertex latitudes in order around the border.
        longitudes (list[float]): Vertex longitudes in order around the border.
        n_points (int): Number of output points along the border (start included, no duplicate at end).
        kwikqdrdist (callable): Function (lat1, lon1, lat2, lon2) -> (bearing_deg, distance_nm).
        kwikpos (callable): Function (lat, lon, bearing_deg, distance_nm) -> (lat2, lon2).

    Returns:
        list[tuple[float, float]]: List of (lat, lon) for the resampled border.
    """
    # Remove duplicate final vertex if caller already closed the polygon
    if latitudes[0] == latitudes[-1] and longitudes[0] == longitudes[-1]:
        latitudes = latitudes[:-1]
        longitudes = longitudes[:-1]

    m = len(latitudes)
    if m < 2:
        raise ValueError("Need at least two vertices")

    # Precompute segment bearings and lengths for each edge i -> i+1 (wrapping at the end)
    seg_bearings = []
    seg_lengths = []
    for i in range(m):
        j = (i + 1) % m
        b, d = bs.tools.geo.kwikqdrdist(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
        seg_bearings.append(b)
        seg_lengths.append(d)

    seg_lengths = np.array(seg_lengths, dtype=float)
    perim = float(seg_lengths.sum())

    # Cumulative distances along the loop: s[0]=0, s[i]=distance up to start of segment i
    # We have m segments; define s of length m+1 to simplify searching
    s = np.zeros(m + 1, dtype=float)
    s[1:] = np.cumsum(seg_lengths)

    # Target distances (exclude the duplicate 0/perimeter point)
    targets = np.linspace(0.0, perim, num=n_points, endpoint=False)

    out = []
    # Walk each target into the correct segment
    k = 0  # index into segments; we’ll advance monotonically
    for t in targets:
        # Find segment such that s[k] <= t < s[k+1]
        # (k only increases, so a while-loop is efficient)
        while not (s[k] <= t < s[k + 1]):
            k += 1
            if k >= m:  # wrap if numerical drift puts us at the end
                k = 0
                t = t % perim

        # Distance from start of segment k
        offset = t - s[k]

        # Segment start (vertex k) and its bearing/length
        lat0 = latitudes[k]
        lon0 = longitudes[k]
        bear = seg_bearings[k]

        # Step along this segment by 'offset'
        lat_t, lon_t = bs.tools.geo.kwikpos(lat0, lon0, bear, float(offset))
        out.append((lat_t, lon_t))
    out = np.array(out, dtype=np.float64)
    return out