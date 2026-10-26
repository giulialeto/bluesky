'''
Functions for the CentralisedStaticObstacleSectorCREnv-0 environment.
'''

import numpy as np
from bluesky.tools.geo import kwikqdrdist
from bluesky.tools.aero import nm
from bluesky.tools.misc import degto180
from bluesky.tools import areafilter
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

# from pyproj import Geod

# WGS84 = Geod(ellps="WGS84")
# NM2_IN_M2 = 1852.0**2  # 1 nm = 1852 m

# def poly_area_nm2(lats, lons):
#     # area_m2 is signed: positive for counter-clockwise, negative for clockwise
#     area_m2, _ = WGS84.polygon_area_perimeter(lons, lats)
#     return abs(area_m2) / NM2_IN_M2


def interpolate_along_obstacle_vertices(vertex_1, vertex_2, n=15):
    """Interpolate n points between vertex_1 and vertex_2."""
    lats = np.linspace(vertex_1[0], vertex_2[0], n)
    lons = np.linspace(vertex_1[1], vertex_2[1], n)
    return list(zip(lats, lons))

def generate_random_aircraft(n_ac, sector_name, obstacle_names, latitude_bounds, longitude_bounds):
    """
    Generate random scenario with random initial positions and random destinations for the aircraft
    """
    # HARDCODED
    orig_altitude = 350
    dest_altitude = 350

    # if the aircraft is generated inside the sector, keep it, otherwise regenerate
    min_lat, max_lat = latitude_bounds
    min_lon, max_lon = longitude_bounds

    lat_orig = np.full(n_ac, np.nan)
    lon_orig = np.full(n_ac, np.nan)
    good = np.zeros(n_ac, dtype=bool)

    max_iter = 10_000  # safety guard
    counter = 0

    while np.isnan(lat_orig).any():
        counter += 1
        if counter > max_iter:
            raise RuntimeError("ORIG Too many iterations - check bounds/sector overlap.")

        # indices that still need valid points
        # remaining_idx = np.isnan(lat_orig)
        remaining_idx = np.flatnonzero(~good)
        m = remaining_idx.size

        # sample only for the remaining positions
        lat_try = np.random.uniform(min_lat, max_lat, size=m)
        lon_try = np.random.uniform(min_lon, max_lon, size=m)
        altitude = np.ones(m)*orig_altitude

        inside_sector = bs.tools.areafilter.checkInside(
            sector_name,
            lat_try,
            lon_try,
            altitude
        ).astype(bool)

        # debug.cyan(f'inside_sector: {inside_sector}')
        inside_any_restricted = np.zeros_like(inside_sector, dtype=bool)
        # debug.cyan(f'inside_any_restricted: {inside_any_restricted}')

        for name in obstacle_names:
            # print(f'name: {name}')

            inside_restricted_area = bs.tools.areafilter.checkInside(
                name,
                lat_try,
                lon_try,
                altitude
            ).astype(bool)
            # debug.cyan(f'inside_restricted: {inside_restricted_area}')
            inside_any_restricted |= inside_restricted_area
            # debug.cyan(f'inside_any_restricted: {inside_any_restricted}')


        inside = inside_sector & (~inside_any_restricted)
        # debug.cyan(f'inside: {inside}')

        # place successful samples into their slots
        lat_orig[remaining_idx[inside]] = lat_try[inside]
        lon_orig[remaining_idx[inside]] = lon_try[inside]
        good[remaining_idx[inside]] = True

    lat_dest = np.full(n_ac, np.nan)
    lon_dest = np.full(n_ac, np.nan)
    good = np.zeros(n_ac, dtype=bool)

    max_iter = 10_000  # safety guard
    counter = 0

    while np.isnan(lat_dest).any():
        counter += 1
        if counter > max_iter:
            raise RuntimeError("DEST Too many iterations — check bounds/sector overlap.")

        # indices that still need valid points
        # remaining_idx = np.isnan(lat_dest)
        remaining_idx = np.flatnonzero(~good)
        m = remaining_idx.size

        # sample only for the remaining positions
        lat_try = np.random.uniform(min_lat, max_lat, size=m)
        lon_try = np.random.uniform(min_lon, max_lon, size=m)
        altitude = np.ones(m)*dest_altitude

        inside_sector = bs.tools.areafilter.checkInside(
            sector_name,
            lat_try,
            lon_try,
            altitude
        ).astype(bool)
        # debug.red(f'inside_sector: {inside_sector}')

        inside_any_restricted = np.zeros_like(inside_sector, dtype=bool)
        # debug.red(f'inside_any_restricted: {inside_any_restricted}')

        for name in obstacle_names:
            # print(f'name: {name}')
            inside_restricted_area = bs.tools.areafilter.checkInside(
                name,
                lat_try,
                lon_try,
                altitude
            ).astype(bool)
            # debug.red(f'inside_restricted_area: {inside_restricted_area}, lat_try: {lat_try}, lon_try: {lon_try}, altitude: {altitude} ')
            inside_any_restricted |= inside_restricted_area
            # debug.red(f'inside_any_restricted: {inside_any_restricted}')

        inside = inside_sector & (~inside_any_restricted)
        # debug.red(f'inside: {inside}')

        # place successful samples into their slots
        lat_dest[remaining_idx[inside]] = lat_try[inside]
        lon_dest[remaining_idx[inside]] = lon_try[inside]
        good[remaining_idx[inside]] = True
        # print(f'lat_dest: {lat_dest}')
        # print(f'lon_dest: {lon_dest}')
    heading, _ = bs.tools.geo.kwikqdrdist(lat_orig, lon_orig, lat_dest, lon_dest)

    for ac_idx in range(n_ac):
        bs.stack.process(f'CRE AC{ac_idx+1}, B787, {lat_orig[ac_idx]}, {lon_orig[ac_idx]}, {heading[ac_idx]}, {orig_altitude}, 150')
        bs.stack.process(f'DEST AC{ac_idx+1} {lat_dest[ac_idx]} {lon_dest[ac_idx]}')
        # bs.stack.stack(f'CRE AC{ac_idx+1}, B787, {lat_orig[ac_idx]}, {lon_orig[ac_idx]}, {heading[ac_idx]}, {orig_altitude}, 150')
        # bs.stack.stack(f'DEST AC{ac_idx+1} {lat_dest[ac_idx]} {lon_dest[ac_idx]}')


def polygon_observation(ac_lat: float, ac_lon: float, ac_hdg: float, ac_alt: float, area_name: str, vertices):
    """ Observation of a (convex) polygonal obstacle as seen from the aircraft.
    The observation consists of:
     - the distance to the nearest point of the polygon (positive when the aircraft is outside the polygon, negative when inside), 
     - the bearing of that point relative to the aircraft heading,
     - the bearings of the left-most and right-most tangents of the polygon relative to the aircraft heading.

    Parameters
    __________
    ac_lat, ac_lon: float
        position of the aircraft (degrees)
    ac_hdg: float
        heading of the aircraft (degrees)
    ac_alt: float
        altitude of the aircraft (m)
    area_name: str
        name of the polygon in the BlueSky area filter
    vertices: array-like, shape (N, 2)
        vertices of the polygon as [lat, lon] (degrees)

    Returns
    __________
    clearance_km: float
        distance (positive when the aircraft is outside the polygon, negative when inside) from the aircraft to the nearest point of the polygon boundary (km)
    cos_bearing, sin_bearing: float
        cosine and sine of the bearing (heading − qdr) of the nearest point relative to the aircraft heading. (-)
    left_edge, right_edge: float
        angle between the heading and the left-most and right-most tangent of the polygon, (heading - qdr)/180 in [-1, 1]. (-)
        If the aircraft is inside the polygon the polygon spans all directions: left_edge = 1, right_edge = -1.
    """
    v = np.asarray(vertices, dtype=float).reshape(-1, 2)

    # bearing (0 is true north, values 0 to 360) [deg] and distance [NM] from the aircraft to each vertex
    qdr_vertices, dist_vertices = kwikqdrdist(ac_lat, ac_lon, v[:, 0], v[:, 1])
    # convert to local coordinates in NM
    x = dist_vertices * np.sin(np.radians(qdr_vertices)) # [NM]
    y = dist_vertices * np.cos(np.radians(qdr_vertices)) # [NM]
    # shift the vertices of one position to later make the segments
    xn, yn = np.roll(x, -1), np.roll(y, -1)

    # vectors between one vertex of the polygon and the next one
    ex, ey = xn - x, yn - y
    # squared length of each edge. 1e-12 is used to avoid division by zero in case of duplicate vertices (which should not happen in theory)
    seg_len2 = np.maximum(ex * ex + ey * ey, 1e-12)
    # projection of the aircraft position onto a line defined by each edge. The projection is clipped to the actual boundary of the edge, so t=0 means the projection is either vertex i or further, t=1 means the projection is either vertex i+1 or further.
    t = np.clip(-(x * ex + y * ey) / seg_len2, 0.0, 1.0)
    # coordinates of the nearest point (corresponding to the projection t) on each edge between one vertex of the polygon and the next one
    px, py = x + t * ex, y + t * ey
    # distance from the aircraft to each nearest point on the segments (either a vertex or on one of the edges)
    dist = np.hypot(px, py)
    # index of the edge on the polygon boundary containing the nearest point
    k = int(np.argmin(dist))
    # bearing to the nearest point from the aircraft (0 is true north, values 0 to 360) [deg]
    qdr_nearest = np.degrees(np.arctan2(px[k], py[k])) % 360.0

    # check if the aircraft is inside the polygon
    inside = bool(areafilter.checkInside(area_name, np.array([ac_lat]), np.array([ac_lon]), np.array([ac_alt]))[0])

    clearance_km = (-dist[k] if inside else dist[k]) * nm / 1000.   # NM -> km

    # the bearing of the nearest point relative to the heading
    bearing = degto180(ac_hdg - qdr_nearest) # relative to the aircraft heading, in [-180, 180]
    cos_bearing = float(np.cos(np.radians(bearing)))
    sin_bearing = float(np.sin(np.radians(bearing)))

    if inside:
        return float(clearance_km), cos_bearing, sin_bearing, 1.0, -1.0 # [km, -, -, -, -]

    # tangents: bearings of the vertices - bearing of the nearest point
    rel = degto180(qdr_vertices - qdr_nearest) # [deg, range -180, 180]
    qdr_left = qdr_nearest + rel.min()    # smallest bearing = left-most [deg]
    qdr_right = qdr_nearest + rel.max()   # largest bearing = right-most [deg]
    left_edge = degto180(ac_hdg - qdr_left) / 180.0 # normalized bearing to [-1, 1]
    right_edge = degto180(ac_hdg - qdr_right) / 180.0 # normalized bearing to [-1, 1]
    return float(clearance_km), cos_bearing, sin_bearing, float(left_edge), float(right_edge) # [km, -, -, -, -]
