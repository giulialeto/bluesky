'''
Functions for the CentralisedStaticObstacleSectorCREnv-0 environment.
'''

import numpy as np
from pygame.gfxdraw import polygon
import bluesky as bs
from math import radians, sin, cos, sqrt, atan2
from statistics import mean
from typing import List
import networkx as nx
from shapely.geometry import Polygon
from plugins.SingleAgentCRTools import functions

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
        point to be converted. point[0] is the x (east) coordinate in nm, point[1] is the y (north) coordinate in nm.
    
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
        center point of the conversion in lat/long coordinates
    point: np.array
        point to be converted in lat/long coordinates
    
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

def build_graph_from_edges(edges):
    """
    Build a NetworkX graph from your self.edges list.

    Each edge is expected to be:
        (u, v, weight, dist_nm)
    where u and v are (lat, lon) tuples.

    Args:
        edges (list[tuple]): Edge list.

    Returns:
        G (nx.Graph): Weighted graph.
    """
    G = nx.Graph()

    for u, v, w, dist_nm in edges:
        u = (float(u[0]), float(u[1]))
        v = (float(v[0]), float(v[1]))

        # Choose your edge cost:
        # - If you want "shortest in NM": cost = dist_nm
        # - If you want to include your 'w' factor: cost = w * dist_nm
        cost = float(dist_nm)

        # Add node positions (helps debugging / later plotting)
        G.add_node(u, lat=u[0], lon=u[1])
        G.add_node(v, lat=v[0], lon=v[1])

        # If multiple edges repeat, keep the smallest cost
        if G.has_edge(u, v):
            if cost < G[u][v]["weight"]:
                G[u][v]["weight"] = cost
                G[u][v]["dist_nm"] = float(dist_nm)
        else:
            G.add_edge(u, v, weight=cost, dist_nm=float(dist_nm), base_w=float(w))

    return G

def astar_route(G, start, goal):
    """
    Run A* on the graph.

    Args:
        G (nx.Graph): Graph with edge attribute "weight".
        start (tuple[float,float]): (lat, lon).
        goal (tuple[float,float]): (lat, lon).

    Returns:
        path (list[tuple]): List of (lat, lon) nodes along the route.
    """
    start = (float(start[0]), float(start[1]))
    goal = (float(goal[0]), float(goal[1]))

    def heuristic(a, b):
        return haversine_m(a[0], a[1], b[0], b[1])* 0.539956803 # Convert meters to NM

    return nx.astar_path(G, start, goal, heuristic=heuristic, weight="weight")

def buffer_util_polygon_centroid_latlon(polygon):
    """
    Calculate centroid in lat/lon by averaging vertices.

    Args:
        polygon (list[tuple[float, float]]): List of (lat, lon) vertices.

    Returns:
        centroid (tuple[float, float]): (lat, lon)
    """
    arr = np.asarray(polygon, dtype=float)
    return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))

def buffer_shapely(polygon, buffer_nm):
    """
    Approximate polygon buffer by pushing each vertex radially outward
    from the polygon centroid.

    Args:
        polygon (list[tuple[float, float]]): List of (lat, lon) vertices.
        buffer_nm (float): Buffer distance [NM].

    Returns:
        buffered_polygon (list[tuple[float, float]]): Buffered vertices.
    """

    c_lat, c_lon = buffer_util_polygon_centroid_latlon(polygon)
    centre = np.array([c_lat, c_lon])
    polygon_nm = [latlong_to_nm(centre, vertex) for vertex in polygon]

    # print(f"[buffer_shapely] input has {len(polygon)} vertices: {polygon}")
    # print(f"[buffer_shapely] polygon_nm has {len(polygon_nm)} points: {polygon_nm}")


    # Close the ring if not already closed (needed for triangles / Shapely >= 2.0)
    if not np.allclose(polygon_nm[0], polygon_nm[-1]):
        polygon_nm.append(polygon_nm[0])

    # print(f"[buffer_shapely] closed polygon_nm has {len(polygon_nm)} points: {polygon_nm}")

    poly_shapely = Polygon(polygon_nm)
    poly_buffer = poly_shapely.buffer(buffer_nm, join_style=2)


    p_buffer = list(poly_buffer.exterior.coords)

    p_buffer = [functions.nm_to_latlong(centre, point) for point in p_buffer] # Convert to lat/long coordinates

    return p_buffer

def buffer_obstacles_nm(obstacles, buffer_nm):
    """
    Apply approximate radial buffer to all obstacle polygons.

    Args:
        obstacles (list[list[tuple[float, float]]]): Obstacle polygons.
        buffer_nm (float): Buffer distance [NM].

    Returns:
        buffered_obstacles (list[list[tuple[float, float]]]): Buffered polygons.
    """
    return [buffer_shapely(poly, buffer_nm) for poly in obstacles]


def closest_point_on_segment(P, A, B):
    """
    Compute closest point from P to segment AB.

    Args:
        P (np.ndarray): Point.
        A (np.ndarray): Segment start.
        B (np.ndarray): Segment end.

    Returns:
        Q (np.ndarray): Closest point on segment.
        dist2 (float): Squared distance.
    """

    AB = B - A
    AP = P - A

    denom = np.dot(AB, AB)

    # Degenerate segment
    if denom == 0:
        Q = A
        return Q, np.dot(P - Q, P - Q)

    t = np.dot(AP, AB) / denom

    # Clamp to segment
    t = np.clip(t, 0.0, 1.0)

    Q = A + t * AB

    dist2 = np.dot(P - Q, P - Q)

    return Q, dist2


def closest_point_on_polygon(aircraft, polygon, safety_margin):

    """

    Find closest point on a closed polygon boundary to an aircraft.

    Args:

        aircraft (tuple): Aircraft position as (lat, lon).

        polygon (list[tuple]): Closed polygon vertices as [(lat, lon), ...].

            The first and last point may be the same.

    Returns:

        closest_point_nm (np.ndarray): Closest boundary point in nautical-mile coordinates.

        closest_distance_nm (float): Distance from aircraft to boundary in nautical miles.

    """

    c_lat, c_lon = buffer_util_polygon_centroid_latlon(polygon)
    centre = np.array([c_lat, c_lon])

    polygon_nm = [latlong_to_nm(centre, vertex) for vertex in polygon]

    aircraft_nm = latlong_to_nm(centre, aircraft)

    polygon_nm = [np.asarray(p, dtype=float) for p in polygon_nm]

    aircraft_nm = np.asarray(aircraft_nm, dtype=float)

    best_Q = None

    best_dist2 = np.inf

    # Since polygon is closed, avoid checking the duplicate final point twice

    n = len(polygon_nm)

    for i in range(n - 1):

        A = polygon_nm[i]

        B = polygon_nm[i + 1]

        Q, dist2 = closest_point_on_segment(aircraft_nm, A, B)

        if dist2 < best_dist2:

            best_Q = Q

            best_dist2 = dist2

    best_Q_latlon = nm_to_latlong(centre, best_Q*(1+safety_margin)) # Add a small margin to ensure the point is outside the obstacle

    return best_Q_latlon, np.sqrt(best_dist2)