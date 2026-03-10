import numpy as np
from bluesky import core, stack, traf, tools, settings 
from bluesky.tools.aero import kts, ft

# (min_alt [ft], max_alt [ft], nominal_speed [kts, TAS], lower_speed [kts, TAS], upper_speed [kts, TAS])
speed_table = [
    (0, 500, 142, 142*0.9, 142*1.1),
    (500, 1000, 143, 143*0.9, 143*1.1),
    (1000, 1500, 149, 149*0.9, 149*1.1),
    (1500, 2000, 160, 160*0.9, 160*1.1),
    (2000, 3000, 192, 192*0.9, 192*1.1),
    (3000, 4000, 230, 230*0.9, 230*1.1),
    (4000, 6000, 233, 233*0.9, 233*1.1),
    (6000, 8000, 240, 240*0.9, 240*1.1),
    (8000, 10000, 280, 280*0.9, 280*1.1),
    (10000, 12000, 289, 289*0.9, 289*1.1),
    (12000, 14000, 356, 356*0.9, 356*1.1),
    (14000, 16000, 366, 366*0.9, 366*1.1),
    (16000, 18000, 377, 377*0.9, 377*1.1),
    (18000, 20000, 388, 388*0.9, 388*1.1),
    (20000, 22000, 400, 400*0.9, 400*1.1),
    (22000, 24000, 412, 412*0.9, 412*1.1),
    (24000, 26000, 425, 425*0.9, 425*1.1),
    (26000, 28000, 438, 438*0.9, 438*1.1),
    (28000, 30000, 452, 452*0.9, 452*1.1),
    (30000, 32000, 459, 459*0.9, 459*1.1),
    (32000, 34000, 455, 455*0.9, 455*1.1),
    (34000, 36000, 451, 451*0.9, 451*1.1),
    (36000, 38000, 447, 447*0.9, 447*1.1),
    (38000, 40000, 447, 447*0.9, 447*1.1),
    (40000, 1000000, 447, 447*0.9, 447*1.1),
]

def get_speed_at_altitude(alt, speed=None):
    """
    returns a truncated speed based on the performance envelope in speed_table

    input: alt[m], speed[m/s, CAS]
    output: speed[m/s, CAS]
    """
    for altmin, altmax, speednom, speedmin, speedmax in speed_table:
        if altmin*ft <= alt < altmax*ft:
            if speed==None:
                return tools.aero.vtas2cas(speednom*kts,alt)
            else:
                return max(tools.aero.vtas2cas(speedmin*kts, alt), min(speed, tools.aero.vtas2cas(speedmax*kts,alt)))

def get_speed_limits_at_altitude(alt):
    """
    returns the speedlimits for a given altitude in m/s CAS

    input: alt[m], speed[m/s, CAS]
    output: speed_min, speed_max[m/s, CAS]
    """
    for altmin, altmax, speednom, speedmin, speedmax in speed_table:
        if altmin*ft <= alt < altmax*ft:
            return tools.aero.vtas2cas(speedmin*kts,alt), tools.aero.vtas2cas(speedmax*kts,alt)
    # return 142*0.9, 447*1.1

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