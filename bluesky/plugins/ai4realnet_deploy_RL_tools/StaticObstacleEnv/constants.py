'''
Constants for the StaticObstacleEnv-0 environment.
'''

import numpy as np

# Conversion factors
NM2KM = 1.852
MpS2Kt = 1.94384
FL2M = 30.48


INTRUSION_DISTANCE = 5 # NM

# Model parameters
NUM_OBSTACLES = 10 #np.random.randint(1,5)
D_HEADING = 45 #degrees
D_SPEED = 20/3 # kts

ACTION_FREQUENCY = 10

CENTER = (52., 4.) # TU Delft AE Faculty coordinates

WAYPOINT_DISTANCE_MAX = 170 # KM
AC_SPD = 150 # kts
OBSTACLE_AREA_RANGE = (50, 1000) # In NM^2
