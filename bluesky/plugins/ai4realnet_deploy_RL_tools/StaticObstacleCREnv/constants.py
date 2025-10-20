'''
Constants for the StaticObstacleCREnv-0 environment.
'''

import numpy as np

# Conversion factors
NM2KM = 1.852
MpS2Kt = 1.94384
FL2M = 30.48

# Model parameters
NUM_OBSTACLES = 5 #np.random.randint(1,5)
NUM_INTRUDERS = 5
AC_SPD = 150 # m/s
D_HEADING = 45 #degrees
D_SPEED = 20/3 # m/s

ACTION_FREQUENCY = 10
DISTANCE_MARGIN = 5 # km