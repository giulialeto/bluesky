'''
Constants for the StaticObstacleSectorEnv-0 environment.
'''

import numpy as np

# Conversion factors
NM2KM = 1.852
MpS2Kt = 1.94384
FL2M = 30.48

# Model parameters
NUM_OBSTACLES = 10 #np.random.randint(1,5)
AC_SPD = 150 # m/s
D_HEADING = 45 #degrees
D_SPEED = 20/3 # m/s
MACH_CRUISE = 0.8 # -

ACTION_FREQUENCY = 10

TOTAL_OBSERVATION_POINTS = 50 # Number of points to be observed along the sector polygon edges