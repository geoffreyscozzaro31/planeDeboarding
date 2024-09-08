'''
Configuration file including parameters used to tune the simulation
'''

import numpy as np

BUFFER_TIME_GATE_CONNECTING = 0 * 60  # in seconds

TIME_STEP_DURATION = 0.5  # in seconds

WALK_DURATION = 1  # in time steps
STAND_UP_DURATION = 1  # in time steps
MOVE_SEAT_DURATION = 1

###luggage collection param
ALPHA_WEIBULL = 1.7
BETA_WEIBULL = 16.0

# aircraft configuration
NB_SEAT_LEFT = 3

# NB_ROWS = int(np.ceil(144/6))
NB_ROWS = int(np.ceil(182 / 6))

# passenger characteristics
PERCENTAGE_HAS_LUGGAGE = 100
LOAD_FACTOR = 1.0  # value between 0 and 1

IS_COURTESY_RULE = False

### Simulation parameters
T_MAX_SIMULATION = 10000  # in time steps
NB_SIMULATION = 100

if IS_COURTESY_RULE:
    DISEMBARKING_RULE_NAME = "courtesy"
else:
    DISEMBARKING_RULE_NAME = "aisle_priority"
