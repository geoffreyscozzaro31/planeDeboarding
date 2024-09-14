'''
Configuration file including parameters used to tune the 10_simulations_40_pct_prereserved_3h_connecting_time_without_gate_closure_time
'''

import numpy as np

BUFFER_TIME_GATE_CONNECTING = 3 * 60  # in seconds

TIME_STEP_DURATION = 0.5  # in seconds

WALK_DURATION = 1  # in time steps
STAND_UP_DURATION = 1 # in time steps
MOVE_SEAT_DURATION = 1

###luggage collection param
ALPHA_WEIBULL = 1.7
BETA_WEIBULL = 16.0

# aircraft configuration
NB_SEAT_LEFT = 3
NB_SEAT_RIGHT = 3 # todo: check if well implemented

# NB_ROWS = int(np.ceil(144/6))
NB_ROWS = int(np.ceil(182 / 6))

# passenger characteristics
MIN_PERCENTAGE_HAS_LUGGAGE = 90 - 5
MAX_PERCENTAGE_HAS_LUGGAGE = 90 + 5  # add stochasticity to percentage luggage

MIN_LOAD_FACTOR = 0.865 - 0.05  # value between 0 and 1
MAX_LOAD_FACTOR = 0.865 + 0.05  # add stochasticity to the load factor

MIN_PERCENTAGE_PRERESERVED_SEATS = 40 - 5
MAX_PERCENTAGE_PRERESERVED_SEATS = 40 + 5  # add stochasticity to the pre-reserved seats


### Simulation parameters
T_MAX_SIMULATION = 10000  # in time steps
NB_SIMULATION = 10

#
# IS_COURTESY_RULE = False
# if IS_COURTESY_RULE:
#     DISEMBARKING_RULE_NAME = "courtesy"
# else:
#     DISEMBARKING_RULE_NAME = "aisle_priority"
#

GATE_CLOSE_TIME = 20 * 60  # in seconds