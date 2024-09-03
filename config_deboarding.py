'''
Configuration file including parameters used to tune the simulation
'''
BUFFER_TIME_GATE_CONNECTING = 5  # in minutes

TIME_STEP_DURATION = 2  # in seconds

WALK_DURATION = 1 # in time steps
STAND_UP_DURATION = 2  # in time steps
COLLECT_BAGGAGE_DURATION = 3  # in time steps
MOVE_SEAT_DURATION = 2

MAX_TIME = 24 * 60 * 60

N_SEAT_LEFT = 3


IS_COURTESY_RULE  = False

if IS_COURTESY_RULE:
    DISEMBARKING_RULE_NAME = "courtesy"
else:
    DISEMBARKING_RULE_NAME = "aisle_priority"
