from scipy.stats import weibull_min

from model.passenger_state import PassengerState
from config_deboarding import ALPHA_WEIBULL,BETA_WEIBULL,TIME_STEP_DURATION


import numpy as np

class Passenger:
    def __init__(self, seat_row, seat, has_luggage, has_pre_reserved_seat):
        self.seat_row = seat_row
        self.seat = seat  # seat number (e.g. 1-3 for places on the right, negative numbers for places to the left)
        self.actual_slack_time = -1  # actual remaining time  for this pax for his transfer, set to np.inf if not connecting pax
        self.scheduled_slack_time = -1  # schedule remaining time for this pax for his transfer, set to np.inf if not connecting pax
        self.deboarding_time = -1
        self.has_luggage = has_luggage
        self.state = PassengerState.SEATED
        self.x = seat
        self.y = seat_row
        self.is_seated = True
        self.next_action_t = 0
        self.has_pre_reserved_seat = has_pre_reserved_seat
        if self.has_luggage:
            time_w = weibull_min.rvs(ALPHA_WEIBULL, scale=BETA_WEIBULL) / 2
            self.collecting_luggage_time = int(np.ceil(time_w / TIME_STEP_DURATION))
        else:
            self.collecting_luggage_time = 0

