import itertools
import logging
from collections import defaultdict
from enum import Enum, IntEnum
from typing import List

import numpy as np

from config_deboarding import *


class State(IntEnum):
    UNDEFINED = 0
    SEATED = 1
    STAND_UP_FROM_SEAT = 2
    # todo: add collect luggage status
    MOVE_WAIT = 3
    MOVE_FROM_ROW = 4
    DISEMBARKED = 5


class SeatAllocation(Enum):
    RANDOM = 0
    CONNECTING_PRIORITY = 1


class Passenger:
    def __init__(self, seat_row, seat, slack_time, has_luggage):
        self.seat_row = seat_row
        self.seat = seat  # seat number (e.g. 1-3 for places on the right, negative numbers for places to the left)
        self.slack_time = slack_time  # = max deboarding time (in minutes) authorised for this pax before missing its next flight, set to np.inf if not connecting pax
        self.deboarding_time = -1
        self.has_luggage = has_luggage
        self.state = State.SEATED
        self.x = seat
        self.y = seat_row
        self.is_seated = True
        self.next_action_t = 0


class Simulation:
    def __init__(self, dummy_rows=2, quiet_mode=True):
        self.dummy_rows = dummy_rows  # Add fictive rows  to model empty space at the top of the aircraft
        self.passengers: List[Passenger] = []
        self.t = 0
        self.history = defaultdict(list)
        self.history_luggage = []
        self.seat_allocation = SeatAllocation.RANDOM
        self.quiet_mode = quiet_mode
        self.reset_stats()

    def set_custom_aircraft(self, n_rows, n_seats_left=2, n_seats_right=2):
        self.n_rows = 2 * n_rows
        self.n_seats_left = n_seats_left
        self.n_seats_right = n_seats_right

    def set_passengers_number(self, n):
        self.n_passengers = n

    def set_passengers_proportion(self, proportion):
        capacity = self.n_rows * (self.n_seats_left + self.n_seats_right) // 2
        self.n_passengers = int(proportion * capacity)

    def set_seat_allocation(self, seat_allocation):
        self.seat_allocation = seat_allocation

    def reset_stats(self):
        self.deboarding_time = []

    def print(self):
        for i in range(self.n_rows + self.dummy_rows):
            row = list(self.side_left[i, :][::-1]) + ['|', '[' + str(self.luggage_bin[i][0]) + ']', self.aisle[i],
                                                      '[' + str(self.luggage_bin[i][1]) + ']', '|'] + list(
                self.side_right[i, :])
            self.print_info(row)

    def print_deboarding_order(self):
        for i in range(self.dummy_rows, self.n_rows + self.dummy_rows):
            row = list(self.deboarding_order_left[i, :][::-1]) + [' '] + list(self.deboarding_order_right[i, :])
            print(row)

    def reset(self):
        self.t = 0
        self.history = defaultdict(list)
        self.history_luggage = []

        self.side_left = np.zeros((self.n_rows * 2 + self.dummy_rows, self.n_seats_left), dtype=int)
        self.side_right = np.zeros((self.n_rows * 2 + self.dummy_rows, self.n_seats_right), dtype=int)
        self.aisle = np.zeros(self.n_rows * 2 + self.dummy_rows, dtype=int)
        self.luggage_bin = np.zeros((self.n_rows * 2 + self.dummy_rows, 2), dtype=int)

        self.deboarding_order_left = np.zeros((self.n_rows * 2 + self.dummy_rows, self.n_seats_left), dtype=int)
        self.deboarding_order_right = np.zeros((self.n_rows * 2 + self.dummy_rows, self.n_seats_right), dtype=int)

        self.randomize_passengers()

    def randomize_passengers(self):
        seat_cols = set(range(-self.n_seats_left, self.n_seats_right + 1)) - {0}  # Possible seats
        seat_rows = np.array(range(self.dummy_rows, self.n_rows + self.dummy_rows)) * 2  # Possible rows

        # 1. Get all seats on the plane
        #    Every seat is described by a 3-element list: [row, column, boarding zone]
        #    Initially we zet all zones to 0, and we set the actual values later
        # 2. Randomly select seat indices for every passenger
        # 3. Create seat's list (i-th seat corresponds to the )
        all_seats = list(list(x) for x in itertools.product(seat_rows, seat_cols))
        all_seats.sort(key=lambda x: 100 * x[0] + abs(x[1]))
        selected_seats_ind = np.arange(self.n_passengers)
        # selected_seats_ind = np.random.choice(len(all_seats), size=self.n_passengers, replace=False)
        selected_seats = [all_seats[seat_ind] for seat_ind in selected_seats_ind]

        # connecting_times = np.random.randint(0,30,len(selected_seats))
        connecting_times = np.arange(0, len(selected_seats))
        np.random.shuffle(connecting_times)
        if self.seat_allocation == SeatAllocation.CONNECTING_PRIORITY:
            connecting_times.sort()  # todo: implementer strategie allocation pax priority, add selected seat unassignable and first rows for business for instance

        # Add a dummy element so that passengers are 1-indexed. We do this so that 0 in self.aisle,  self.side_left etc. represents "no passenger"
        self.passengers = [None]
        for i, seat in enumerate(selected_seats):
            has_bagage = np.random.choice([True, True, False])
            self.passengers.append(
                Passenger(seat_row=seat[0], seat=seat[1], slack_time=connecting_times[i], has_luggage=has_bagage))
            if (seat[1] < 0):
                self.side_left[seat[0]][N_SEAT_LEFT + seat[1]] = i + 1
            else:
                self.side_right[seat[0]][seat[1] - 1] = i + 1

    def print_info(self, *args):
        if not self.quiet_mode:
            print(*args)

    # Run multiple simulations
    def run_multiple(self, n):
        self.reset_stats()
        for i in range(n):
            self.run()

    # Run a single simulation
    def run(self):
        self.reset()
        deboarding_completed = False
        while (self.t < 1000) & (not deboarding_completed):
            self.print_info(f'\n*** Step {self.t}')
            deboarding_completed = self.step()
            if not self.quiet_mode:
                self.print()

            self.t += 1
        print(
            f"Total minutes to disembark all pax: {BUFFER_TIME_GATE_CONNECTING + round(self.t * TIME_STEP_DURATION / 60, 2)}min")
        # Update stats
        self.deboarding_time.append(self.t)

    # Process a single animations step.

    def step(self):
        # Process passengers.
        # This basically iterates over all the passengers, and performs appropriate actions based on their state.
        deboarded_pax = 0  # Number of passengers already deboarded.
        still_pax_sitted = 0

        for i, p in enumerate(self.passengers):
            if i == 0: continue

            if p.state != State.DISEMBARKED:
                self.history[self.t].append([self.t, i, p.x, p.y, int(p.state)])

            if p.next_action_t > self.t:
                continue
            match p.state:
                case State.SEATED:
                    # If the first space in the aisle is empty, move there.
                    r1 = np.random.randint(0, 10000)
                    r2 = np.random.randint(0, 2)
                    #
                    if p.x == 1:
                        if self.aisle[p.y] == 0:
                            if self.side_left[p.y][2] != 0 and (r2 > 0):
                                p.next_action_t = self.t + 1
                            else:
                                if IS_COURTESY_RULE or (p.y + 1 == len(self.aisle)) or (self.aisle[p.y + 1] == 0):
                                    self.side_right[p.y][0] = 0
                                    p.x = 0
                                    self.aisle[p.y] = i
                                    p.state = State.STAND_UP_FROM_SEAT
                                    p.next_action_t = self.t + STAND_UP_DURATION
                                else:
                                    p.next_action_t = self.t + 1
                        else:
                            p.next_action_t = self.t + 1
                    if p.x == -1:
                        if self.aisle[p.y] == 0:
                            self.side_left[p.y][N_SEAT_LEFT - 1] = 0
                            p.x = 0
                            self.aisle[p.y] = i
                            p.state = State.STAND_UP_FROM_SEAT
                            p.next_action_t = self.t + STAND_UP_DURATION
                        else:
                            p.next_action_t = self.t + 1
                    elif p.x < -1:
                        if self.side_left[p.seat_row][N_SEAT_LEFT + p.x + 1] == 0:
                            self.side_left[p.seat_row][N_SEAT_LEFT + p.x] = 0
                            p.x += 1
                            self.side_left[p.seat_row][N_SEAT_LEFT + p.x] = i
                            p.next_action_t = self.t + MOVE_SEAT_DURATION
                        else:
                            p.next_action_t = self.t + 1
                    elif p.x > 1:
                        if self.side_right[p.seat_row][p.x - 2] == 0:
                            self.side_right[p.seat_row][p.x - 1] = 0
                            p.x -= 1
                            self.side_right[p.seat_row][p.x - 1] = i
                            p.next_action_t = self.t + MOVE_SEAT_DURATION
                        else:
                            p.next_action_t = self.t + 1
                    still_pax_sitted += 1
                case State.STAND_UP_FROM_SEAT:
                    if p.has_luggage:
                        p.state = State.MOVE_WAIT
                        p.next_action_t = self.t + COLLECT_BAGGAGE_DURATION
                        ind = 0 if p.seat < 0 else 1
                        self.luggage_bin[p.seat_row][ind] += 1
                        self.history_luggage.append([self.t, p.seat_row, ind])
                    elif self.aisle[p.y - 1] != 0:
                        p.state = State.MOVE_WAIT
                        p.next_action_t = self.t + 1
                    else:
                        # We can go!
                        p.next_action_t = self.t + WALK_DURATION
                        p.state = State.MOVE_FROM_ROW
                        self.aisle[p.y] = 0
                        p.y -= 1
                        self.aisle[p.y] = i

                case State.MOVE_WAIT:
                    if self.aisle[p.y - 1] != 0:
                        continue

                    # We can go!
                    p.next_action_t = self.t + WALK_DURATION
                    p.state = State.MOVE_FROM_ROW
                    self.aisle[p.y] = 0
                    p.y -= 1
                    self.aisle[p.y] = i
                case State.MOVE_FROM_ROW:
                    if p.y == 0:
                        p.state = State.DISEMBARKED
                        self.aisle[p.y] = 0
                        p.deboarding_time = (self.t * TIME_STEP_DURATION) // 60 + BUFFER_TIME_GATE_CONNECTING
                        deboarded_pax += 1
                    else:
                        if self.aisle[p.y - 1] != 0:
                            p.state = State.MOVE_WAIT
                        else:
                            p.next_action_t = self.t + WALK_DURATION
                            self.aisle[p.y] = 0
                            p.y -= 1
                            self.aisle[p.y] = i
                case State.DISEMBARKED:
                    deboarded_pax += 1

                case _:
                    self.print_info(f'State {p.state} is not handled.')

        # Check whether everyone is already deboarded
        # print(f"% deboarded pax: {100 * round(deboarded_pax / self.n_passengers, 2)}")
        # print(f"paxSitted: {still_pax_sitted}")
        return deboarded_pax == self.n_passengers

    # Save boarding history to a file.
    def serialize_history(self, path):
        with open(path, 'w') as f:
            # General parameters in the header.
            f.write(
                f'{self.n_rows} {self.dummy_rows} {self.n_seats_left} {self.n_seats_right} {self.n_passengers} {len(self.history_luggage)}\n')

            # Save passengers' history.
            for id, h in self.history.items():
                f.write(f'{len(h)}\n')
                for entry in h:
                    f.write(' '.join(map(str, entry)) + '\n')

            # # Save luggage history.
            # for entry in self.history_luggage:
            #     f.write(' '.join(map(str, entry)) + '\n')

    def evaluate_missing_pax(self):
        "iterate over pax and test if deboarding time > slack time"
        nb_missed_pax = 0
        nb_deboarded_pax = 0
        # print(self.passengers)
        for i, passenger in enumerate(self.passengers):
            if i == 0: continue

            if passenger.deboarding_time < 0:
                logging.warning(f"Error when computing deboarding time of passenger {i}")
            else:
                if (passenger.deboarding_time > passenger.slack_time):
                    nb_missed_pax += 1
                nb_deboarded_pax += 1
        print(f"Total missed pax: {nb_missed_pax}")
        print(f"Percentage missed pax: {round(100 * nb_missed_pax / nb_deboarded_pax, 2)}%")


if __name__ == "__main__":
    nb_simu = 1
    passengers_proportion = 0.8
    # seat_allocation = SeatAllocation.RANDOM
    # seat_allocation = SeatAllocation.CONNECTING_PRIORITY
    for seat_allocation in [SeatAllocation.RANDOM, SeatAllocation.CONNECTING_PRIORITY]:
        simulation = Simulation(quiet_mode=True, dummy_rows=2)

        simulation.set_custom_aircraft(n_rows=30, n_seats_left=3, n_seats_right=3)
        simulation.set_passengers_proportion(0.9)

        simulation.set_passengers_proportion(passengers_proportion)
        simulation.set_seat_allocation(seat_allocation)
        simulation.run_multiple(nb_simu)
        simulation.evaluate_missing_pax()

# todo : utiliser loi weibull avec param papier schultz pour simuler collecting luggage
