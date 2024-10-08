"""
This script is used to simulate the deboarding of passengers from an aircraft with a single aisle cabin.
"""

import logging
from collections import defaultdict
from typing import List

import numpy as np

import prereserved_seats
from model.deboarding_strategy import DeboardingStrategy
from model.passenger import Passenger
from model.passenger_state import PassengerState
from model.seat_allocation import SeatAllocation
from utils import configloader


configuration = configloader.ConfigLoader("configuration_deboarding.yaml")


class DeboardingSimulation:
    def __init__(self, quiet_mode=True):
        self.dummy_rows = configuration.nb_dummy_rows  # Add fictive rows  to model empty space at the top of the aircraft
        self.nb_rows = -1
        self.n_fictive_rows = -1  # depends on the number of rows and the dummy rows and cell size
        self.n_seats_left = -1
        self.n_seats_right = -1

        self.passengers: List[Passenger] = []
        self.t = 0
        self.history = defaultdict(list)
        self.history_luggage = []
        self.seat_allocation_strategy = None
        self.deboarding_strategy = None
        self.quiet_mode = quiet_mode
        self.reset_stats()
        self.t_max = configuration.t_max_simulation
        self.disembarked_pax = 0
        self.actual_connecting_time_pax_list = []
        self.scheduled_connecting_time_pax_list = []
        self.passengers_seat_allocation_list = []
        self.probability_matrix_prereserved_seats = []
        self.passenger_has_luggage_list = []
        self.passenger_has_pre_reserved_seat_list = []
        self.nb_prereserved_seats = 0
        self.disembarkation_time = -1

    def prepare_simulation(self, nb_rows, nb_pax_carried, connecting_pax_df, percentage_prereserved_seats=-1,
                           activate_connecting_pax=True):
        self.set_custom_aircraft(n_rows=nb_rows, n_seats_left=configuration.nb_seat_left,
                                 n_seats_right=configuration.nb_seat_right)
        self.set_number_passengers(nb_pax_carried)

        self.set_connecting_time_pax(connecting_pax_df, nb_pax_carried)
        self.set_passengers_having_luggage()
        self.set_probability_matrix_prereserved_seats()
        self.set_passengers_with_prereserved_seats(percentage_prereserved_seats)
        self.set_passenger_seat_allocation()
        self.create_passengers()
        if activate_connecting_pax:
            self.sort_passengers()

    def randomize_initialisation(self, activate_connecting_pax=True):
        self.set_passengers_with_prereserved_seats()
        self.set_passengers_having_luggage()
        self.set_passenger_seat_allocation()
        self.create_passengers()
        if activate_connecting_pax:
            self.sort_passengers()

    def set_connecting_time_pax(self, df_connecting_flight, nb_pax_carried, default_transfer_time_seconds=-1):
        if len(df_connecting_flight) == 0:
            print("warning: empty connecting flight dataframe")
        else:
            nb_pax_list = df_connecting_flight['nb_connecting_pax'].values
            list_connecting_pax_lists = []
            for label in ["theoretical", "actual"]:
                buffer_time_list = df_connecting_flight[f'buffer_time_{label}_seconds'].values
                if default_transfer_time_seconds >= 0:
                    buffer_time_list = np.full(len(buffer_time_list), default_transfer_time_seconds)
                connecting_pax_list = np.repeat(buffer_time_list, nb_pax_list)
                nb_connecting_pax = len(connecting_pax_list)
                remaining_pax = nb_pax_carried - len(connecting_pax_list)
                connecting_pax_list = np.concatenate((connecting_pax_list, np.full(remaining_pax, np.inf)))
                list_connecting_pax_lists.append(connecting_pax_list)

            scheduled_connecting_pax_list, actual_connecting_pax_list = list_connecting_pax_lists
            indexes = np.arange(len(scheduled_connecting_pax_list))
            np.random.shuffle(indexes)

            self.scheduled_connecting_time_pax_list = scheduled_connecting_pax_list[indexes]
            self.actual_connecting_time_pax_list = actual_connecting_pax_list[indexes]
            print(f"nb_connecting pax:{nb_connecting_pax},  nb_other_pax:{remaining_pax}")

    def set_custom_aircraft(self, n_rows, n_seats_left=2, n_seats_right=2):
        self.nb_rows = n_rows
        self.n_fictive_rows = 2 * n_rows
        self.n_seats_left = n_seats_left
        self.n_seats_right = n_seats_right

    def set_passengers_number(self, n):
        self.n_passengers = n

    def set_default_passengers_proportion(self, proportion):
        capacity = self.n_fictive_rows * (self.n_seats_left + self.n_seats_right) // 2
        self.n_passengers = int(proportion * capacity)

    def set_number_passengers(self, n):
        self.n_passengers = n

    def set_seat_allocation_strategy(self, seat_allocation_strategy):
        self.seat_allocation_strategy = seat_allocation_strategy

    def set_deboarding_strategy(self, deboarding_strategy):
        self.deboarding_strategy = deboarding_strategy

    def reset_stats(self):
        self.disembarkation_time = -1

    def reset_passengers(self):
        for p in self.passengers:
            if p != None:
                p.state = PassengerState.SEATED
                p.deboarding_time = -1
                p.next_action_t = 0
                p.x = p.seat
                p.y = p.seat_row

    def print(self):
        for i in range(self.n_fictive_rows + self.dummy_rows):
            row = list(self.side_left[i, :][::-1]) + ['|', '[' + str(self.luggage_bin[i][0]) + ']', self.aisle[i],
                                                      '[' + str(self.luggage_bin[i][1]) + ']', '|'] + list(
                self.side_right[i, :])
            self.print_info(row)

    def print_deboarding_order(self):
        for i in range(self.dummy_rows, self.n_fictive_rows + self.dummy_rows):
            row = list(self.deboarding_order_left[i, :][::-1]) + [' '] + list(self.deboarding_order_right[i, :])
            print(row)

    def reset(self):
        self.t = 0
        self.history = defaultdict(list)
        self.history_luggage = []
        self.disembarked_pax = 0
        self.side_left = np.zeros((self.n_fictive_rows * 2 + self.dummy_rows, self.n_seats_left), dtype=int)
        self.side_right = np.zeros((self.n_fictive_rows * 2 + self.dummy_rows, self.n_seats_right), dtype=int)
        self.exit_corridor = np.zeros(self.n_seats_left, dtype=int)
        self.aisle = np.zeros(self.n_fictive_rows * 2 + self.dummy_rows, dtype=int)
        self.luggage_bin = np.zeros((self.n_fictive_rows * 2 + self.dummy_rows, 2), dtype=int)
        self.deboarding_order_left = np.zeros((self.n_fictive_rows * 2 + self.dummy_rows, self.n_seats_left), dtype=int)
        self.deboarding_order_right = np.zeros((self.n_fictive_rows * 2 + self.dummy_rows, self.n_seats_right),
                                               dtype=int)
        self.fill_aircraft()
        self.reset_stats()
        self.reset_passengers()

    def fill_aircraft(self):
        for i, passenger in enumerate(self.passengers[1:]):
            if (passenger.seat < 0):
                self.side_left[passenger.seat_row][configuration.nb_seat_left + passenger.seat] = i + 1
            else:
                self.side_right[passenger.seat_row][passenger.seat - 1] = i + 1

    def set_passengers_having_luggage(self):
        stochastic_percentage_has_luggage = np.random.uniform(configuration.min_percentage_has_luggage,
                                                              configuration.max_percentage_has_luggage)
        self.passenger_has_luggage_list = np.random.rand(self.n_passengers) < stochastic_percentage_has_luggage / 100

    def set_probability_matrix_prereserved_seats(self):
        self.probability_matrix_prereserved_seats = prereserved_seats.generate_probability_matrix(self.nb_rows,
                                                                                                  configuration.nb_seat_left + configuration.nb_seat_right)

    def set_passengers_with_prereserved_seats(self, percentage_prereserved_seats=-1):
        if percentage_prereserved_seats == -1:
            stochastic_percentage_prereserved_seats = np.random.uniform(configuration.min_percentage_prereserved_seats,
                                                                        configuration.max_percentage_prereserved_seats)
            self.passenger_has_pre_reserved_seat_list = np.random.rand(
                self.n_passengers) < stochastic_percentage_prereserved_seats / 100
        else:
            self.passenger_has_pre_reserved_seat_list = np.random.rand(
                self.n_passengers) < percentage_prereserved_seats / 100
        self.nb_prereserved_seats = np.sum(self.passenger_has_pre_reserved_seat_list)

    def set_passenger_seat_allocation(self):
        prereserved_seats_pax_allocation, other_seats = prereserved_seats.assign_seats(
            self.nb_prereserved_seats, self.n_passengers, self.probability_matrix_prereserved_seats)

        prereserved_index = 0
        other_index = 0
        self.passengers_seat_allocation_list = []
        for i in range(self.n_passengers):
            if self.passenger_has_pre_reserved_seat_list[i]:
                self.passengers_seat_allocation_list.append(prereserved_seats_pax_allocation[prereserved_index])
                prereserved_index += 1
            else:
                self.passengers_seat_allocation_list.append(other_seats[other_index])
                other_index += 1

    def create_passengers(self):
        self.passengers = [None]
        for i, seat in enumerate(self.passengers_seat_allocation_list):
            seat_row = (self.dummy_rows + seat[0]) * 2
            if (seat[1] < 3):
                seat_column = seat[1] - configuration.nb_seat_left
            else:
                seat_column = seat[1] - configuration.nb_seat_left + 1
            self.passengers.append(Passenger(seat_row=seat_row, seat=seat_column,
                                             has_luggage=self.passenger_has_luggage_list[i],
                                             has_pre_reserved_seat=self.passenger_has_pre_reserved_seat_list[i]))

    def sort_passengers(self):
        combined_list = list(
            zip(self.passengers[1:], self.passenger_has_pre_reserved_seat_list, self.scheduled_connecting_time_pax_list,
                self.actual_connecting_time_pax_list))
        combined_list.sort(key=lambda p: (p[0].seat_row, abs(p[0].seat)))
        self.passengers, self.passenger_has_pre_reserved_seat_list, self.scheduled_connecting_time_pax_list, self.actual_connecting_time_pax_list = zip(
            *combined_list)
        self.passengers = [None] + list(self.passengers)
        self.passenger_has_pre_reserved_seat_list = list(self.passenger_has_pre_reserved_seat_list)
        self.scheduled_connecting_time_pax_list = list(self.scheduled_connecting_time_pax_list)
        self.actual_connecting_time_pax_list = list(self.actual_connecting_time_pax_list)

    def assign_passenger_connecting_times(self, activate_connecting_pax=True):
        if activate_connecting_pax:
            for i in range(len(self.passengers) - 1):
                self.passengers[i + 1].actual_slack_time = self.actual_connecting_time_pax_list[i]
                self.passengers[i + 1].scheduled_slack_time = self.actual_connecting_time_pax_list[i]
            if self.seat_allocation_strategy != SeatAllocation.RANDOM:
                non_pre_reserved_passengers = [p for p in self.passengers[1:] if not p.has_pre_reserved_seat]

                if self.deboarding_strategy == DeboardingStrategy.COURTESY_RULE:
                    non_pre_reserved_passengers.sort(key=lambda p: (p.seat_row, abs(p.seat)))

                elif self.deboarding_strategy == DeboardingStrategy.AISLE_PRIORITY_RULE:
                    non_pre_reserved_passengers.sort(key=lambda p: (abs(p.seat), p.seat_row))

                else:
                    raise ValueError("Unknown seat allocation strategy")

                other_passengers_scheduled_deboarding_time = [p.scheduled_slack_time for p in
                                                              non_pre_reserved_passengers]
                other_passengers_actual_deboarding_time = [p.actual_slack_time for p in non_pre_reserved_passengers]

                sorted_indices = sorted(range(len(other_passengers_scheduled_deboarding_time)),
                                        key=lambda i: other_passengers_scheduled_deboarding_time[i])

                other_passengers_scheduled_deboarding_time = [other_passengers_scheduled_deboarding_time[i] for i in
                                                              sorted_indices]
                other_passengers_actual_deboarding_time = [other_passengers_actual_deboarding_time[i] for i in
                                                           sorted_indices]
                for p in non_pre_reserved_passengers:
                    p.scheduled_slack_time = other_passengers_scheduled_deboarding_time.pop(0)
                    p.actual_slack_time = other_passengers_actual_deboarding_time.pop(0)

    def print_info(self, *args):
        if not self.quiet_mode:
            print(*args)

    def run_simulation(self, nb_rows, nb_pax_carried, connecting_pax_df, seat_allocation_strategy_list,
                       deboarding_strategy_list,
                       percentage_prereserved_seats=-1, quiet_mode=True):
        activate_connecting_pax = len(connecting_pax_df) > 0

        self.prepare_simulation(nb_rows, nb_pax_carried, connecting_pax_df, percentage_prereserved_seats,
                                activate_connecting_pax)
        results_missed_pax = {(seat_allocation_strategy.value, deboarding_strategy.value): 0 for
                              seat_allocation_strategy in
                              seat_allocation_strategy_list for deboarding_strategy in deboarding_strategy_list}
        results_deboarding_time = {(seat_allocation_strategy.value, deboarding_strategy.value): 0 for
                                   seat_allocation_strategy in
                                   seat_allocation_strategy_list for deboarding_strategy in deboarding_strategy_list}
        for seat_allocation_strategy in seat_allocation_strategy_list:
            for deboarding_strategy in deboarding_strategy_list:
                self.set_seat_allocation_strategy(seat_allocation_strategy)
                self.set_deboarding_strategy(deboarding_strategy)
                self.assign_passenger_connecting_times(activate_connecting_pax)
                self.reset()
                self.run()
                missed_pax = self.evaluate_missing_pax()
                if not quiet_mode:
                    print(f"Nb of passengers missing their flights: {missed_pax}")
                results_missed_pax[(seat_allocation_strategy.value, deboarding_strategy.value)] += missed_pax
                results_deboarding_time[
                    (seat_allocation_strategy.value, deboarding_strategy.value)] += self.disembarkation_time

        return results_missed_pax, results_deboarding_time

    def run(self):
        self.reset()
        deboarding_completed = False
        while (self.t < self.t_max) & (not deboarding_completed):
            self.print_info(f'\n*** Step {self.t}')
            deboarding_completed = self.step()
            if not self.quiet_mode:
                self.print()
            self.t += 1
        disembarkation_time = self.t * configuration.time_step_duration
        print(f"Total minutes to disembark all pax: {round(disembarkation_time / 60, 2)}minutes")
        self.disembarkation_time = disembarkation_time

    def step(self) -> str:
        """
        Executes a single step in the deboarding simulation.
        This method iterates over all passengers and updates their state based on the current simulation time.
        Returns:
            str: A boolean indicating whether all passengers have disembarked.
        """
        for i, p in enumerate(self.passengers):
            if i == 0: continue

            if p.state != PassengerState.DISEMBARKED:
                self.history[self.t].append([self.t, i, p.x, p.y, int(p.state)])

            if p.next_action_t > self.t:
                continue
            match p.state:
                case PassengerState.SEATED:
                    self.move_seated_passsenger(i, p)
                case PassengerState.STAND_UP_FROM_SEAT:
                    self.move_stand_up_passenger(i, p)
                case PassengerState.MOVE_WAIT:
                    self.move_waiting_passenger(i, p)
                case PassengerState.MOVE_FROM_ROW:
                    self.move_passenger_in_row(i, p)
                case PassengerState.DISEMBARKED:
                    self.disembarked_pax += 1
                    p.next_action_t = self.t_max
                case _:
                    self.print_info(f'State {p.state} is not handled.')
        return self.disembarked_pax == self.n_passengers

    def move_passenger_in_row(self, i, p):
        if p.y == 0:
            if p.x > -2:
                if self.exit_corridor[p.x + self.n_seats_left - 1] != 0:
                    p.state = PassengerState.MOVE_WAIT
                    p.next_action_t = self.t + 1  # Wait until the next corridor position is free
                else:
                    if p.x == 0:
                        self.aisle[p.y] = 0
                    else:
                        self.exit_corridor[p.x + self.n_seats_left] = 0
                    p.x -= 1
                    self.exit_corridor[p.x + self.n_seats_left] = i
            else:
                if self.t < configuration.buffer_time_gate_connecting / configuration.time_step_duration:
                    p.state = PassengerState.MOVE_WAIT
                    p.next_action_t = configuration.buffer_time_gate_connecting / configuration.time_step_duration  # Wait until the next corridor position is free

                else:
                    p.state = PassengerState.DISEMBARKED
                    self.exit_corridor[p.x + self.n_seats_left] = 0
                    p.deboarding_time = (self.t * configuration.time_step_duration)
        else:
            if self.aisle[p.y - 1] != 0:
                p.state = PassengerState.MOVE_WAIT
                p.next_action_t = self.t + 1  # Wait until the aisle is free
            else:
                p.next_action_t = self.t + configuration.walk_duration
                self.aisle[p.y] = 0
                p.y -= 1
                self.aisle[p.y] = i

    def move_waiting_passenger(self, i, p):
        if p.y == 0:
            if self.exit_corridor[p.x + self.n_seats_left - 1] != 0:
                p.next_action_t = self.t + 1  # Wait until the corridor is free
            else:
                p.next_action_t = self.t + configuration.walk_duration
                p.state = PassengerState.MOVE_FROM_ROW
                if p.x == 0:
                    self.aisle[p.y] = 0
                else:
                    self.exit_corridor[p.x + self.n_seats_left] = 0
                p.x -= 1
                self.exit_corridor[p.x + self.n_seats_left] = i
        else:
            if self.aisle[p.y - 1] != 0:
                p.next_action_t = self.t + 1  # Wait until the aisle is free
            else:
                p.next_action_t = self.t + configuration.walk_duration
                p.state = PassengerState.MOVE_FROM_ROW
                self.aisle[p.y] = 0
                p.y -= 1
                self.aisle[p.y] = i

    def move_stand_up_passenger(self, i, p):
        if p.has_luggage:
            p.state = PassengerState.MOVE_WAIT
            p.next_action_t = self.t + p.collecting_luggage_time
            ind = 0 if p.seat < 0 else 1
            self.luggage_bin[p.seat_row][ind] += 1
            self.history_luggage.append([self.t, p.seat_row, ind])
        elif self.aisle[p.y - 1] != 0:
            p.state = PassengerState.MOVE_WAIT
            p.next_action_t = self.t + 1
        else:
            # We can go!
            p.next_action_t = self.t + configuration.walk_duration
            p.state = PassengerState.MOVE_FROM_ROW
            self.aisle[p.y] = 0
            p.y -= 1
            self.aisle[p.y] = i

    def move_seated_passsenger(self, i, p):
        random_left_right = np.random.randint(0, 2)
        is_courtesy_rule = self.deboarding_strategy == DeboardingStrategy.COURTESY_RULE
        if p.x == 1:
            if self.aisle[p.y] == 0:
                if self.side_left[p.y][2] != 0 and (random_left_right > 0):
                    p.next_action_t = self.t + 1
                else:
                    if is_courtesy_rule or (p.y + 1 == len(self.aisle)) or (self.aisle[p.y + 1] == 0):
                        self.side_right[p.y][0] = 0
                        p.x = 0
                        self.aisle[p.y] = i
                        p.state = PassengerState.STAND_UP_FROM_SEAT
                        p.next_action_t = self.t + configuration.stand_up_duration
                    else:
                        p.next_action_t = self.t + 1
            else:
                p.next_action_t = self.t + 1
        if p.x == -1:
            if self.aisle[p.y] == 0:
                if is_courtesy_rule or (p.y + 1 == len(self.aisle)) or (self.aisle[p.y + 1] == 0):
                    self.side_left[p.y][configuration.nb_seat_left - 1] = 0
                    p.x = 0
                    self.aisle[p.y] = i
                    p.state = PassengerState.STAND_UP_FROM_SEAT
                    p.next_action_t = self.t + configuration.stand_up_duration
                else:
                    p.next_action_t = self.t + 1
            else:
                p.next_action_t = self.t + 1
        elif p.x < -1:
            if self.side_left[p.seat_row][configuration.nb_seat_left + p.x + 1] == 0:
                self.side_left[p.seat_row][configuration.nb_seat_left + p.x] = 0
                p.x += 1
                self.side_left[p.seat_row][configuration.nb_seat_left + p.x] = i
                p.next_action_t = self.t + configuration.move_seat_duration
            else:
                p.next_action_t = self.t + 1
        elif p.x > 1:
            if self.side_right[p.seat_row][p.x - 2] == 0:
                self.side_right[p.seat_row][p.x - 1] = 0
                p.x -= 1
                self.side_right[p.seat_row][p.x - 1] = i
                p.next_action_t = self.t + configuration.move_seat_duration
            else:
                p.next_action_t = self.t + 1

    def serialize_history(self, path):
        with open(path, 'w') as f:
            # General parameters in the header.
            f.write(
                f'{self.n_fictive_rows} {self.dummy_rows} {self.n_seats_left} {self.n_seats_right} {self.n_passengers} {len(self.history_luggage)}\n')

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
        for i, passenger in enumerate(self.passengers):
            if i == 0: continue

            if passenger.deboarding_time < 0:
                logging.warning(f"Error when computing deboarding time of passenger {i}")
            else:
                if (passenger.deboarding_time > passenger.actual_slack_time - configuration.gate_close_time):
                    nb_missed_pax += 1
        return nb_missed_pax
