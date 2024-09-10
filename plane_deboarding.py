import logging
from collections import defaultdict
import time
from enum import Enum, IntEnum
from typing import List

import pandas as pd
from scipy.stats import weibull_min

import flight_schedule
import prereserved_seats
from config_deboarding import *

DAY_LABEL = "max_delay_day"

class State(IntEnum):
    UNDEFINED = 0
    SEATED = 1
    STAND_UP_FROM_SEAT = 2
    # todo: add collect luggage status
    MOVE_WAIT = 3
    MOVE_FROM_ROW = 4
    DISEMBARKED = 5


class SeatAllocation(Enum):
    RANDOM = "RANDOM_SEAT_ALLOCATION"
    CONNECTING_PRIORITY = "CONNECTING_PAX_PRIORITY_SEAT_ALLOCATION"


class DeboardingStrategy(Enum):
    COURTESY_RULE = "COURTESY_DEBOARDING_RULE_DEBOARDING"
    AISLE_PRIORITY_RULE = "AISLE_PRIORITY_DEBOARDING_RULE"


class Passenger:
    def __init__(self, seat_row, seat, has_luggage, has_pre_reserved_seat):
        self.seat_row = seat_row
        self.seat = seat  # seat number (e.g. 1-3 for places on the right, negative numbers for places to the left)
        self.slack_time = -1  # = max deboarding time (in minutes) authorised for this pax before missing its next flight, set to np.inf if not connecting pax
        self.deboarding_time = -1
        self.has_luggage = has_luggage
        self.state = State.SEATED
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


class Simulation:
    def __init__(self, dummy_rows=2, quiet_mode=True):
        self.dummy_rows = dummy_rows  # Add fictive rows  to model empty space at the top of the aircraft
        self.n_rows = -1
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
        self.t_max = T_MAX_SIMULATION
        self.disembarked_pax = 0
        self.connecting_time_pax_list = []
        self.passengers_seat_allocation_list = []
        self.probability_matrix_prereserved_seats = []
        self.passenger_has_luggage_list = []
        self.passenger_has_pre_reserved_seat_list = []
        self.nb_prereserved_seats = 0

    def prepare_simulation(self, nb_rows, nb_pax_carried, crop_df):
        """
        Prepares the simulation by setting up the aircraft and passengers. Do it stochastically, each time this method is launched
        a new simulation instance is generated.

        Parameters:
        nb_rows (int): Number of rows in the aircraft.
        nb_pax_carried (int): Total number of passengers carried on the flight.
        crop_df (pd.DataFrame): DataFrame containing information about connecting flights.

        The function performs the following steps:
        1. Sets up the custom aircraft configuration.
        2. Sets the number of passengers.
        3. Sets the connecting time for passengers.
        4. Identifies passengers with pre-reserved seats.
        5. Identifies passengers having luggage.
        6. Sets the probability matrix for pre-reserved seats.
        7. Allocates seats to passengers.
        8. Creates passenger objects.
        9. Sorts passengers for simulation.
        """
        self.set_custom_aircraft(n_rows=nb_rows, n_seats_left=NB_SEAT_LEFT, n_seats_right=NB_SEAT_RIGHT)
        self.set_number_passengers(nb_pax_carried)

        self.set_connecting_time_pax(crop_df, nb_pax_carried)
        self.set_passengers_having_luggage()
        self.set_probability_matrix_prereserved_seats()
        self.set_passengers_with_prereserved_seats()
        self.set_passenger_seat_allocation()
        self.create_passengers()
        self.sort_passengers()
        return simulation

    def randomize_initialisation(self):
        self.set_passengers_with_prereserved_seats()
        self.set_passengers_having_luggage()
        self.set_passenger_seat_allocation()
        self.create_passengers()
        self.sort_passengers()

    def set_connecting_time_pax(self, df_connecting_flight, nb_pax_carried, default_transfer_time_seconds=-1):
        """
        Sets the connecting time for passengers based on the provided flight data.

        Parameters:
        df_connecting_flight (pd.DataFrame): DataFrame containing information about connecting flights.
        nb_pax_carried (int): Total number of passengers carried on the flight.

        The function performs the following steps:
        1. Extracts the number of connecting passengers and their buffer times from the DataFrame.
        2. Repeats the buffer times according to the number of connecting passengers.
        3. Calculates the number of remaining passengers and assigns them an infinite buffer time.
        4. Shuffles the list of connecting times to randomize the order.

        """
        nb_pax_list = df_connecting_flight['nb_connecting_pax'].values
        buffer_time_list = df_connecting_flight['buffer_time_actual_seconds'].values
        if default_transfer_time_seconds >= 0:
            buffer_time_list = np.full(len(buffer_time_list), default_transfer_time_seconds)
        connecting_pax_list = np.repeat(buffer_time_list, nb_pax_list)
        nb_connecting_pax = len(connecting_pax_list)
        remaining_pax = nb_pax_carried - len(connecting_pax_list)
        print(remaining_pax,nb_pax_carried, len(connecting_pax_list))
        connecting_pax_list = np.concatenate((connecting_pax_list, np.full(remaining_pax, np.inf)))
        np.random.shuffle(connecting_pax_list)
        self.connecting_time_pax_list = connecting_pax_list
        print(f"nb_connecting pax:{nb_connecting_pax},  nb_other_pax:{remaining_pax}")

    def set_custom_aircraft(self, n_rows, n_seats_left=2, n_seats_right=2):
        self.n_rows = 2 * n_rows
        self.n_seats_left = n_seats_left
        self.n_seats_right = n_seats_right

    def set_passengers_number(self, n):
        self.n_passengers = n

    def set_default_passengers_proportion(self, proportion):
        capacity = self.n_rows * (self.n_seats_left + self.n_seats_right) // 2
        self.n_passengers = int(proportion * capacity)

    def set_number_passengers(self, n):
        self.n_passengers = n

    def set_seat_allocation_strategy(self, seat_allocation_strategy):
        self.seat_allocation_strategy = seat_allocation_strategy

    def set_deboarding_strategy(self, deboarding_strategy):
        self.deboarding_strategy = deboarding_strategy

    def reset_stats(self):
        self.disembarkation_times = []

    def reset_passengers(self):
        for p in self.passengers:
            if p != None:
                p.state = State.SEATED
                p.deboarding_time = -1
                p.next_action_t = 0
                p.x = p.seat
                p.y = p.seat_row

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
        self.disembarked_pax = 0
        self.side_left = np.zeros((self.n_rows * 2 + self.dummy_rows, self.n_seats_left), dtype=int)
        self.side_right = np.zeros((self.n_rows * 2 + self.dummy_rows, self.n_seats_right), dtype=int)
        self.exit_corridor = np.zeros(self.n_seats_left, dtype=int)
        self.aisle = np.zeros(self.n_rows * 2 + self.dummy_rows, dtype=int)
        self.luggage_bin = np.zeros((self.n_rows * 2 + self.dummy_rows, 2), dtype=int)
        self.deboarding_order_left = np.zeros((self.n_rows * 2 + self.dummy_rows, self.n_seats_left), dtype=int)
        self.deboarding_order_right = np.zeros((self.n_rows * 2 + self.dummy_rows, self.n_seats_right), dtype=int)
        self.fill_aircraft()
        self.reset_stats()
        self.reset_passengers()

    def fill_aircraft(self):
        for i, passenger in enumerate(self.passengers[1:]):
            if (passenger.seat < 0):
                self.side_left[passenger.seat_row][NB_SEAT_LEFT + passenger.seat] = i + 1
            else:
                self.side_right[passenger.seat_row][passenger.seat - 1] = i + 1

    def set_passengers_having_luggage(self):
        stochastic_percentage_has_luggage = np.random.uniform(MIN_PERCENTAGE_HAS_LUGGAGE, MAX_PERCENTAGE_HAS_LUGGAGE)
        self.passenger_has_luggage_list = np.random.rand(self.n_passengers) < stochastic_percentage_has_luggage / 100

    def set_probability_matrix_prereserved_seats(self):
        self.probability_matrix_prereserved_seats = prereserved_seats.generate_probability_matrix(nb_rows,
                                                                                                  NB_SEAT_LEFT + NB_SEAT_RIGHT)

    def set_passengers_with_prereserved_seats(self):
        stochastic_percentage_prereserved_seats = np.random.uniform(MIN_PERCENTAGE_PRERESERVED_SEATS,
                                                                    MAX_PERCENTAGE_PRERESERVED_SEATS)
        self.passenger_has_pre_reserved_seat_list = np.random.rand(
            self.n_passengers) < stochastic_percentage_prereserved_seats / 100
        self.nb_prereserved_seats = np.sum(self.passenger_has_pre_reserved_seat_list)

    def set_passenger_seat_allocation(self):
        prereserved_seats_pax_allocation, other_seats = prereserved_seats.assign_prereserved_seats(
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
                seat_column = seat[1] - NB_SEAT_LEFT
            else:
                seat_column = seat[1] - NB_SEAT_LEFT + 1
            self.passengers.append(Passenger(seat_row=seat_row, seat=seat_column,
                                             has_luggage=self.passenger_has_luggage_list[i],
                                             has_pre_reserved_seat=self.passenger_has_pre_reserved_seat_list[i]))

    def sort_passengers(self):
        combined_list = list(
            zip(self.passengers[1:], self.passenger_has_pre_reserved_seat_list, self.connecting_time_pax_list))
        combined_list.sort(key=lambda p: (p[0].seat_row, abs(p[0].seat)))  # sort for simulation iteration
        self.passengers, self.passenger_has_pre_reserved_seat_list, self.connecting_time_pax_list = zip(*combined_list)
        self.passengers = [None] + list(self.passengers)
        # print(self.passengers)
        self.passenger_has_pre_reserved_seat_list = list(self.passenger_has_pre_reserved_seat_list)
        self.connecting_time_pax_list = list(self.connecting_time_pax_list)

    def assign_passenger_connecting_times(self):
        for i in range(len(self.passengers) - 1):
            self.passengers[i + 1].slack_time = self.connecting_time_pax_list[i]
        if self.seat_allocation_strategy != SeatAllocation.RANDOM:
            non_pre_reserved_passengers = [p for p in self.passengers[1:] if not p.has_pre_reserved_seat]

            if self.deboarding_strategy == DeboardingStrategy.COURTESY_RULE:
                non_pre_reserved_passengers.sort(key=lambda p: (p.seat_row, abs(p.seat)))

            elif self.deboarding_strategy == DeboardingStrategy.AISLE_PRIORITY_RULE:
                non_pre_reserved_passengers.sort(key=lambda p: (abs(p.seat), p.seat_row))

            else:
                raise ValueError("Unknown seat allocation strategy")
            print(self.deboarding_strategy)
            non_prereserved_passengers_deboarding_time = sorted([p.slack_time for p in non_pre_reserved_passengers])
            for p in non_pre_reserved_passengers:
                p.slack_time = non_prereserved_passengers_deboarding_time.pop(0)

    def print_info(self, *args):
        if not self.quiet_mode:
            print(*args)

    # Run multiple simulations
    def run_multiple(self, nb_run, nb_rows, nb_pax_carried, crop_df, seat_allocation_strategy_list, quiet_mode=True):
        self.prepare_simulation(nb_rows, nb_pax_carried, crop_df)
        results = {(seat_allocation_strategy.value, deboarding_strategy.value): 0 for seat_allocation_strategy in
                   seat_allocation_strategy_list for deboarding_strategy in deboarding_strategy_list}
        for i in range(nb_run):
            print(f'*** Simulation {i} ***')
            self.randomize_initialisation()
            for seat_allocation_strategy in seat_allocation_strategy_list:
                for deboarding_strategy in deboarding_strategy_list:
                    print(
                        f"******* run simulation with {seat_allocation_strategy.value} with {deboarding_strategy.value} ....")
                    self.set_seat_allocation_strategy(seat_allocation_strategy)
                    self.set_deboarding_strategy(deboarding_strategy)
                    self.assign_passenger_connecting_times()
                    self.reset()
                    self.run()
                    missed_pax = self.evaluate_missing_pax()
                    if not quiet_mode:
                        print(f"Nb of passengers missing their flights: {missed_pax}")
                    results[(seat_allocation_strategy.value, deboarding_strategy.value)] += missed_pax
        return results

    # Run a single simulation
    def run(self):
        self.reset()
        deboarding_completed = False
        while (self.t < self.t_max) & (not deboarding_completed):
            self.print_info(f'\n*** Step {self.t}')
            deboarding_completed = self.step()
            if not self.quiet_mode:
                self.print()
            self.t += 1
        disembarkation_time = self.t * TIME_STEP_DURATION
        # print(f"Total minutes to disembark all pax: {round(disembarkation_time/60, 2)}minutes")
        self.disembarkation_times.append(disembarkation_time)

    def step(self):

        for i, p in enumerate(self.passengers):
            if i == 0: continue

            if p.state != State.DISEMBARKED:
                self.history[self.t].append([self.t, i, p.x, p.y, int(p.state)])

            if p.next_action_t > self.t:
                continue

            match p.state:
                case State.SEATED:
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
                                    p.state = State.STAND_UP_FROM_SEAT
                                    p.next_action_t = self.t + STAND_UP_DURATION
                                else:
                                    p.next_action_t = self.t + 1
                        else:
                            p.next_action_t = self.t + 1
                    if p.x == -1:
                        if self.aisle[p.y] == 0:
                            if is_courtesy_rule or (p.y + 1 == len(self.aisle)) or (self.aisle[p.y + 1] == 0):
                                self.side_left[p.y][NB_SEAT_LEFT - 1] = 0
                                p.x = 0
                                self.aisle[p.y] = i
                                p.state = State.STAND_UP_FROM_SEAT
                                p.next_action_t = self.t + STAND_UP_DURATION
                            else:
                                p.next_action_t = self.t + 1
                        else:
                            p.next_action_t = self.t + 1
                    elif p.x < -1:
                        if self.side_left[p.seat_row][NB_SEAT_LEFT + p.x + 1] == 0:
                            self.side_left[p.seat_row][NB_SEAT_LEFT + p.x] = 0
                            p.x += 1
                            self.side_left[p.seat_row][NB_SEAT_LEFT + p.x] = i
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

                case State.STAND_UP_FROM_SEAT:
                    if p.has_luggage:
                        p.state = State.MOVE_WAIT
                        p.next_action_t = self.t + p.collecting_luggage_time
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
                    if p.y == 0:
                        if self.exit_corridor[p.x + self.n_seats_left - 1] != 0:
                            p.next_action_t = self.t + 1  # Wait until the corridor is free
                        else:
                            p.next_action_t = self.t + WALK_DURATION
                            p.state = State.MOVE_FROM_ROW
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
                            p.next_action_t = self.t + WALK_DURATION
                            p.state = State.MOVE_FROM_ROW
                            self.aisle[p.y] = 0
                            p.y -= 1
                            self.aisle[p.y] = i

                case State.MOVE_FROM_ROW:
                    if p.y == 0:
                        if p.x > -2:
                            if self.exit_corridor[p.x + self.n_seats_left - 1] != 0:
                                p.state = State.MOVE_WAIT
                                p.next_action_t = self.t + 1  # Wait until the next corridor position is free
                            else:
                                if p.x == 0:
                                    self.aisle[p.y] = 0
                                else:
                                    self.exit_corridor[p.x + self.n_seats_left] = 0
                                p.x -= 1
                                self.exit_corridor[p.x + self.n_seats_left] = i
                        else:
                            if self.t < BUFFER_TIME_GATE_CONNECTING / TIME_STEP_DURATION:
                                p.state = State.MOVE_WAIT
                                p.next_action_t = BUFFER_TIME_GATE_CONNECTING / TIME_STEP_DURATION  # Wait until the next corridor position is free

                            else:
                                p.state = State.DISEMBARKED
                                self.exit_corridor[p.x + self.n_seats_left] = 0
                                p.deboarding_time = (self.t * TIME_STEP_DURATION)
                    else:
                        if self.aisle[p.y - 1] != 0:
                            p.state = State.MOVE_WAIT
                            p.next_action_t = self.t + 1  # Wait until the aisle is free
                        else:
                            p.next_action_t = self.t + WALK_DURATION
                            self.aisle[p.y] = 0
                            p.y -= 1
                            self.aisle[p.y] = i

                case State.DISEMBARKED:
                    self.disembarked_pax += 1
                    p.next_action_t = self.t_max
                case _:
                    self.print_info(f'State {p.state} is not handled.')

        # Check whether everyone is already deboarded
        return self.disembarked_pax == self.n_passengers

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
        for i, passenger in enumerate(self.passengers):
            if i == 0: continue

            if passenger.deboarding_time < 0:
                logging.warning(f"Error when computing deboarding time of passenger {i}")
            else:
                if (passenger.deboarding_time > passenger.slack_time):
                    nb_missed_pax += 1
        return nb_missed_pax


def prepare_data_for_simulation(df_arrival_flight, flight_id):
    nb_pax_carried = \
        df_arrival_flight[df_arrival_flight["arrival_flight_id"] == flight_id]["actual_passenger_count"].values[0]

    stochastic_load_factor = np.random.uniform(MIN_LOAD_FACTOR, MAX_LOAD_FACTOR)
    nb_rows = int(np.ceil(nb_pax_carried / (6 * stochastic_load_factor)))

    print("nb_rows:", nb_rows, "nb_passengers_carried", nb_pax_carried)

    df_connections = pd.read_csv(f"data/{DAY_LABEL}/connecting_passengers.csv")  # todo: change in function of day
    df_connections = flight_schedule.compute_buffer_times(df_connections)
    crop_df = df_connections[df_connections["arrival_flight_id"] == flight_id]
    return nb_rows, nb_pax_carried, crop_df


def save_results(results, filename):
    df = pd.DataFrame(
        columns=['Seat Allocation', 'Deboarding Strategy', 'Average Missed Pax', "All Missed Pax", "List Missed Pax"])
    print(results)
    cpt = 0
    for (seat_allocation, deboarding_strategy), missed_pax in results.items():
        df.loc[cpt] = [seat_allocation, deboarding_strategy, np.mean(missed_pax), np.sum(missed_pax), missed_pax]
        cpt += 1
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    start_time = time.time()

    df_arrival_flight = pd.read_csv(f"data/{DAY_LABEL}/df_arrival_flights.csv")  # todo: change in function of day
    flight_ids = df_arrival_flight["arrival_flight_id"].unique()

    list_seat_allocation_strategy = [SeatAllocation.RANDOM, SeatAllocation.CONNECTING_PRIORITY]
    deboarding_strategy_list = [DeboardingStrategy.COURTESY_RULE, DeboardingStrategy.AISLE_PRIORITY_RULE]
    results = {(seat_allocation.value, deboarding_strategy.value): [] for seat_allocation in
               list_seat_allocation_strategy for
               deboarding_strategy in deboarding_strategy_list}
    simulation = Simulation(quiet_mode=True, dummy_rows=2)
    nb_flights = len(flight_ids)
    for flight_id in flight_ids[:nb_flights]:
        nb_rows, nb_pax_carried, crop_df = prepare_data_for_simulation(df_arrival_flight, flight_id)
        missed_pax_per_allocation = simulation.run_multiple(NB_SIMULATION, nb_rows, nb_pax_carried, crop_df,
                                                            list_seat_allocation_strategy, deboarding_strategy_list)
        for strategy, missed_pax in missed_pax_per_allocation.items():
            results[strategy] += [missed_pax]

    print("Missed pax: ", results)
    print("Missed pax per strategy:")
    for strategy, missed_pax in results.items():
        avg_missed_pax = np.mean(missed_pax)
        print(f"Strategy {strategy}: Average missed pax = {avg_missed_pax:.2f}")

    output_filename = f"results/{DAY_LABEL}/results_missed_pax.csv"
    save_results(results, output_filename)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.2f} seconds")
    # missing_pax_list = np.array(missing_pax_list)
    # print(f"Average disembarkation time : {round(np.mean(self.disembarkation_times) / 60, 2)}min")
    # print(f"Min disembarkation time : {round(np.min(self.disembarkation_times) / 60, 2)}min")
    # print(f"Max disembarkation time : {round(np.max(self.disembarkation_times) / 60, 2)}min")
