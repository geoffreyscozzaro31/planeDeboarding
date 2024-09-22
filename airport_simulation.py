"""
This script is used to run for a whole day the simulation of passenger transfers at  the airport.
The simulation is run for a whole day and the number of passengers missing their flights is computed for connecting flight.
"""
import time

import numpy as np
import pandas as pd

import flight_schedule
from config_deboarding import DAY_LABEL, NB_SIMULATION, MIN_LOAD_FACTOR, MAX_LOAD_FACTOR
from deboarding_simulation import DeboardingSimulation
from model.deboarding_strategy import DeboardingStrategy
from model.seat_allocation import SeatAllocation


def prepare_data_for_simulation(df_arrival_flight, flight_id):
    nb_pax_carried = \
        df_arrival_flight[df_arrival_flight["arrival_flight_id"] == flight_id]["actual_passenger_count"].values[0]

    stochastic_load_factor = np.random.uniform(MIN_LOAD_FACTOR, MAX_LOAD_FACTOR)
    nb_rows = int(np.ceil(nb_pax_carried / (6 * stochastic_load_factor)))

    df_connections = pd.read_csv(f"data/{DAY_LABEL}/connecting_passengers_3h_max_connecting_time.csv")
    df_connections = flight_schedule.compute_buffer_times(df_connections)
    crop_df = df_connections[df_connections["arrival_flight_id"] == flight_id]
    return nb_rows, nb_pax_carried, crop_df


def save_results(missed_pax_per_strategy, deboarding_times_per_strategy, filename):
    df = pd.DataFrame(
        columns=['Seat Allocation', 'Deboarding Strategy', "Total Missed Pax", "List Missed Pax",
                 'Average Deboarding Time', "List Deboarding Time"])
    cpt = 0
    for (seat_allocation, deboarding_strategy), missed_pax in missed_pax_per_strategy.items():
        deboarding_times = deboarding_times_per_strategy[(seat_allocation, deboarding_strategy)]
        df.loc[cpt] = [seat_allocation, deboarding_strategy, np.sum(missed_pax), missed_pax,
                       np.mean(deboarding_times), deboarding_times]
        cpt += 1
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    start_time = time.time()

    df_arrival_flight = pd.read_csv(
        f"data/{DAY_LABEL}/df_arrival_flights_3h_max_connecting_time.csv")  # todo: change in function of day
    flight_ids = df_arrival_flight["arrival_flight_id"].unique()

    list_seat_allocation_strategy = [SeatAllocation.RANDOM, SeatAllocation.CONNECTING_PRIORITY]
    deboarding_strategy_list = [DeboardingStrategy.COURTESY_RULE, DeboardingStrategy.AISLE_PRIORITY_RULE]
    simulation = DeboardingSimulation(quiet_mode=True)
    nb_flights = len(flight_ids)
    for i in range(nb_flights):
        results_missing_pax = {(seat_allocation.value, deboarding_strategy.value): [] for seat_allocation in
                               list_seat_allocation_strategy for
                               deboarding_strategy in deboarding_strategy_list}
        results_deboarding_times = {(seat_allocation.value, deboarding_strategy.value): [] for seat_allocation in
                                    list_seat_allocation_strategy for
                                    deboarding_strategy in deboarding_strategy_list}

        print(f"**************Simulation {i + 1}/{NB_SIMULATION}*****************")
        for flight_id in flight_ids[:nb_flights]:
            nb_rows, nb_pax_carried, crop_df = prepare_data_for_simulation(df_arrival_flight, flight_id)
            missed_pax_per_allocationn, deboarding_time_per_allocation = simulation.run_simulation(nb_rows,
                                                                                                   nb_pax_carried,
                                                                                                   crop_df,
                                                                                                   list_seat_allocation_strategy,
                                                                                                   deboarding_strategy_list)

            for strategy, missed_pax in missed_pax_per_allocationn.items():
                results_missing_pax[strategy] += [missed_pax]
            for strategy, deboarding_time in deboarding_time_per_allocation.items():
                results_deboarding_times[strategy] += [deboarding_time]

        print("Missed pax: ", results_missing_pax)
        print("Missed pax per strategy:")
        for strategy, missed_pax in results_missing_pax.items():
            avg_missed_pax = np.mean(missed_pax)
            print(f"Strategy {strategy}: Average missed pax = {avg_missed_pax:.2f}")

        output_filename = f"results/{DAY_LABEL}/results_simulation_{i}.csv"
        # save_results(results_missing_pax, results_deboarding_times, output_filename)
    end_time = time.time()

    print(f"Total execution time: {end_time - start_time:.2f} seconds")

