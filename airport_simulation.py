"""
This script simulates passenger transfers at the airport for a whole day and computes the number of passengers missing their flights for connecting flights.
"""

import time

import numpy as np
import pandas as pd

from deboarding_simulation import DeboardingSimulation
from flight_schedule import compute_buffer_times
from model.deboarding_strategy import DeboardingStrategy
from model.seat_allocation import SeatAllocation
from utils import configloader

CONFIG_FILE_PATH = "configuration_deboarding.yaml"


class AirportSimulation:
    def __init__(self, config):
        self.config = config
        self.simulation = DeboardingSimulation(quiet_mode=True)
        self.seat_allocation_strategies = [SeatAllocation.RANDOM, SeatAllocation.CONNECTING_PRIORITY]
        self.deboarding_strategies = [DeboardingStrategy.COURTESY_RULE, DeboardingStrategy.AISLE_PRIORITY_RULE]

    def run(self):
        start_time = time.time()

        flight_ids, df_arrival_flight = self._load_flight_data()
        nb_flights = len(flight_ids)

        for i in range(self.config.nb_simulation):
            print(f"**************Simulation {i + 1}/{self.config.nb_simulation}*****************")
            results_missing_pax, results_deboarding_times = self._run_simulation_for_flights(df_arrival_flight,
                                                                                             flight_ids)

            self._display_results(results_missing_pax)
            self._save_results(results_missing_pax, results_deboarding_times, i)

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

    def _load_flight_data(self):
        df_arrival_flight = pd.read_csv(f"data/{self.config.day_label}/df_arrival_flights_3h_max_connecting_time.csv")
        flight_ids = df_arrival_flight["arrival_flight_id"].unique()
        return flight_ids, df_arrival_flight

    def _run_simulation_for_flights(self, df_arrival_flight, flight_ids):
        results_missing_pax = self._init_results_dict()
        results_deboarding_times = self._init_results_dict()

        for flight_id in flight_ids:
            nb_rows, nb_pax_carried, crop_df = self._prepare_flight_data(df_arrival_flight, flight_id)
            missed_pax, deboarding_times = self.simulation.run_simulation(nb_rows, nb_pax_carried, crop_df,
                                                                          self.seat_allocation_strategies,
                                                                          self.deboarding_strategies)
            self._update_results(missed_pax, results_missing_pax)
            self._update_results(deboarding_times, results_deboarding_times)

        return results_missing_pax, results_deboarding_times

    def _prepare_flight_data(self, df_arrival_flight, flight_id):
        nb_pax_carried = \
        df_arrival_flight[df_arrival_flight["arrival_flight_id"] == flight_id]["actual_passenger_count"].values[0]
        stochastic_load_factor = np.random.uniform(self.config.min_load_factor, self.config.max_load_factor)
        nb_rows = int(np.ceil(nb_pax_carried / (6 * stochastic_load_factor)))

        df_connections = pd.read_csv(f"data/{self.config.day_label}/connecting_passengers_3h_max_connecting_time.csv")
        df_connections = compute_buffer_times(df_connections)
        crop_df = df_connections[df_connections["arrival_flight_id"] == flight_id]

        return nb_rows, nb_pax_carried, crop_df

    def _init_results_dict(self):
        return {(sa.value, ds.value): [] for sa in self.seat_allocation_strategies for ds in self.deboarding_strategies}

    def _update_results(self, new_results, results_dict):
        for strategy, result in new_results.items():
            results_dict[strategy] += [result]

    def _display_results(self, results_missing_pax):
        print("Missed pax per strategy:")
        for strategy, missed_pax in results_missing_pax.items():
            avg_missed_pax = np.mean(missed_pax)
            print(f"Strategy {strategy}: Average missed pax = {avg_missed_pax:.2f}")

    def _save_results(self, missed_pax_per_strategy, deboarding_times_per_strategy, simulation_index):
        output_filename = f"results/{self.config.day_label}/results_simulation_{simulation_index}.csv"
        df = pd.DataFrame(columns=['Seat Allocation', 'Deboarding Strategy', "Total Missed Pax", "List Missed Pax",
                                   'Average Deboarding Time', "List Deboarding Time"])
        for i, ((seat_allocation, deboarding_strategy), missed_pax) in enumerate(missed_pax_per_strategy.items()):
            deboarding_times = deboarding_times_per_strategy[(seat_allocation, deboarding_strategy)]
            df.loc[i] = [seat_allocation, deboarding_strategy, np.sum(missed_pax), missed_pax,
                         np.mean(deboarding_times), deboarding_times]
        df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    config = configloader.ConfigLoader(CONFIG_FILE_PATH)
    config.display()

    simulation = AirportSimulation(config)
    simulation.run()
