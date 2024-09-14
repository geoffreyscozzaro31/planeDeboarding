import numpy as np
import os

import plane_deboarding
from config_deboarding import *

OUTPUT_DIR = 'medias/deboarding/'


def measure_deboarding_time(simulation, disembarking_rule_name, n=10):
    for passengers_proportion in [0.8, 0.9, 1.0]:
        for seat_allocation in [plane_deboarding.SeatAllocation.RANDOM]:
            simulation.set_passengers_proportion(passengers_proportion)
            simulation.set_seat_allocation_strategy(seat_allocation)
            simulation.run_simulation(n)

            # print(seat_allocation, passengers_proportion, np.mean(10_simulations_40_pct_prereserved_3h_connecting_time_without_gate_closure_time.deboarding_time))

            file_name = f'{seat_allocation.name.lower()}_{disembarking_rule_name}_{int(100 * passengers_proportion)}pct'
            full_path = os.path.join(OUTPUT_DIR,
                                     f'{file_name}_{NB_ROWS}_{simulation.n_seats_left}_total_time.txt')
            print("start writing file deboarding time..")
            with open(full_path, "w") as file:
                file.write(f'{seat_allocation.name.lower()} {passengers_proportion}\n')
                file.write(' '.join(map(str, simulation.disembarkation_times)))
            print("file written !..")


def save_deboarding_orders(simulation,disembarking_rule_name):
    for seat_allocation in [plane_deboarding.SeatAllocation.RANDOM]:
        simulation.set_seat_allocation_strategy(seat_allocation)
        simulation.reset()

        # print(seat_allocation)
        # 10_simulations_40_pct_prereserved_3h_connecting_time_without_gate_closure_time.print_deboarding_order()

        full_path = os.path.join(OUTPUT_DIR,
                                 f'{seat_allocation.name.lower()}_{disembarking_rule_name}_{NB_ROWS}_{simulation.n_seats_left}_deboarding_order.txt')

        with open(full_path, "w") as file:
            for i in range(simulation.dummy_rows, simulation.n_fictive_rows + simulation.dummy_rows):
                row = list(simulation.deboarding_order_left[i, :][::-1]) + [-1] + list(
                    simulation.deboarding_order_right[i, :])
                file.write(' '.join(map(str, row)) + '\n')


from plane_deboarding import Simulation, SeatAllocation, DeboardingStrategy, prepare_data_for_simulation
import pandas as pd


def main():
    seat_allocation_strategy = SeatAllocation.RANDOM
    deboarding_strategy = DeboardingStrategy.AISLE_PRIORITY_RULE
    # deboarding_strategy = DeboardingStrategy.COURTESY_RULE
    deboarding_strategy_name = deboarding_strategy.value.lower()
    print(deboarding_strategy_name)
    simulation = Simulation(quiet_mode=True, dummy_rows=2)
    nb_rows = 20
    nb_pax_carried = (nb_rows - 1) * 6
    crop_df = pd.DataFrame()
    simulation.run_simulation(nb_rows, nb_pax_carried, crop_df, [seat_allocation_strategy], [deboarding_strategy], 0)
    file_name = f'{deboarding_strategy.value.lower()}_{NB_ROWS}rows_history.txt'
    simulation.serialize_history(os.path.join(OUTPUT_DIR + file_name))


if __name__ == "__main__":
    main()
