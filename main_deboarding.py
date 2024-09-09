import numpy as np
import os

import plane_deboarding
from config_deboarding import *

OUTPUT_DIR = 'medias/deboarding/'




def save_history(simulation, n=1):
    for passengers_proportion in [1.0]:
        for seat_allocation in [plane_deboarding.SeatAllocation.RANDOM]:
            simulation.set_seat_allocation_strategy(seat_allocation)
            for i in range(n):
                simulation.run()
                file_name = f'{seat_allocation.name.lower()}_{DISEMBARKING_RULE_NAME}_{100*int(passengers_proportion)}pct_{NB_ROWS}_{simulation.n_seats_left}_history_{i}.txt'
                simulation.serialize_history(os.path.join(OUTPUT_DIR, file_name))
            break


def measure_deboarding_time(simulation, n=10):
    for passengers_proportion in [0.8, 0.9, 1.0]:
        for seat_allocation in [plane_deboarding.SeatAllocation.RANDOM]:
            simulation.set_passengers_proportion(passengers_proportion)
            simulation.set_seat_allocation_strategy(seat_allocation)
            simulation.run_multiple(n)

            # print(seat_allocation, passengers_proportion, np.mean(simulation.deboarding_time))

            file_name = f'{seat_allocation.name.lower()}_{DISEMBARKING_RULE_NAME}_{int(100*passengers_proportion)}pct'
            full_path = os.path.join(OUTPUT_DIR,
                                     f'{file_name}_{NB_ROWS}_{simulation.n_seats_left}_total_time.txt')
            print("start writing file deboarding time..")
            with open(full_path, "w") as file:
                file.write(f'{seat_allocation.name.lower()} {passengers_proportion}\n')
                file.write(' '.join(map(str, simulation.disembarkation_times)))
            print("file written !..")


def save_deboarding_orders(simulation):
    for seat_allocation in [plane_deboarding.SeatAllocation.RANDOM]:
        simulation.set_seat_allocation_strategy(seat_allocation)
        simulation.reset()

        # print(seat_allocation)
        # simulation.print_deboarding_order()

        full_path = os.path.join(OUTPUT_DIR,
                                 f'{seat_allocation.name.lower()}_{DISEMBARKING_RULE_NAME}_{NB_ROWS}_{simulation.n_seats_left}_deboarding_order.txt')

        with open(full_path, "w") as file:
            for i in range(simulation.dummy_rows, simulation.n_rows + simulation.dummy_rows):
                row = list(simulation.deboarding_order_left[i, :][::-1]) + [-1] + list(
                    simulation.deboarding_order_right[i, :])
                file.write(' '.join(map(str, row)) + '\n')


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    simulation = plane_deboarding.Simulation(quiet_mode=True, dummy_rows=2)
    simulation.set_custom_aircraft(n_rows=NB_ROWS, n_seats_left=3, n_seats_right=3)
    simulation.set_passengers_proportion(1.0)
    simulation.set_seat_allocation_strategy(plane_deboarding.SeatAllocation.RANDOM)

    save_deboarding_orders(simulation)
    save_history(simulation, n=1)
    # measure_deboarding_time(simulation, n=1)


if __name__ == "__main__":
    main()
