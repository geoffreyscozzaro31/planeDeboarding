from plane_deboarding import SeatAllocation, Simulation
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

MAX_TRANSFER_TIME = 24 * 6 * 60  # for non-connecting passengers

DAY_LABEL = "max_flight_day"
# DAY_LABEL = "max_delay_day"
DATA_FOLDER = f"data/{DAY_LABEL}/"

if DAY_LABEL == "max_flight_day":
    FLIGHT_FILENAME= "df_max_flights_2019_06_24.csv"
else:
    FLIGHT_FILENAME= "df_max_delay_2019_06_07.csv"


def extend_list_to_length(lst, target_length, default_value):
    # Vérifier si la longueur actuelle de la liste est inférieure à la longueur cible
    if len(lst) < target_length:
        # Calculer combien d'éléments doivent être ajoutés
        elements_to_add = target_length - len(lst)
        # Étendre la liste avec la valeur par défaut
        lst.extend([default_value] * elements_to_add)
    return lst


def generate_flight_passengers(df_flight, df_connecting_pax):
    indexes = df_connecting_pax["arrival_flight_id"].unique()

    passenger_counts = {}
    print(df_flight)
    for idx in indexes:
        passenger_counts[idx] = df_flight.loc[idx, 'actual_passenger_count']

    transfer_times = {index: [] for index in indexes}

    for index, row in df_connecting_pax.iterrows():
        transfer_times[row["arrival_flight_id"]].extend([row['transfer_time_actual']] * int(row['nb_connecting_pax']))

    nb_total_pax = 0
    max_pax = 0
    min_pax = 10000
    for id_flight in transfer_times.keys():
        transfer_times[id_flight] = extend_list_to_length(transfer_times[id_flight], passenger_counts[id_flight],
                                                          MAX_TRANSFER_TIME)
        nb_pax = len(transfer_times[id_flight])
        nb_total_pax += nb_pax
        max_pax = max(max_pax, nb_pax)
        min_pax = min(min_pax, nb_pax)
    print(f"Total pax: {nb_total_pax} min pax:{min_pax} max pax: {max_pax}")
    passenger_counts_values = list(passenger_counts.values())

    bins = range(min(passenger_counts_values), max(passenger_counts_values) + 10, 10)
    hist, bin_edges = np.histogram(passenger_counts_values, bins=bins)

    # Créer un diagramme en barres
    plt.bar(bin_edges[:-1], hist, width=10, color='blue', edgecolor='black', align='edge')

    plt.xlabel('Number of passengers')
    plt.ylabel('Number of flights')
    plt.savefig(f"medias/connecting_passengers_distribution/{DAY_LABEL}/distribution_pax_flight_{DAY_LABEL}.png", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    # start_time = time.time()
    # nb_simu = 1
    # passengers_proportion = 0.8
    # # seat_allocation = SeatAllocation.RANDOM
    # # seat_allocation = SeatAllocation.CONNECTING_PRIORITY
    # for seat_allocation in [SeatAllocation.RANDOM, SeatAllocation.CONNECTING_PRIORITY]:
    #     10_simulations_40_pct_prereserved_3h_connecting_time = Simulation(quiet_mode=True, dummy_rows=2)
    #
    #     10_simulations_40_pct_prereserved_3h_connecting_time.set_custom_aircraft(n_rows=30, n_seats_left=3, n_seats_right=3)
    #     10_simulations_40_pct_prereserved_3h_connecting_time.set_passengers_proportion(0.9)
    #
    #     10_simulations_40_pct_prereserved_3h_connecting_time.set_passengers_proportion(passengers_proportion)
    #     10_simulations_40_pct_prereserved_3h_connecting_time.set_seat_allocation(seat_allocation)
    #     10_simulations_40_pct_prereserved_3h_connecting_time.run_multiple(nb_simu)
    #     10_simulations_40_pct_prereserved_3h_connecting_time.evaluate_missing_pax()
    # end_time = time.time()
    # print(f"Total simulation_time: {round(end_time-start_time,3)}s")
    df_flight = pd.read_csv(DATA_FOLDER + FLIGHT_FILENAME)
    df_connecting_pax = pd.read_csv(DATA_FOLDER + "connecting_passengers_old.csv")
    generate_flight_passengers(df_flight, df_connecting_pax)
