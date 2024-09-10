import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAX_PASSENGER_COUNT_VALUE = 200  # to not consider large flights

MIN_CONNECTING_PAX_RATIO = 0.3
MAX_CONNECTING_PAX_RATIO = 0.5

MINIMUM_TRANSFER_TIME = pd.Timedelta(minutes=45)
MAXIMUM_TRANSFER_TIME = pd.Timedelta(hours=4)

MINIMUM_WALKING_TIME = 10  # in minutes
MAXIMUM_WALKING_TIME = 95  # in minutes

new_column_names = {
    'Horaire théorique': 'scheduled_time',
    'Horaire bloc': 'block_time',
    'Horaire piste': 'runway_time',
    'Type de mouvement': 'movement_type',
    'Numéro de vol': 'flight_number',
    'Code IATA compagnie': 'airline_iata_code',
    'Code aéroport IATA': 'airport_iata_code',
    'Terminal': 'terminal',
    'Salle': 'gate',
    'Immatriculation': 'registration',
    'QFU': 'qfu',
    'Nombre de passagers réalisés': 'actual_passenger_count'
}


def crop_flight_schedule():
    filename = DATA_FOLDER + "Thesis-SariaRCDG.csv"
    df = pd.read_csv(filename)
    df = df.rename(columns=new_column_names)
    df = convert_str_date_to_date_time(df)

    df['delay_seconds'] = (df['block_time'] - df['scheduled_time']).dt.total_seconds()
    df['day'] = df['scheduled_time'].dt.date

    vols_par_jour = df.groupby('day').size()
    jour_max_vols = vols_par_jour.idxmax()

    df_max_vols = df[df['day'] == jour_max_vols]

    df_positive_delay = df[df['delay_seconds'] > 0]
    retard_par_jour = df_positive_delay.groupby('day')['delay_seconds'].sum()
    jour_max_retard = retard_par_jour.idxmax()

    df_max_retard = df[df['day'] == jour_max_retard]

    df_max_vols.to_csv(DATA_FOLDER + "df_max_flights_2019_06_24.csv", index=False)
    df_max_retard.to_csv(DATA_FOLDER + "df_max_delay_2019_06_07.csv", index=False)


def convert_str_date_to_date_time(df):
    df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
    df['block_time'] = pd.to_datetime(df['block_time'])
    return df


def generate_number_of_connecting_passengers(total_passengers):
    """
    Génère le nombre de passagers en correspondance en fonction du nombre total de passagers.
    Peut être amélioré en utilisant une distribution plus sophistiquée.
    """
    proportion = np.random.uniform(MIN_CONNECTING_PAX_RATIO, MAX_CONNECTING_PAX_RATIO)
    return int(proportion * total_passengers)


def process_all_arrivals(filename):
    df = pd.read_csv(filename)
    df = convert_str_date_to_date_time(df)

    df = df[df["airline_iata_code"] == "AF"]
    df_arrivals = df[(df["movement_type"] == "Arrivée") & (df["actual_passenger_count"] < MAX_PASSENGER_COUNT_VALUE)]
    df_departures = df[df["movement_type"] == "Départ"]

    connecting_pax_df = pd.DataFrame(columns=[
        'arrival_flight_id', 'departure_flight_id', 'nb_connecting_pax',
        'transfer_time_theoretical', 'transfer_time_actual', 'min_walking_time'
    ])

    df_arrival_flights = pd.DataFrame(columns=["arrival_flight_id", "actual_passenger_count"])

    for i, arrival_row in df_arrivals.iterrows():
        print(f"Processing arrival flight {i} .... ")
        arrival_row, connecting_pax_df, nb_connecting_pax = generate_connecting_passenger_one_flight(arrival_row, df_departures,
                                                                                  connecting_pax_df)
        id_arrival_flight, nb_pax = arrival_row.name, arrival_row['actual_passenger_count']
        new_entry = pd.DataFrame({"arrival_flight_id": [id_arrival_flight], "actual_passenger_count": [nb_pax]})
        df_arrival_flights = pd.concat([df_arrival_flights, new_entry], ignore_index=True)

    df_arrival_flights.to_csv(DATA_FOLDER + "df_arrival_flights.csv", index=False)
    connecting_pax_df.to_csv(DATA_FOLDER + "connecting_passengers.csv", index=False)
    return connecting_pax_df


def generate_connecting_passenger_one_flight(arrival_row, df_departures: pd.DataFrame, connecting_pax_df: pd.DataFrame):
    """
    Assigne les passagers en correspondance à un vol de départ parmi les candidats identifiés.
    Stocke cette information dans un nouvel objet DataFrame `connecting_pax_df`.
    """
    arrival_time = arrival_row['scheduled_time']
    arrival_block_time = arrival_row['block_time']

    # Filtrer les vols candidats selon les contraintes de temps de transfert
    departures_candidates = df_departures[
        (df_departures['scheduled_time'] >= arrival_time + MINIMUM_TRANSFER_TIME) &
        (df_departures['scheduled_time'] <= arrival_time + MAXIMUM_TRANSFER_TIME)
        ]

    nb_connecting_pax = generate_number_of_connecting_passengers(arrival_row['actual_passenger_count'])
    nb_actual_connecting_pax = 0
    # Groupes de passagers par vol de départ
    if len(departures_candidates) > 0 & (nb_connecting_pax >= 1):
        # Calculer combien de groupes de passagers nous avons besoin
        max_groups = min(len(departures_candidates), nb_connecting_pax // 5 + 1)
        if max_groups == 1:
            num_groups = 1
        else:
            num_groups = np.random.randint(1, max_groups)
        connected_flights_indices = np.random.choice(departures_candidates.index, size=num_groups, replace=False)
        connecting_passengers_dict = dict((zip(connected_flights_indices, np.zeros(len(connected_flights_indices)))))
        for i in range(nb_connecting_pax):
            id_flight = np.random.choice(connected_flights_indices)
            connecting_passengers_dict[id_flight] += 1
        for flight_idx in connected_flights_indices:
            if connecting_passengers_dict[flight_idx] >0:
                departure_row = df_departures.loc[flight_idx]
                transfer_time_theoretical = (departure_row['scheduled_time'] - arrival_time).total_seconds()
                transfer_time_actual = (departure_row['block_time'] - arrival_block_time).total_seconds()

                min_walking_time_seconds = np.random.uniform(MINIMUM_WALKING_TIME * 60,
                                                             np.min([MAXIMUM_WALKING_TIME * 60,
                                                                     int(transfer_time_theoretical) - 10 * 60]))

                new_entry = pd.DataFrame({
                    'arrival_flight_id': [arrival_row.name],
                    'departure_flight_id': [flight_idx],
                    'nb_connecting_pax': [int(connecting_passengers_dict[flight_idx])],
                    'transfer_time_theoretical': [int(transfer_time_theoretical)],
                    'transfer_time_actual': [int(transfer_time_actual)],
                    'min_walking_time': [int(min_walking_time_seconds)]
                })
                nb_actual_connecting_pax += int(connecting_passengers_dict[flight_idx])
                connecting_pax_df = pd.concat([connecting_pax_df, new_entry], ignore_index=True)

    arrival_row['connecting_passengers'] = connecting_passengers_dict

    return arrival_row, connecting_pax_df, nb_actual_connecting_pax


def display_connecting_pax_distribution():
    df = pd.read_csv(DATA_FOLDER + 'connecting_passengers.csv')

    df['transfer_time_theoretical_minutes'] = df['transfer_time_theoretical'] / 60
    df['transfer_time_actual_minutes'] = df['transfer_time_actual'] / 60

    min_transfer_time = min(df['transfer_time_theoretical_minutes'].min(), df['transfer_time_actual_minutes'].min())
    max_transfer_time = max(df['transfer_time_theoretical_minutes'].max(), df['transfer_time_actual_minutes'].max())
    bins = np.arange(np.floor(min_transfer_time), np.ceil(max_transfer_time) + 10, 10)

    theoretical_histogram, _ = np.histogram(df['transfer_time_theoretical_minutes'], bins=bins,
                                            weights=df['nb_connecting_pax'])
    actual_histogram, _ = np.histogram(df['transfer_time_actual_minutes'], bins=bins, weights=df['nb_connecting_pax'])

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    title_fontsize = 18
    label_fontsize = 16
    tick_fontsize = 14

    ax[0].bar(bins[:-1], theoretical_histogram, width=np.diff(bins), edgecolor='black', alpha=0.7)
    ax[0].set_title('Distribution of theoretical transfer times', fontsize=title_fontsize)
    ax[0].set_xlabel('Transfer time (min)', fontsize=label_fontsize)
    ax[0].set_ylabel('Total number of \n connecting passengers', fontsize=label_fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    ax[1].bar(bins[:-1], actual_histogram, width=np.diff(bins), edgecolor='black', alpha=0.7, color='orange')
    ax[1].set_title('Distribution of actual transfer times', fontsize=title_fontsize)
    ax[1].set_xlabel('Transfer time (minutes)', fontsize=label_fontsize)
    ax[1].set_ylabel('Total number of \n connecting passengers', fontsize=label_fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    plt.tight_layout()
    plt.savefig(f"medias/connecting_passengers_distribution/{DAY_LABEL}/transfer_time_distribution_{DAY_LABEL}.png")
    # plt.show()


def compute_buffer_times(df):
    df['buffer_time_theoretical_seconds'] = df['transfer_time_theoretical'] - df['min_walking_time']
    df['buffer_time_actual_seconds'] = df['transfer_time_actual'] - df['min_walking_time']
    df['buffer_time_theoretical_minutes'] = ((df['transfer_time_theoretical'] - df['min_walking_time']) / 60).astype(int)
    df['buffer_time_actual_minutes'] = ((df['transfer_time_actual'] - df['min_walking_time']) / 60).astype(int)
    return df

def display_buffer_time_distribution():
    df = pd.read_csv(DATA_FOLDER + 'connecting_passengers.csv')

    df = compute_buffer_times(df)

    min_buffer_time = min(df['buffer_time_theoretical_minutes'].min(), df['buffer_time_actual_minutes'].min())
    max_buffer_time = max(df['buffer_time_theoretical_minutes'].max(), df['buffer_time_actual_minutes'].max())
    bins = np.arange(np.floor(min_buffer_time), np.ceil(max_buffer_time) + 10, 10)

    theoretical_histogram, _ = np.histogram(df['buffer_time_theoretical_minutes'], bins=bins,
                                            weights=df['nb_connecting_pax'])
    actual_histogram, _ = np.histogram(df['buffer_time_actual_minutes'], bins=bins, weights=df['nb_connecting_pax'])

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    title_fontsize = 18
    label_fontsize = 16
    tick_fontsize = 14

    ax[0].bar(bins[:-1], theoretical_histogram, width=np.diff(bins), edgecolor='black', alpha=0.7)
    ax[0].set_title('Distribution of theoretical buffer times', fontsize=title_fontsize)
    ax[0].set_xlabel('Buffer time (min)', fontsize=label_fontsize)
    ax[0].set_ylabel('Total number of \n connecting passengers', fontsize=label_fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    ax[1].bar(bins[:-1], actual_histogram, width=np.diff(bins), edgecolor='black', alpha=0.7, color='orange')
    ax[1].set_title('Distribution of actual buffer times', fontsize=title_fontsize)
    ax[1].set_xlabel('Buffer time (minutes)', fontsize=label_fontsize)
    ax[1].set_ylabel('Total number of \n connecting passengers', fontsize=label_fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    plt.tight_layout()
    # plt.savefig(f"medias/connecting_passengers_distribution/{DAY_LABEL}/buffer_time_distribution_{DAY_LABEL}.png")
    plt.show()


if __name__ == "__main__":
    DAY_LABEL = "max_delay_day"
    filename = "df_max_delay_2019_06_07.csv"

    # DAY_LABEL = "max_flight_day"
    # filename = "df_max_flights_2019_06_24.csv"

    DATA_FOLDER = f"data/{DAY_LABEL}/"

    connecting_pax_df = process_all_arrivals(DATA_FOLDER + filename)

    # print("Connecting passengers DataFrame:")
    # print(connecting_pax_df)
    display_connecting_pax_distribution()
    display_buffer_time_distribution()
