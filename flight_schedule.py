import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DAY_LABEL = "max_delay_day"
# DAY_LABEL = "max_flight_day"
DATA_FOLDER = f"data/{DAY_LABEL}/"



MINIMUM_TRANSFER_TIME = pd.Timedelta(minutes=45)
MAXIMUM_TRANSFER_TIME = pd.Timedelta(hours=4)

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

    # Calcul du délai en secondes
    df['delay_seconds'] = (df['block_time'] - df['scheduled_time']).dt.total_seconds()
    df['day'] = df['scheduled_time'].dt.date

    # 1. Trouver le jour avec le plus de vols
    vols_par_jour = df.groupby('day').size()
    jour_max_vols = vols_par_jour.idxmax()

    # Extraire les vols du jour avec le plus de vols
    df_max_vols = df[df['day'] == jour_max_vols]

    # 2. Trouver le jour avec le plus de retard total
    df_positive_delay = df[df['delay_seconds'] > 0]
    retard_par_jour = df_positive_delay.groupby('day')['delay_seconds'].sum()
    jour_max_retard = retard_par_jour.idxmax()

    # Extraire les vols du jour avec le plus de retard total
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
    return int(0.4 * total_passengers)


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

    # Groupes de passagers par vol de départ
    if len(departures_candidates)>0 & (nb_connecting_pax>=1):
        # Calculer combien de groupes de passagers nous avons besoin
        max_groups = min(len(departures_candidates), nb_connecting_pax // 5 + 1)
        if max_groups == 1:
            num_groups = 1
        else:
            num_groups = np.random.randint(1,max_groups)
        connected_flights_indices = np.random.choice(departures_candidates.index, size=num_groups, replace=False)
        connecting_passengers_dict = dict((zip(connected_flights_indices, np.zeros(len(connected_flights_indices)))))
        for i in range(nb_connecting_pax):
            id_flight = np.random.choice(connected_flights_indices)
            connecting_passengers_dict[id_flight] +=1

        for flight_idx in connected_flights_indices:
            departure_row = df_departures.loc[flight_idx]
            transfer_time_theoretical = (departure_row['scheduled_time'] - arrival_time).total_seconds()
            transfer_time_actual = (departure_row['block_time'] - arrival_block_time).total_seconds()



            new_entry = pd.DataFrame({
                'arrival_flight_id': [arrival_row.name],  # Assurez-vous que `arrival_row` a un ID unique
                'departure_flight_id': [flight_idx],
                'nb_connecting_pax': [1],
                'transfer_time_theoretical': [transfer_time_theoretical],
                'transfer_time_actual': [transfer_time_actual]
            })
            connecting_pax_df = pd.concat([connecting_pax_df, new_entry], ignore_index=True)

    arrival_row['connecting_passengers'] = connecting_passengers_dict

    return arrival_row, connecting_pax_df


def process_all_arrivals(filename):
    df = pd.read_csv(filename)
    df = convert_str_date_to_date_time(df)

    df = df[df["airline_iata_code"] == "AF"]
    df_arrivals = df[df["movement_type"] == "Arrivée"]
    df_departures = df[df["movement_type"] == "Départ"]

    connecting_pax_df = pd.DataFrame(columns=[
        'arrival_flight_id', 'departure_flight_id', 'nb_connecting_pax',
        'transfer_time_theoretical', 'transfer_time_actual'
    ])

    # Itérer sur chaque vol à l'arrivée
    for i, arrival_row in df_arrivals.iterrows():
        # Appeler la méthode pour générer les passagers connectés pour ce vol
        print(f"Processing arrival flight {i} .... ")
        arrival_row, connecting_pax_df = generate_connecting_passenger_one_flight(arrival_row, df_departures,
                                                                                  connecting_pax_df)

    connecting_pax_df.to_csv(DATA_FOLDER + "connecting_passengers.csv", index=False)
    return connecting_pax_df


def display_connecting_pax_distribution():
    df = pd.read_csv(DATA_FOLDER+'connecting_passengers.csv')

    # Convertir les temps de transfert de secondes en minutes
    df['transfer_time_theoretical_minutes'] = df['transfer_time_theoretical'] / 60
    df['transfer_time_actual_minutes'] = df['transfer_time_actual'] / 60

    # Créer les histogrammes en utilisant les poids (nb_connecting_pax)
    # Créer les bins pour l'histogramme
    min_transfer_time = min(df['transfer_time_theoretical_minutes'].min(), df['transfer_time_actual_minutes'].min())
    max_transfer_time = max(df['transfer_time_theoretical_minutes'].max(), df['transfer_time_actual_minutes'].max())
    bins = np.arange(np.floor(min_transfer_time), np.ceil(max_transfer_time) + 10, 10)

    # Calculer le nombre total de passagers connectés pour chaque intervalle
    theoretical_histogram, _ = np.histogram(df['transfer_time_theoretical_minutes'], bins=bins,
                                            weights=df['nb_connecting_pax'])
    actual_histogram, _ = np.histogram(df['transfer_time_actual_minutes'], bins=bins, weights=df['nb_connecting_pax'])

    # Créer la figure et les axes pour les sous-graphes
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Paramètres pour agrandir les polices
    title_fontsize = 18
    label_fontsize = 16
    tick_fontsize = 14

    # Tracer l'histogramme des temps de transfert théorique
    ax[0].bar(bins[:-1], theoretical_histogram, width=np.diff(bins), edgecolor='black', alpha=0.7)
    ax[0].set_title('Distribution of Theoretical Transfer Times', fontsize=title_fontsize)
    ax[0].set_xlabel('Transfer Time (minutes)', fontsize=label_fontsize)
    ax[0].set_ylabel('Total Number of Connecting Passengers', fontsize=label_fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Tracer l'histogramme des temps de transfert réels
    ax[1].bar(bins[:-1], actual_histogram, width=np.diff(bins), edgecolor='black', alpha=0.7, color='orange')
    ax[1].set_title('Distribution of Actual Transfer Times', fontsize=title_fontsize)
    ax[1].set_xlabel('Transfer Time (minutes)', fontsize=label_fontsize)
    ax[1].set_ylabel('Number of Connecting Passengers', fontsize=label_fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Ajuster l'affichage
    plt.tight_layout()
    plt.savefig(f"medias/connecting_passengers_distribution/{DAY_LABEL}/transfer_time_distribution_{DAY_LABEL}.png")
    # plt.show()




if __name__ == "__main__":
    # filename = DATA_FOLDER + "df_max_flights_2019_06_24.csv"
    filename = DATA_FOLDER + "df_max_delay_2019_06_07.csv"
    # connecting_pax_df = process_all_arrivals(filename)

    # print("Connecting passengers DataFrame:")
    # print(connecting_pax_df)
    display_connecting_pax_distribution()