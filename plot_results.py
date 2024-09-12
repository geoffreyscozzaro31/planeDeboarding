import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import numpy as np

DAY_LABEL = "max_flight_day"

RESULT_FOLDER = f"results/{DAY_LABEL}/10_simulations_20_pct_prereserved_3h_connecting_time/"


def get_all_files_in_folder(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


def get_all_files_in_folder(folder):
    """Returns a list of all files in the given folder."""
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


def display_missed_pax_strategies(savefig=False):
    files = get_all_files_in_folder(RESULT_FOLDER)
    y_missed_pax = {}

    for csv_file in files:
        df = pd.read_csv(RESULT_FOLDER + csv_file)
        indexes = df[['Seat Allocation', 'Deboarding Strategy']].agg('-'.join, axis=1).values
        values_missed_pax = df['Total Missed Pax'].values
        if len(y_missed_pax) == 0:
            y_missed_pax = dict(zip(indexes, [[e] for e in values_missed_pax]))
        else:
            for i in range(len(indexes)):
                y_missed_pax[indexes[i]].append(df['Total Missed Pax'].values[i])

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.boxplot(y_missed_pax.values(), patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='darkblue'),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(color='black', linewidth=1.5),
               capprops=dict(color='black', linewidth=1.5))

    x_tick_labels = ["Courtesy-Random", "Aisle-Random", "Courtesy-Connecting pax", "Aisle-Connecting pax"]
    # Rotate x-tick labels to fit
    ax.set_xticklabels(x_tick_labels, fontsize=14)

    ax.set_ylabel('Total Passengers Missing their Flights', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)

    if savefig:
        plt.savefig(f"medias/boxplot/{DAY_LABEL}_missed_pax_.png", bbox_inches='tight')
    else:
        plt.show()


def display_boxplot_deboarding_time(savefig=False):
    files = get_all_files_in_folder(RESULT_FOLDER)
    y_missed_pax = {}

    for csv_file in files:
        df = pd.read_csv(RESULT_FOLDER + csv_file)
        indexes = df[['Seat Allocation', 'Deboarding Strategy']].agg('-'.join, axis=1).values
        values_missed_pax = df['Average Deboarding Time'].values
        if len(y_missed_pax) == 0:
            y_missed_pax = dict(zip(indexes, [[e] for e in values_missed_pax]))
        else:
            for i in range(len(indexes)):
                y_missed_pax[indexes[i]].append(df['Average Deboarding Time'].values[i])

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.boxplot(y_missed_pax.values(), patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='darkblue'),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(color='black', linewidth=1.5),
               capprops=dict(color='black', linewidth=1.5))

    x_tick_labels = ["Courtesy-Random", "Aisle-Random", "Courtesy-Connecting pax", "Aisle-Connecting pax"]
    # Rotate x-tick labels to fit
    ax.set_xticklabels(x_tick_labels, fontsize=14)

    ax.set_ylabel('Average Deboarding Time', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)

    if savefig:
        plt.savefig(f"medias/boxplot/{DAY_LABEL}_missed_pax_.png", bbox_inches='tight')
    else:
        plt.show()


def display_deboarding_time_bar(savefig=False):
    files = get_all_files_in_folder(RESULT_FOLDER)
    y_deboarding_time = {}

    for csv_file in files:
        df = pd.read_csv(RESULT_FOLDER + csv_file)
        indexes = df[['Seat Allocation', 'Deboarding Strategy']].agg('-'.join, axis=1).values
        values_deboarding_time = df['Average Deboarding Time'].values
        if len(y_deboarding_time) == 0:
            y_deboarding_time = dict(zip(indexes, [e for e in values_deboarding_time]))
        else:
            for i in range(len(indexes)):
                y_deboarding_time[indexes[i]] += df['Average Deboarding Time'].values[i]
    print(y_deboarding_time)

    for key in y_deboarding_time.keys():
        y_deboarding_time[key]/= len(files)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(y_deboarding_time.keys(), y_deboarding_time.values(), color='darkblue', alpha=0.6)

    x_tick_labels = ["Courtesy-Random", "Aisle-Random", "Courtesy-Connecting pax", "Aisle-Connecting pax"]
    ax.set_xticklabels(x_tick_labels, fontsize=14)

    ax.set_ylabel('Average Deboarding Time (seconds)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if savefig:
        plt.savefig(f"medias/barplot/{DAY_LABEL}_deboarding_time.png", bbox_inches='tight')
    else:
        plt.show()



if __name__ == '__main__':
    display_missed_pax_strategies(savefig=True)
    # display_boxplot_deboarding_time()
    display_deboarding_time_bar(savefig=True)
    # display_deboarding_time_line()
