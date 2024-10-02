import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.colors as mcolors

from utils import configloader
import prereserved_seats

# DAY_LABEL = "max_delay_day"
DAY_LABEL = "max_flight_day"

RESULT_FOLDER = f"{DAY_LABEL}/10_simulations_20_pct_prereserved_3h_connecting_time/"

config_file_path = os.path.join(os.path.dirname(os.getcwd()), "configuration_deboarding.yaml")
configuration = configloader.ConfigLoader(config_file_path)
GATE_CLOSE_TIME = configuration.gate_close_time


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

    fig, ax = plt.subplots(figsize=(16, 6))

    for key in y_missed_pax.keys():
        print(key, np.mean(y_missed_pax[key]))

    ax.boxplot(y_missed_pax.values(), patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='darkblue', linewidth=1.5),
               medianprops=dict(color='red', linewidth=1.5),
               whiskerprops=dict(color='black', linewidth=1.5),
               capprops=dict(color='black', linewidth=1.5))

    x_tick_labels = ["Random-Courtesy", "Random-Aisle", "Connecting-Courtesy", "Connecting-Aisle"]
    ax.set_xticklabels(x_tick_labels, fontsize=18)
    plt.tick_params(axis='y', labelsize=16)

    ax.set_ylabel('Nb of passengers missing their flights', fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.6)

    if savefig:
        plt.savefig(f"medias/boxplot/{DAY_LABEL}_missed_pax_{int(GATE_CLOSE_TIME / 60)}min_gate_closure.png",
                    bbox_inches='tight')
    else:
        plt.show()


def display_boxplot_deboarding_time(savefig=False):
    files = get_all_files_in_folder(RESULT_FOLDER)
    y_missed_pax = {}

    for csv_file in files:
        df = pd.read_csv(RESULT_FOLDER + csv_file)
        indexes = df[['Seat Allocation', 'Deboarding Strategy']].agg('-'.join, axis=1).values
        values_missed_pax = df['List Deboarding Time'].apply(lambda x: eval(x)).values
        if len(y_missed_pax) == 0:
            y_missed_pax = dict(zip(indexes, [e for e in values_missed_pax]))
        else:
            for i in range(len(indexes)):
                y_missed_pax[indexes[i]] += values_missed_pax[i]

    # New color palette
    colors = ['darkturquoise', 'blue', 'black', 'blue']  # Improved color palette
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.boxplot(y_missed_pax.values(), patch_artist=True,
               boxprops=dict(facecolor=colors[0], color=colors[1], linewidth=2),
               medianprops=dict(color=colors[2], linewidth=2),
               whiskerprops=dict(color=colors[3], linewidth=2),
               capprops=dict(color=colors[3], linewidth=2))

    x_tick_labels = ["Random-Courtesy", "Random-Aisle", "Connecting-Courtesy", "Connecting-Aisle"]
    ax.set_xticklabels(x_tick_labels, fontsize=18)
    plt.tick_params(axis='y', labelsize=16)

    ax.set_ylabel('Passenger disembarkation time', fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.6)

    if savefig:
        plt.savefig(f"medias/boxplot/{DAY_LABEL}_deboarding_time_{int(GATE_CLOSE_TIME / 60)}min_gate_closure.png",
                    bbox_inches='tight')
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

    for key in y_deboarding_time.keys():
        y_deboarding_time[key] /= len(files)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.bar(y_deboarding_time.keys(), y_deboarding_time.values(), color="darkturquoise")

    ax.set_ylabel('Passenger Deboarding Time (seconds)', fontsize=16, weight='bold')

    x_tick_labels = ["Random-Courtesy", "Random-Aisle", "Connecting-Courtesy", "Connecting-Aisle"]
    ax.set_xticklabels(x_tick_labels, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='y', labelsize=16)

    plt.tight_layout()

    if savefig:
        plt.savefig(f"medias/barplot/{DAY_LABEL}_deboarding_time_{int(GATE_CLOSE_TIME / 60)}min_gate_closure.png",
                    bbox_inches='tight')
    else:
        plt.show()


def display_bar_plot_missed_pax_prereserved_seats(folder_path, savefig=False):
    files = get_all_files_in_folder(folder_path)
    y_missed_pax = {}

    # Extracting the deboarding time data
    for i, csv_file in enumerate(files):
        percentage = int(csv_file.split("_")[2][:-7])
        df = pd.read_csv(folder_path + csv_file)
        print(percentage)
        y_missed_pax[percentage] = df['Total Missed Pax'].values[0]

    # Plotting the improved bar plot
    fig, ax = plt.subplots(figsize=(14, 7))

    print(y_missed_pax)

    # Using a gradient color for the bars
    ax.bar(y_missed_pax.keys(), y_missed_pax.values(), width=15, color="darkcyan")

    xticks = [0, 20, 40, 60, 80, 100]
    ax.set_xticks(xticks)
    x_tick_labels = [f"{i}%" for i in xticks]
    ax.set_xticklabels(x_tick_labels, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='y', labelsize=14)  # Change '14' to the desired size

    ax.set_ylabel('Total number of passenger missing their flights', fontsize=16)
    ax.set_xlabel('Percentage of pre-reserved seats', fontsize=16)
    if savefig:
        plt.savefig(f"medias/barplot/{DAY_LABEL}_missed_pax_prereserved_seats_evolution.png", bbox_inches='tight')
    else:
        plt.show()


def display_prereserved_seat_probability(nb_rows=20, nb_cols=6, savefig=False):
    nb_cols += 1  # to account for the aisle
    grid = prereserved_seats.generate_probability_matrix(nb_rows, nb_cols)

    grid = grid / np.sum(grid) * 100
    grid = np.transpose(grid)

    vmax = int(np.ceil(np.max(grid)))
    cmap = plt.get_cmap('Reds')
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(20, 6))
    cax = ax.matshow(grid, cmap=cmap, norm=norm)

    ft = 14

    cbar = plt.colorbar(cax, ax=ax, orientation='horizontal')
    cbar.set_label('Selection Probability (%)', fontsize=ft)

    xticks = [i for i in range(vmax + 1)]
    cbar.set_ticks(xticks)
    cbar.set_ticklabels([f"{x}%" for x in xticks], fontsize=ft)
    cbar.ax.invert_yaxis()  # Highest probability at the top

    ax.set_yticks(np.arange(-0.5, nb_cols, 1), minor=True)
    ax.set_xticks(np.arange(-0.5, nb_rows, 1), minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    ax.set_yticks(np.arange(nb_cols))
    ax.set_yticklabels(['Window-side', 'Middle', 'Aisle-side', '', 'Aisle-side', 'Middle', 'Window-side'], fontsize=ft)
    ax.set_xticks(np.arange(nb_rows))
    ax.set_xticklabels(np.arange(1, nb_rows + 1), fontsize=11)
    ax.xaxis.set_ticks_position('bottom')

    for r in range(nb_rows):
        for c in range(nb_cols):
            if grid[c, r] > 0:
                ax.text(r, c, f'{grid[c, r]:.1f}%', ha='center', va='center', color='black', fontsize=11)

    plt.xlabel('Row number', fontsize=ft)
    if savefig:
        plt.savefig('medias/prereserved_seat_probability_distribution.png', bbox_inches="tight")
    else:
        plt.show()


if __name__ == '__main__':
    display_missed_pax_strategies(savefig=False)
    display_boxplot_deboarding_time(savefig=False)
    display_deboarding_time_bar(savefig=False)
    # display_deboarding_time_line()
    folder_result = "max_flight_day/simulations_evolution_percentage_prereserved_seats/"
    display_bar_plot_missed_pax_prereserved_seats(folder_result, savefig=False)
    display_prereserved_seat_probability(savefig=False)
