import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random


def generate_probability_matrix(nb_rows, nb_columns):
    grid = np.zeros((nb_rows, nb_columns))

    # Define probabilities
    # Higher probabilities at the top and closer to the aisle
    for r in range(nb_rows):
        for c in range(nb_columns):
            if c == 3:  # Aisle seats
                grid[r, c] = 0.0  # Probability 0% for aisles
            elif c == 1 or c == 5:  # Middle seats
                grid[r, c] = 0.3 * (1 - r / nb_rows)  # Lower probability for middle seats
            else:  # Window seats
                grid[r, c] = 0.6 * (1 - r / nb_rows)  # Higher probability towards the top
    return grid


def assign_prereserved_seats(nb_passengers_prereserved, total_passengers, probability_matrix):
    prereserved_seats = []
    remaining_seats = []
    nb_rows, nb_columns = probability_matrix.shape
    total_seats = nb_rows * nb_columns

    if total_passengers > total_seats:
        raise ValueError("Number of passengers exceeds the number of available seats")

    flat_probs = probability_matrix.flatten()
    flat_indices = list(range(len(flat_probs)))

    # Normalize probabilities for pre-reserved seats
    total_prob = sum(flat_probs)
    normalized_probs = [p / total_prob for p in flat_probs]

    # Randomly assign seats for pre-reserved passengers
    assigned_indices_prereserved = []
    while len(assigned_indices_prereserved) < nb_passengers_prereserved:
        index = random.choices(flat_indices, weights=normalized_probs, k=1)[0]
        if index not in assigned_indices_prereserved:
            assigned_indices_prereserved.append(index)
            flat_probs[index] = -1  # Mark this seat as taken

    for index in assigned_indices_prereserved:
        row = index // nb_columns
        col = index % nb_columns
        prereserved_seats.append((row, col))

    # Collect remaining available seats
    available_indices = [i for i in flat_indices if flat_probs[i] > -1]

    # Uniformly assign seats to remaining passengers from available seats
    remaining_passengers = total_passengers - nb_passengers_prereserved
    assigned_indices_remaining = random.sample(available_indices, remaining_passengers)

    for index in assigned_indices_remaining:
        row = index // nb_columns
        col = index % nb_columns
        remaining_seats.append((row, col))

    # Ensure no duplicates
    if len(set(prereserved_seats + remaining_seats)) != len(prereserved_seats) + len(remaining_seats):
        raise ValueError("Duplicate seats found in the assignment")

    return prereserved_seats, remaining_seats


def check_for_duplicates(seats):
    """Check for duplicate seats."""
    return len(seats) != len(set(seats))


def display_prereserved_seat_probability():
    nb_rows, nb_cols = 20, 7

    grid = generate_probability_matrix(nb_rows, nb_cols)

    grid = grid / np.sum(grid) * 100
    grid = np.transpose(grid)

    vmax = int(np.ceil(np.max(grid)))
    cmap = plt.get_cmap('Reds')
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(16, 8))  # Increased width for wider cells
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

    # plt.savefig('medias/prereserved_seat_probability_distribution.png', bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    nb_rows, nb_cols = 20, 7

    grid = generate_probability_matrix(nb_rows, nb_cols)

    nb_pax_prereserved = 40
    nb_total_pax = 50

    res = assign_prereserved_seats(nb_pax_prereserved, nb_total_pax, probability_matrix=grid)
    print(res)
