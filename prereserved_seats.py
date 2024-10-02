import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random


def generate_probability_matrix(nb_rows, nb_columns):
    grid = np.zeros((nb_rows, nb_columns))

    for r in range(nb_rows):
        for c in range(nb_columns):
            if c == 3:  # Aisle
                grid[r, c] = 0.0
            elif c == 1 or c == 5:  # Middle seats
                grid[r, c] = 0.3 * (1 - r / nb_rows)  # Lower probability for middle seats
            else:  # Window seats
                grid[r, c] = 0.6 * (1 - r / nb_rows)  # Higher probability towards the top
    return grid


def assign_seats(nb_passengers_prereserved, total_passengers, probability_matrix):
    nb_rows, nb_columns = probability_matrix.shape
    total_seats = nb_rows * nb_columns

    if total_passengers > total_seats:
        raise ValueError("Number of passengers exceeds the number of available seats")

    flat_probs = probability_matrix.flatten()
    flat_indices = list(range(total_seats))

    normalized_probs = [p / sum(flat_probs) for p in flat_probs]

    assigned_indices_prereserved = random.sample([i for i in flat_indices if normalized_probs[i] > 0], nb_passengers_prereserved)

    prereserved_seats = [(i // nb_columns, i % nb_columns) for i in assigned_indices_prereserved]

    available_indices = [i for i in flat_indices if i not in assigned_indices_prereserved]
    assigned_indices_remaining = random.sample(available_indices, total_passengers - nb_passengers_prereserved)

    remaining_seats = [(i // nb_columns, i % nb_columns) for i in assigned_indices_remaining]
    check_for_duplicates(prereserved_seats + remaining_seats)
    return prereserved_seats, remaining_seats


def check_for_duplicates(seats):
    """Check for duplicate seats and raise ValueError if duplicates are found."""
    if len(seats) != len(set(seats)):
        raise ValueError("Duplicate seats found !!")





