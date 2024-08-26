import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from enum import IntEnum

FILE_PATH = 'medias/deboarding/random_1.0_32_3_history_0.txt'

NB_STEPS = 6
INTERVAL = 2  # in seconds


class State(IntEnum):
    SEATED = 1
    STAND_UP_FROM_SEAT = 2
    MOVE_WAIT = 3
    MOVE_FROM_ROW = 4


def read_history(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract general parameters
    n_rows, dummy_rows, n_seats_left, n_seats_right, n_passengers, n_baggage_events = map(int, lines[0].split())

    # Initialize history
    history = {}

    line_idx = 1
    for t in range(10000):  # Maximum steps, adjust based on your data
        if line_idx >= len(lines):
            break
        n_entries = int(lines[line_idx].strip())
        line_idx += 1
        if n_entries == 0:
            continue
        history[t] = []
        for _ in range(n_entries):
            entry = list(map(int, lines[line_idx].strip().split()))
            history[t].append(entry)
            line_idx += 1

    return history, n_rows, dummy_rows, n_seats_left, n_seats_right, n_passengers, n_baggage_events


# Define the colors for each state
state_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
cmap = ListedColormap(state_colors)
norm = plt.Normalize(vmin=0, vmax=len(State) - 0.5)


# Function to interpolate positions
def interpolate_positions(start, end, steps):
    return np.linspace(start, end, steps)


# Function to generate fictive steps
def generate_fictive_steps(history, steps):
    new_history = {}
    for t, state in history.items():
        for step in range(steps):
            new_t = t * steps + step
            new_history[new_t] = []
            for entry in state:
                _, pax_id, x_pos, y_pos, pax_state = entry

                # Find previous state
                if t > 0:
                    prev_state = history[t - 1]
                    prev_entry = next((e for e in prev_state if e[1] == pax_id), entry)
                    prev_x_pos, prev_y_pos = prev_entry[2], prev_entry[3]
                else:
                    prev_x_pos, prev_y_pos = x_pos, y_pos

                # Interpolate positions
                x_interp = interpolate_positions(prev_x_pos, x_pos, steps)
                y_interp = interpolate_positions(prev_y_pos, y_pos, steps)

                # Add interpolated entry
                new_entry = [new_t, pax_id, x_interp[step], y_interp[step], pax_state]
                new_history[new_t].append(new_entry)

    return new_history


def generate_animation(save_animation=False):
    history, n_rows, dummy_rows, n_seats_left, n_seats_right, n_passengers, n_baggage_events = read_history(FILE_PATH)

    history_with_fictive_steps = generate_fictive_steps(history, NB_STEPS)

    fig, ax = plt.subplots(figsize=(6, 12))
    ax.set_xlim(-n_seats_left - 1, n_seats_right + 1)
    ax.set_ylim(-1, n_rows + dummy_rows)
    ax.set_xticks(np.arange(-n_seats_left, n_seats_right + 1))
    ax.set_yticks(np.arange(n_rows + dummy_rows))
    plt.gca().invert_yaxis()

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel('Seat Columns')
    ax.set_ylabel('Rows')
    ax.set_title('Passenger Deboarding Animation')

    scat = ax.scatter([], [], c=[], cmap=cmap, norm=plt.Normalize(vmin=0.5, vmax=len(State) + 0.5), s=100)
    cbar = plt.colorbar(scat, ax=ax, ticks=np.arange(1, len(State) + 1))
    cbar.ax.set_yticklabels([state.name for state in State])
    cbar.set_label('Passenger State')

    def plot_state(state):
        x = []
        y = []
        c = []

        for entry in state:
            _, pax_id, x_pos, y_pos, pax_state = entry
            if pax_state != 9:  # Not deboarded
                x.append(x_pos)
                y.append(y_pos)
                c.append(pax_state)

        return np.c_[x, y], np.array(c)

    # Update function for animations
    def update(frame):
        state = history_with_fictive_steps[frame]
        positions, colors = plot_state(state)
        scat.set_offsets(positions)
        scat.set_array(colors)
        return scat,

    frames = list(history_with_fictive_steps.keys())
    frame_number = 0
    update(frame_number)  # Render the first frame (frame index 0)
    plt.savefig(f'medias/deboarding/frames/frame_{frame_number}.png', bbox_inches="tight")
    ani = FuncAnimation(fig, update, frames=frames, interval=2, blit=True)

    if save_animation:
        nb_fps = 45
        ani.save(f'medias/deboarding/animations/animation_{INTERVAL}_interval_{nb_fps}fps.gif', writer='pillow',
                 fps=nb_fps, progress_callback=lambda i, n: print(f'Saving frame {i}/{len(frames)}'))
        print("ani saved ! ")
    else:
        plt.show()


if __name__ == "__main__":
    generate_animation(save_animation=True)
