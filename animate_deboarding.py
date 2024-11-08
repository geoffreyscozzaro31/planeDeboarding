from enum import IntEnum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d


FOLDER_DATA = "medias/deboarding/"

# DISEMBARKING_RULE_NAME = "aisle_priority_deboarding_rule"
DISEMBARKING_RULE_NAME = "courtesy_deboarding_rule"
NB_ROWS = 31
TIME_STEP_DURATION = 0.5
FILE_PATH = FOLDER_DATA + f"{DISEMBARKING_RULE_NAME}_{NB_ROWS}rows_history_test.txt"

NB_STEPS = 6


class State(IntEnum):
    seated = 1
    standup_from_seat = 2
    move_wait = 3
    move_from_row = 4


def read_history(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    n_rows, dummy_rows, n_seats_left, n_seats_right, n_passengers, n_luggage_events = map(int, lines[0].split())
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

    return history, n_rows, dummy_rows, n_seats_left, n_seats_right, n_passengers, n_luggage_events


state_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
cmap = ListedColormap(state_colors)
norm = plt.Normalize(vmin=0, vmax=len(State) - 0.5)


def interpolate_positions(start, end, steps):
    t = np.linspace(0, 1, num=steps)
    interp_func = interp1d([0, 1], [start, end], kind='linear')
    return interp_func(t)


def generate_fictive_steps(history, steps):
    new_history = {}
    last_positions = {}

    for t, state in history.items():
        for entry in state:
            _, pax_id, x_pos, y_pos, pax_state = entry

            if pax_id not in last_positions:
                last_positions[pax_id] = (x_pos, y_pos, t * steps)

            prev_x_pos, prev_y_pos, last_time = last_positions[pax_id]

            if (x_pos, y_pos) == (prev_x_pos, prev_y_pos):
                for step in range(steps):
                    new_t = t * steps + step
                    if new_t not in new_history:
                        new_history[new_t] = []
                    new_history[new_t].append([new_t, pax_id, x_pos, y_pos, pax_state])
            else:
                x_interp = interpolate_positions(prev_x_pos, x_pos, steps + (t * steps - last_time) // steps)
                y_interp = interpolate_positions(prev_y_pos, y_pos, steps + (t * steps - last_time) // steps)

                for step in range(steps):
                    new_t = t * steps + step
                    if new_t not in new_history:
                        new_history[new_t] = []
                    new_history[new_t].append([new_t, pax_id, x_interp[step], y_interp[step], pax_state])

                last_positions[pax_id] = (x_pos, y_pos, new_t)

    return new_history


def generate_animation(save_animation=False):
    history, n_rows, dummy_rows, n_seats_left, n_seats_right, n_passengers, n_luggage_events = read_history(FILE_PATH)

    history_with_fictive_steps = generate_fictive_steps(history, NB_STEPS)

    # fig, ax = plt.subplots(figsize=(6, 12))
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.subplots_adjust(left=-1.2, right=0.95, top=0.99, bottom=0.05)

    ax.set_xlim(-n_seats_left - 1, n_seats_right + 1)
    ax.set_ylim(-2, n_rows + dummy_rows)
    ax.set_xticks(np.arange(-n_seats_left, n_seats_right + 1))
    ax.set_yticks(np.arange(0, n_rows + dummy_rows, 2), np.arange(0, (n_rows + dummy_rows) // 2))
    plt.gca().invert_yaxis()
    fontsize = 18
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel('Seat Columns', fontsize=fontsize)
    ax.set_ylabel('Rows', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    ax.set_aspect('equal', adjustable='box')

    # scat = ax.scatter([], [], c=[], cmap=cmap, norm=plt.Normalize(vmin=0.5, vmax=len(State) + 0.5), s=100)
    scat = ax.scatter([], [], c=[], cmap=cmap, norm=plt.Normalize(vmin=0.5, vmax=len(State) + 0.5), s=100)
    cbar = plt.colorbar(scat, ax=ax, ticks=np.arange(1, len(State) + 1))
    cbar.ax.set_yticklabels([state.name for state in State], fontsize=11)

    # Time text
    time_template = 'Time: {:d}s'
    time_text = ax.text(0.25, 0.97, '', transform=ax.transAxes, fontsize=fontsize - 2)

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

    saveframe_idx = []

    def update(frame):
        state = history_with_fictive_steps[frame]
        positions, colors = plot_state(state)
        scat.set_offsets(positions)
        scat.set_array(colors)
        time_text.set_text(time_template.format(int(frame * TIME_STEP_DURATION / NB_STEPS)))
        if frame in saveframe_idx:
            fig.savefig(FOLDER_DATA + f'frames/sframe{frame}_{DISEMBARKING_RULE_NAME}.png', bbox_inches='tight')
            print(f"frame {frame} saved ! ")
        return scat, time_text

    ellipse_start_frame = 1000
    ellipse_end_frame = 2000
    frames = np.array(list(history_with_fictive_steps.keys())) #[frame_min:frame_max]
    mask = np.ones(len(frames), dtype=bool)
    mask[ellipse_start_frame:ellipse_end_frame] = False
    frames_subset = frames[mask]
    ani = FuncAnimation(fig, update, frames=frames_subset, interval=0.1, blit=True)
    if save_animation:
        nb_fps = 45
        ani.save(
            FOLDER_DATA + f'animations/animation_deboarding_{DISEMBARKING_RULE_NAME}_{NB_STEPS}steps_frame_{ellipse_start_frame}_{ellipse_end_frame}_ellipse.gif',
            writer='pillow',
            fps=nb_fps, progress_callback=lambda i, n: print(f'Saving frame {i}/{len(frames)}'))
        print("Animation saved!")
    else:
        plt.show()


if __name__ == "__main__":
    generate_animation(save_animation=True)
