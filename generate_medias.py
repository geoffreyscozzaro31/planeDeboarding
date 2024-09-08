import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define the size of the grid (20 rows, 7 columns)
rows, cols = 20, 7

# Create an empty grid
grid = np.zeros((rows, cols))

# Define probabilities
# Higher probabilities at the top and closer to the aisle
for r in range(rows):
    for c in range(cols):
        if c == 3:  # Aisle seats
            grid[r, c] = 0.0  # Probability 0% for aisles
        elif c == 1 or c == 5:  # Middle seats
            grid[r, c] = 0.3 * (1 - r / rows)  # Lower probability for middle seats
        else:  # Window seats
            grid[r, c] = 0.6 * (1 - r / rows)  # Higher probability towards the top

# Normalize the grid
grid = grid / np.sum(grid) * 100
grid = np.transpose(grid)


vmax = int(np.ceil(np.max(grid)))
# Create a colormap
cmap = plt.get_cmap('Reds')
norm = mcolors.Normalize(vmin=0, vmax=vmax)

# Plot the grid
fig, ax = plt.subplots(figsize=(16, 8))  # Increased width for wider cells
cax = ax.matshow(grid, cmap=cmap, norm=norm)

ft = 14

# Add color bar
cbar = plt.colorbar(cax, ax=ax, orientation='horizontal')
cbar.set_label('Selection Probability (%)',fontsize=ft)

xticks = [i for i in range(vmax+1)]
cbar.set_ticks(xticks)
cbar.set_ticklabels([f"{x}%" for x in xticks], fontsize=ft)
cbar.ax.invert_yaxis()  # Highest probability at the top

# Add gridlines and labels
ax.set_yticks(np.arange(-0.5, cols, 1), minor=True)
ax.set_xticks(np.arange(-0.5, rows, 1), minor=True)
ax.grid(which="minor", color="k", linestyle='-', linewidth=2)
ax.tick_params(which="minor", size=0)

ax.set_yticks(np.arange(cols))
ax.set_yticklabels(['Window-side', 'Middle', 'Aisle-side', '', 'Aisle-side', 'Middle', 'Window-side'], fontsize=ft)
ax.set_xticks(np.arange(rows))
ax.set_xticklabels(np.arange(1, rows + 1), fontsize=11)
ax.xaxis.set_ticks_position('bottom')

# Add probability values in each cell
for r in range(rows):
    for c in range(cols):
        if grid[c, r] > 0:
            ax.text(r, c, f'{grid[c, r]:.1f}%', ha='center', va='center', color='black', fontsize=11)

# Title and labels
plt.xlabel('Row number',fontsize=ft)


plt.savefig('medias/prereserved_seat_probability_distribution.png', bbox_inches="tight")
plt.show()
