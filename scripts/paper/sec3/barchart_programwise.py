import matplotlib.pyplot as plt
import numpy as np

# Data definitions
configs = [
    "DC1", "DC2", "DC3",
    "WC1", "WC2", "WC3",
    "OC1"
]
programs_per_config = [
    ["Axion", "Batik", "Xerces", "Jasml"],
    ["Axion", "Batik", "Jasml"],
    ["Axion", "Batik", "Xerces", "Jasml"],
    ["Axion", "Batik", "Jasml"],
    ["Axion", "Batik", "Jasml"],
    ["Axion", "Batik", "Jasml"],
    ["Axion", "Batik", "Xerces", "Jasml"],
]
true_counts = [
    [4180, 15277, 5212, 2225],
    [2223, 14618, 2225],
    [4162, 15259, 5213, 2225],
    [4072, 14821, 2225],
    [4053, 14813, 2225],
    [4071, 14816, 2225],
    [4348, 17112, 5205, 2215],
]
false_counts = [
    [1560, 5886, 1004, 12],
    [445, 3860, 0],
    [1242, 4028, 25, 0],
    [1219, 3771, 143],
    [880, 3041, 77],
    [1175, 5524, 121],
    [1455, 5644, 696, 0],
]
unknown_counts = [
    [2969, 28327, 22131, 559],
    [1458, 22872, 559],
    [2969, 28327, 22131, 559],
    [3097, 25298, 562],
    [2850, 23611, 562],
    [2850, 23611, 562],
    [69305, 307552, 103396, 3024],
]

# Plotting parameters
bar_width = 0.1
config_colors = ["#28d122", "#df1828", "#A09993"]  # true, false, unknown

# Setup the figure with two y-axes (broken axis)
fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 2]})
plt.subplots_adjust(hspace=0.05)

bar_positions = []
xtick_labels = []
current_pos = 0

for i, config in enumerate(configs):
    n_programs = len(programs_per_config[i])
    positions = np.arange(n_programs) * (bar_width + 0.03) + current_pos
    bar_positions.extend(positions)
    xtick_labels.extend(programs_per_config[i])
    current_pos = positions[-1] + bar_width + 0.2  # gap between config groups

    # Stack bars
    true_vals = true_counts[i]
    false_vals = false_counts[i]
    unknown_vals = unknown_counts[i]

    ax.bar(positions, true_vals, bar_width, label='True' if i == 0 else "", color=config_colors[0])
    ax.bar(positions, false_vals, bar_width, bottom=true_vals, label='False' if i == 0 else "", color=config_colors[1])
    ax.bar(positions, unknown_vals, bar_width, bottom=np.array(true_vals)+np.array(false_vals),
           label='Unknown' if i == 0 else "", color=config_colors[2])

    ax2.bar(positions, true_vals, bar_width, color=config_colors[0])
    ax2.bar(positions, false_vals, bar_width, bottom=true_vals, color=config_colors[1])
    ax2.bar(positions, unknown_vals, bar_width, bottom=np.array(true_vals)+np.array(false_vals), color=config_colors[2])

# Y-axis limits for zoom and break
ax.set_ylim(100000, 330000)   # top plot (zoomed in to OC1)
ax2.set_ylim(0, 50000)        # bottom plot (all others)

# Add break lines
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-0.01, +0.01), (-0.02, +0.02), **kwargs)        # top-left diagonal
ax.plot((1 - 0.01, 1 + 0.01), (-0.02, +0.02), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)
ax2.plot((-0.01, +0.01), (1 - 0.02, 1 + 0.02), **kwargs)  # bottom-left diagonal
ax2.plot((1 - 0.01, 1 + 0.01), (1 - 0.02, 1 + 0.02), **kwargs)  # bottom-right diagonal

# X-axis labels and ticks
midpoints = []
last = 0
for programs in programs_per_config:
    n = len(programs)
    pos = np.arange(n) * (bar_width + 0.03) + last
    mid = pos.mean()
    midpoints.append(mid)
    last = pos[-1] + bar_width + 0.2
ax2.set_xticks(bar_positions)
ax2.set_xticklabels(xtick_labels, rotation=45, ha='right')

# Config labels under each group
for i, mid in enumerate(midpoints):
    ax2.text(mid, -9000, configs[i], ha='center', va='top', fontsize=10, fontweight='bold')

# Legend
ax.legend(loc='upper center')

# Axis labels
# fig.suptitle("Call Graph Edge Labels per Configuration and Program", fontsize=14)
ax.set_ylabel("Edge Count")
ax2.set_ylabel("Edge Count")
# ax2.set_xlabel("Programs grouped by Configuration")

plt.tight_layout()
plt.savefig("barchart.png", dpi=300)
