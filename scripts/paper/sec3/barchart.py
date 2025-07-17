import matplotlib.pyplot as plt
import numpy as np

# Data definitions
configs = [
    "DC1", "DC2", "DC3",
    "WC1", "WC2", "WC3",
    "OC1"
]
true_counts = [
    [4180, 15277, 5212, 2225], [2223, 14618, 2225], [4162, 15259, 5213, 2225],
    [4072, 14821, 2225], [4053, 14813, 2225], [4071, 14816, 2225],
    [4348, 17112, 5205, 2215],
]
false_counts = [
    [1560, 5886, 1004, 12], [445, 3860, 0], [1242, 4028, 25, 0],
    [1219, 3771, 143], [880, 3041, 77], [1175, 5524, 121],
    [1455, 5644, 696, 0],
]
unknown_counts = [
    [2969, 28327, 22131, 559], [1458, 22872, 559], [2969, 28327, 22131, 559],
    [3097, 25298, 562], [2850, 23611, 562], [2850, 23611, 562],
    [69305, 307552, 103396, 3024],
]

# --- Aggregation Step ---
agg_true = np.array([sum(c) for c in true_counts])
agg_false = np.array([sum(c) for c in false_counts])
agg_unknown = np.array([sum(c) for c in unknown_counts])

print("Aggregated True Counts:", agg_true)
print("Aggregated False Counts:", agg_false)
print("Aggregated Unknown Counts:", agg_unknown)

# --- Plotting Parameters ---
bar_width = 0.1  # <-- 1. DECREASE the bar width
spacing = 0.3    # <-- 2. DEFINE a spacing smaller than the default of 1.0
colors = ["#28d122", "#df1828", "#A09993"]

# <-- 3. ADJUST positions by multiplying by the new spacing
x_positions = np.arange(len(configs)) * spacing

# Setup the figure with a broken y-axis
# <-- 4. REDUCE the overall figure size
fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 2]})
plt.subplots_adjust(hspace=0.08)

# Plot bars on both subplots
ax.bar(x_positions, agg_true, bar_width, color=colors[0], label='True')
ax.bar(x_positions, agg_false, bar_width, bottom=agg_true, color=colors[1], label='False')
ax.bar(x_positions, agg_unknown, bar_width, bottom=agg_true + agg_false, color=colors[2], label='Unknown')

ax2.bar(x_positions, agg_true, bar_width, color=colors[0])
ax2.bar(x_positions, agg_false, bar_width, bottom=agg_true, color=colors[1])
ax2.bar(x_positions, agg_unknown, bar_width, bottom=agg_true + agg_false, color=colors[2])

# Y-axis limits for the broken axis effect
ax.set_ylim(480000, 530000)
ax2.set_ylim(0, 95000)

# Hide spines and ticks for a cleaner look
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()

# Add diagonal break lines
d = .015
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# --- Labels, Ticks, and Layout ---
# fig.suptitle("Aggregated Call Graph Edge Labels per Configuration", fontsize=14, y=0.96)
ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.95))
fig.text(0.06, 0.5, 'Total Edge Count', va='center', rotation='vertical', fontsize=12)
ax2.set_xticks(x_positions)
ax2.set_xticklabels(configs, rotation=0, fontsize=10)
ax2.set_xlabel("Configuration", fontsize=12, labelpad=10)

plt.tight_layout(rect=[0.08, 0, 1, 0.93])
plt.savefig("barchart_aggregated_smaller.png", dpi=300)
plt.show()