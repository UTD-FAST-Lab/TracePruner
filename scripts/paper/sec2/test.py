import numpy as np
import matplotlib.pyplot as plt

curated_breakdown = {
    'True Edges\nManual: 313\nDynamic: 48,374': 48687,
    'False Edges\nManual: 35\nP.O: 779': 814,
    'Unknown Edges': 25349,
}

original_breakdown = {
    'True Edges\nDirect: 64,382': 71921,
    'False Edges\nDirect: 131,939': 1198543,
}

# Total direct edges (for fun label)
original_direct_total = 196321

# Colors
curated_colors = ['#4CAF50', '#F44336', '#9E9E9E']  # green, red, gray
original_colors = ['#4CAF50', '#F44336']            # green, red



# --- Data ---

data_breakdown = original_breakdown
colors = original_colors

# --- Create figure and axes ---
fig, ax = plt.subplots(figsize=(8, 12))

# --- Create the Pie Chart ---
total_sum = sum(data_breakdown.values())
wedges, texts, autotexts = ax.pie(
    data_breakdown.values(),
    labels=None,
    colors=colors,
    startangle=140,
    # 👇 **CHANGE 1**: Move the number labels further from the center (default is 0.6)
    pctdistance=0.6,
    wedgeprops=dict(edgecolor='w', linewidth=2),
    autopct=lambda pct: f"{int(round(pct * total_sum / 100)):,}\n({pct:.1f}%)"
)

# --- Style the inner number/percentage text ---
plt.setp(autotexts, size=16, weight="bold", color="black")

# --- Add Custom Labels (inside or outside based on size) ---
values = list(data_breakdown.values())
labels = list(data_breakdown.keys())
placement_threshold = 25

for i, wedge in enumerate(wedges):
    angle = (wedge.theta2 + wedge.theta1) / 2.
    percentage = (values[i] / total_sum) * 100

    if percentage > placement_threshold:
        # 👇 **CHANGE 2**: Place the label box closer to the center to avoid the numbers
        x = wedge.r * -0.1 * np.cos(np.deg2rad(angle))
        y = wedge.r * 0.8 * np.sin(np.deg2rad(angle))
        
        ax.text(
            x, y,
            labels[i],
            ha='center',
            va='center',
            fontsize=17,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='none', alpha=0.8)
        )
    else:
        # This part for outer labels remains the same
        xy = (np.cos(np.deg2rad(angle)) * wedge.r, np.sin(np.deg2rad(angle)) * wedge.r)
        xytext = (np.cos(np.deg2rad(angle)) * wedge.r * 1.35, np.sin(np.deg2rad(angle)) * wedge.r * 1.35)

        ax.annotate(
            labels[i],
            xy=xy,
            xytext=xytext,
            arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="arc3,rad=0.1"),
            ha='center',
            va='center',
            fontsize=17,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', lw=1)
        )

# ax.set_title(f"Total Edges: {total_sum:,}", fontsize=18)
fig.tight_layout()
fig.savefig("original_improved.png", dpi=300)
# To display the plot directly in a script or notebook, use:
# plt.show()


# --- Select which data to plot ---
data_breakdown = curated_breakdown
colors = curated_colors

# --- Create figure and axes ---
fig, ax = plt.subplots(figsize=(8, 8))

# --- Create the Pie Chart ---
total_sum = sum(data_breakdown.values())
wedges, texts, autotexts = ax.pie(
    data_breakdown.values(),
    labels=None,
    colors=colors,
    startangle=140,
    # Move the number labels further from the center to avoid overlap
    pctdistance=0.8,
    wedgeprops=dict(edgecolor='w', linewidth=2),
    # Format the text inside the wedges (count and percentage)
    autopct=lambda pct: f"{int(round(pct * total_sum / 100)):,}\n({pct:.1f}%)"
)
# --- Style the inner number/percentage text ---
plt.setp(autotexts, size=16, weight="bold", color="black")

# --- Add Custom Labels (inside or outside based on size) ---
values = list(data_breakdown.values())
labels = list(data_breakdown.keys())

# A slice must be larger than this percentage to have its label placed inside.
# You can adjust this value to fit your data's distribution.
placement_threshold = 25

for i, wedge in enumerate(wedges):
    angle = (wedge.theta2 + wedge.theta1) / 2.
    percentage = (values[i] / total_sum) * 100

    # If the slice is large enough, place the label inside
    if percentage > placement_threshold:
        # Place the label box closer to the center to avoid the numbers
        x = wedge.r * 0.45 * np.cos(np.deg2rad(angle))
        y = wedge.r * 0.45 * np.sin(np.deg2rad(angle))
        
        ax.text(
            x, y,
            labels[i],
            ha='center',
            va='center',
            fontsize=17,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='none', alpha=0.8)
        )
    # Otherwise, place the label outside with an arrow
    else:
        # Define connection point and text position
        xy = (np.cos(np.deg2rad(angle)) * wedge.r, np.sin(np.deg2rad(angle)) * wedge.r)
        xytext = (np.cos(np.deg2rad(angle)) * wedge.r * 1.35, np.sin(np.deg2rad(angle)) * wedge.r * 1.35)

        # Add the label using ax.annotate
        ax.annotate(
            labels[i],
            xy=xy,
            xytext=xytext,
            arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="arc3,rad=0.1"),
            ha='center',
            va='center',
            fontsize=17,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', lw=1)
        )

# ax.set_title(f"Total Edges: {total_sum:,}", fontsize=18)
# fig.tight_layout()
# fig.savefig("updated_njr_improved.png", dpi=300, bbox_inches='tight')
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
fig.savefig("updated_njr_improved.png", dpi=300)