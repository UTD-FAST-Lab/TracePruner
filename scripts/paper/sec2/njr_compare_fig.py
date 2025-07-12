import numpy as np
import matplotlib.pyplot as plt

# Data breakdown
curated_breakdown = {
    'True Edges\nManual: 313\nDynamic: 35,709': 36022,
    'False Edges\nManual: 35\nDynamic: 779': 814,
    'Unknown Edges': 25349,
}

original_breakdown = {
    'True Edges\nDirect: 64,382': 65676,
    'False Edges\nDirect: 131,939': 1198543,
    # 'True Edges\nTotal: 65,676\nDirect: 64,382': 65676,
    # 'False Edges\nTotal: 1,198,543\nDirect: 131,939': 1198543,
}

# Total direct edges (for fun label)
original_direct_total = 196321

# Colors
curated_colors = ['#4CAF50', '#F44336', '#9E9E9E']  # green, red, gray
original_colors = ['#4CAF50', '#F44336']            # green, red

# Create figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# --- Curated DB Chart ---
wedges1, texts1, autotexts1 = axes[0].pie(
    curated_breakdown.values(),
    labels=None,
    colors=curated_colors,
    startangle=140,
    wedgeprops=dict(edgecolor='w'),
    autopct=lambda pct: f"{int(round(pct * sum(curated_breakdown.values()) / 100)):,}\n({pct:.1f}%)"
)

axes[0].set_title('Curated DB\nTotal: 62,185', fontsize=14)

for i, (wedge, label) in enumerate(zip(wedges1, curated_breakdown.keys())):
    angle = (wedge.theta2 + wedge.theta1) / 2.
    x = wedge.r * 1.2 * np.cos(np.deg2rad(angle))
    y = wedge.r * 1.2 * np.sin(np.deg2rad(angle))
    axes[0].annotate(
        # f"{label}\nCount: {list(curated_breakdown.values())[i]:,}",
        f"{label}",
        xy=(np.cos(np.deg2rad(angle)) * wedge.r * 0.7,
            np.sin(np.deg2rad(angle)) * 0.7),
        xytext=(x, y),
        arrowprops=dict(arrowstyle="->", lw=1.5),
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray')
    )

# --- Original DB Chart ---
wedges2, texts2, autotexts2 = axes[1].pie(
    original_breakdown.values(),
    labels=None,
    colors=original_colors,
    startangle=140,
    wedgeprops=dict(edgecolor='w'),
    autopct=lambda pct: f"{int(round(pct * sum(original_breakdown.values()) / 100)):,}\n({pct:.1f}%)"
)

axes[1].set_title(f"Total: 1,264,219", fontsize=14)

for i, (wedge, label) in enumerate(zip(wedges2, original_breakdown.keys())):
    angle = (wedge.theta2 + wedge.theta1) / 2.
    x = wedge.r * 1.2 * np.cos(np.deg2rad(angle))
    y = wedge.r * 1.2 * np.sin(np.deg2rad(angle))
    axes[1].annotate(
        # f"{label}\nCount: {list(original_breakdown.values())[i]:,}",
        f"{label}",
        xy=(np.cos(np.deg2rad(angle)) * wedge.r * 0.7,
            np.sin(np.deg2rad(angle)) * 0.7),
        xytext=(x, y),
        arrowprops=dict(arrowstyle="->", lw=1.5),
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray')
    )

# Final layout
plt.tight_layout(pad=3.0)
plt.savefig("curated_vs_original_edge_breakdown.png")
