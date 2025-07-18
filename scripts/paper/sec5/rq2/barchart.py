import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np

# Corrected and cleaned data provided by the user
data = """
type model_type configuration cg_precision cg_recall cg_gt_precision cg_gt_recall
general RF DC1 0.88 0.84 0.66 0.77
doop RF DC1 0.84 0.82 0.78 0.87
general BERT DC1 0.90 0.81 0.68 0.83
general T5 DC1 0.88 0.82 0.67 0.88
doop BERT DC1 0.88 0.71 0.81 0.8
doop T5 DC1 0.85 0.76 0.77 0.93
config BERT DC1 0.89 0.78 0.67 0.80
config T5 DC1 0.86 0.75 0.67 0.79
config RF DC1 0.88 0.83 0.66 0.78
general RF DC2 0.92 0.68 0.8 0.73
doop RF DC2 0.89 0.69 0.78 0.76
general BERT DC2 0.94 0.67 0.83 0.74
general T5 DC2 0.93 0.67 0.78 0.75
doop BERT DC2 0.93 0.62 0.83 0.65
doop T5 DC2 0.91 0.63 0.78 0.75
config BERT DC2 0.92 0.67 0.79 0.73
config T5 DC2 0.93 0.65 0.80 0.75
config RF DC2 0.89 0.7 0.79 0.76
general RF DC3 0.92 0.84 0.66 0.77
doop RF DC3 0.87 0.82 0.78 0.87
general BERT DC3 0.94 0.81 0.68 0.83
general T5 DC3 0.93 0.82 0.67 0.88
doop BERT DC3 0.91 0.71 0.81 0.8
doop T5 DC3 0.88 0.76 0.77 0.93
config BERT DC3 0.91 0.83 0.67 0.90
config T5 DC3 0.91 0.84 0.66 0.89
config RF DC3 0.89 0.85 0.64 0.81
general RF OC1 0.89 0.84 0.54 0.77
general BERT OC1 0.91 0.84 0.59 0.86
general T5 OC1 0.89 0.85 0.55 0.91
config BERT OC1 0.89 0.86 0.55 0.88
config T5 OC1 0.87 0.87 0.53 0.88
config RF OC1 0.87 0.87 0.56 0.88
general RF WC1 0.87 0.8 0.77 0.86
wala RF WC1 0.86 0.8 0.75 0.83
general BERT WC1 0.9 0.77 0.81 0.85
general T5 WC1 0.89 0.78 0.77 0.86
wala BERT WC1 0.89 0.72 0.8 0.81
wala T5 WC1 0.87 0.78 0.79 0.86
config BERT WC1 0.89 0.72 0.8 0.81
config T5 WC1 0.87 0.73 0.78 0.83
config RF WC1 0.85 0.77 0.75 0.82
general RF WC2 0.92 0.8 0.77 0.86
wala RF WC2 0.91 0.8 0.75 0.82
general BERT WC2 0.94 0.77 0.81 0.85
general T5 WC2 0.94 0.78 0.77 0.86
wala BERT WC2 0.94 0.72 0.8 0.81
wala T5 WC2 0.92 0.78 0.79 0.86
config BERT WC2 0.94 0.76 0.79 0.82
config T5 WC2 0.92 0.74 0.78 0.85
config RF WC2 0.91 0.82 0.75 0.86
general RF WC3 0.88 0.8 0.77 0.86
wala RF WC3 0.87 0.8 0.75 0.83
general BERT WC3 0.91 0.77 0.81 0.85
general T5 WC3 0.9 0.78 0.77 0.86
wala BERT WC3 0.9 0.72 0.8 0.81
wala T5 WC3 0.88 0.78 0.79 0.86
config BERT WC3 0.88 0.79 0.77 0.9
config T5 WC3 0.88 0.76 0.78 0.84
config RF WC3 0.86 0.82 0.76 0.87
"""

df = pd.read_csv(io.StringIO(data), sep='\s+')

# --- Data Preparation ---
df['type'] = df['type'].replace(['doop', 'wala'], 'tool')
df['x_label'] = df['configuration'] + '/' + df['model_type']
df_grouped = df.melt(id_vars=['type', 'x_label'], 
                     value_vars=['cg_precision', 'cg_recall', 'cg_gt_precision', 'cg_gt_recall'],
                     var_name='cg_metric', value_name='score').groupby(['x_label', 'type'])['score'].mean().unstack()

type_order = ['general', 'tool', 'config']
df_grouped = df_grouped.reindex(columns=type_order)

# --- Plotting Setup ---
# MODIFICATION 2: Make the figure shorter
fig, ax = plt.subplots(figsize=(22, 6))
x_labels = df_grouped.index
x_pos = np.arange(len(x_labels))
n_types = len(type_order)

custom_colors = ['#004c6d', '#73a3c6', '#f28e2b']

# MODIFICATION 1: Make the total group of bars narrower to increase space between groups
group_width = 0.7
bar_width = group_width / n_types

# --- Manual Plotting Loop ---
for i, type_name in enumerate(type_order):
    positions = x_pos + (i - (n_types - 1) / 2) * bar_width
    values = df_grouped[type_name]

    if type_name == 'config':
        for k, label in enumerate(x_labels):
            if label.startswith('OC1') and pd.isna(df_grouped.loc[label, 'tool']):
                tool_position_index = 1
                positions[k] = x_pos[k] + (tool_position_index - (n_types - 1) / 2) * bar_width

    ax.bar(positions[~values.isna()], values.dropna(), width=bar_width, color=custom_colors[i], label=type_name)

# --- Chart Finalization ---
ax.set_xlabel('Configuration/Model', fontsize=17)
ax.set_ylabel('Average Score of CG Metrics', fontsize=17)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels,  rotation=45, ha='right')

ax.tick_params(axis='x', labelsize=17)

# MODIFICATION 2: Set specific Y-axis ticks starting from 0.5
y_ticks = np.arange(0.5, 0.9, 0.1)
ax.set_yticks(y_ticks)
ax.set_ylim(0.6)
# font size for y-axis ticks
ax.tick_params(axis='y', labelsize=15)

# Tidy up legend to avoid duplicates from the loop
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), fontsize=16, loc='upper left')

plt.tight_layout()
plt.savefig('bar_chart_modified.png')
plt.close()