import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Data ---
ratios = ['0.1', '0.25', '0.5', '0.75', '1.0']
models = ['RF', 'BERT_NN', 'T5_NN']
techniques = ['Down-sampling', 'Over-sampling', 'None']

# data = {
#     'BERT_NN': {
#         'None': [(0.70, 0.66)],
#         'Over-sampling': [
#             (0.67, 0.67),
#             (0.71, 0.66),
#             (0.68, 0.64),
#             (0.68, 0.65),
#             (0.70, 0.67)
#         ],
#         'Down-sampling': [
#             (0.66, 0.64),
#             (0.66, 0.62),
#             (0.65, 0.57),
#             (0.63, 0.56),
#             (0.64, 0.59)
#         ]
#     },
#     'T5_NN': {
#         'None': [(0.64, 0.61)],
#         'Over-sampling': [
#             (0.63, 0.56),
#             (0.69, 0.63),
#             (0.67, 0.59),
#             (0.64, 0.59),
#             (0.70, 0.62)
#         ],
#         'Down-sampling': [
#             (0.62, 0.60),
#             (0.69, 0.64),
#             (0.58, 0.56),
#             (0.50, 0.56),
#             (0.61, 0.56)
#         ]
#     },
#     'RF': {
#         'None': [(0.88, 0.75)],
#         'Over-sampling': [
#             (0.88, 0.75),
#             (0.88, 0.75),
#             (0.88, 0.74),
#             (0.88, 0.75),
#             (0.88, 0.74)
#         ],
#         'Down-sampling': [
#             (0.88, 0.75),
#             (0.88, 0.75),
#             (0.88, 0.74),
#             (0.88, 0.74),
#             (0.87, 0.74)
#         ]
#     }
# }

data = {
    'BERT_NN': {
        'None': [(0.70, 0.69)],
        'Over-sampling': [
            (0.67, 0.71),
            (0.71, 0.68),
            (0.68, 0.70),
            (0.68, 0.68),
            (0.70, 0.70)
        ],
        'Down-sampling': [
            (0.66, 0.69),
            (0.66, 0.64),
            (0.65, 0.60),
            (0.63, 0.54),
            (0.64, 0.59)
        ]
    },
    'T5_NN': {
        'None': [(0.64, 0.65)],
        'Over-sampling': [
            (0.63, 0.61),
            (0.69, 0.67),
            (0.67, 0.64),
            (0.64, 0.64),
            (0.70, 0.67)
        ],
        'Down-sampling': [
            (0.62, 0.64),
            (0.69, 0.68),
            (0.58, 0.60),
            (0.50, 0.59),
            (0.61, 0.59)
        ]
    },
    'RF': {
        'None': [(0.88, 0.83)],
        'Over-sampling': [
            (0.88, 0.83),
            (0.88, 0.83),
            (0.88, 0.83),
            (0.88, 0.83),
            (0.88, 0.83)
        ],
        'Down-sampling': [
            (0.88, 0.83),
            (0.88, 0.83),
            (0.88, 0.83),
            (0.88, 0.83),
            (0.87, 0.83)
        ]
    }
}


## --- Plotting Setup ---
colors = {'RF': '#1f77b4', 'BERT_NN': '#ff7f0e', 'T5_NN': '#2ca02c'}


fig, axs = plt.subplots(1, 3, figsize=(20, 8), sharey=True, gridspec_kw={'width_ratios': [5, 5, 1]})

sampling_types = ['Down-sampling', 'Over-sampling']
x_ratios = np.arange(len(ratios))
n_models = len(models)
total_width = 0.8
model_width = total_width / n_models
bar_width = model_width * 0.45

# --- Subplots for Down-sampling and Over-sampling ---
for i, technique in enumerate(sampling_types):
    ax = axs[i]
    for j, model in enumerate(models):
        offset = (j - n_models / 2) * model_width + model_width / 2
        values = data[model][technique]
        val_scores = [v[0] for v in values]
        holdout_scores = [v[1] for v in values]

        ax.bar(x_ratios + offset - bar_width / 2, val_scores, bar_width, color=colors[model], alpha=1.0)
        ax.bar(x_ratios + offset + bar_width / 2, holdout_scores, bar_width, color=colors[model], alpha=0.6)

    ax.set_title(f"{technique}", fontsize=20)
    ax.set_xlabel("Labeling Ratio", fontsize=20)
    ax.set_xticks(x_ratios)
    ax.set_xticklabels(ratios)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

# --- MODIFIED Subplot for "None" ---
ax = axs[2]
none_total_width = 0.4 # Total width for all bars in the 'None' plot
none_model_width = none_total_width / n_models # Space allocated for each model

# Width of a single bar (e.g., validation). Must be half of the model's total bar width.
# Making it slightly less than half of none_model_width creates the gap between models.
single_bar_width = none_model_width * 0.3 

x_none = np.array([0]) 

for j, model in enumerate(models):
    # Calculate the center for the group of bars for this model
    offset = (j - n_models / 2) * none_model_width + none_model_width / 2
    val, holdout = data[model]['None'][0]
    
    # Place bars right next to each other.
    # Bar 1 (Validation): center is moved left by half a bar's width.
    # Bar 2 (Holdout): center is moved right by half a bar's width.
    ax.bar(x_none + offset - single_bar_width / 2, [val], single_bar_width, color=colors[model], alpha=1.0)
    ax.bar(x_none + offset + single_bar_width / 2, [holdout], single_bar_width, color=colors[model], alpha=0.6)

ax.set_title("No Balancing", fontsize=20)
# ax.set_xlabel("Labeling", fontsize=13)
ax.set_xticks(x_none)
ax.set_xticklabels(["None"])
ax.grid(axis='y', linestyle='--', alpha=0.6)

# --- Common Y label ---
axs[0].set_ylabel("Average F1 Score", fontsize=20)

y_ticks = np.arange(0, 0.9, 0.1) 
axs[0].set_yticks(y_ticks)

for ax in axs:
    # 'axis='both'' applies to x and y
    # 'labelsize' sets the font size for the tick labels (the numbers)
    ax.tick_params(axis='both', which='major', labelsize=20)

# # --- Legend ---
# legend1 = plt.legend(handles=model_patches, title='Models', loc='upper left', bbox_to_anchor=(1.05, 1))
# plt.gca().add_artist(legend1)
# plt.legend(handles=[val_patch, holdout_patch], title='Score Type', loc='upper left', bbox_to_anchor=(1.05, 0.75))


# model_patches = [mpatches.Patch(color=color, label=model.replace('_', ' ')) for model, color in colors.items()]
# val_patch = mpatches.Patch(facecolor='grey', alpha=1.0, label='Validation F1')
# holdout_patch = mpatches.Patch(facecolor='grey', alpha=0.6, label='Holdout F1')
# all_handles = model_patches + [val_patch, holdout_patch]
# fig.legend(handles=all_handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(all_handles), fontsize='large')

all_handles = []
for model, color in colors.items():
    # Create the solid patch for the Validation score
    val_patch = mpatches.Patch(color=color, alpha=1.0, label=f"{model.replace('_', ' ')} Validation")
    # Create the semi-transparent patch for the Holdout score
    holdout_patch = mpatches.Patch(color=color, alpha=0.6, label=f"{model.replace('_', ' ')} Holdout")
    # Add both patches to the list
    all_handles.extend([val_patch, holdout_patch])

# Create the figure legend with the new handles
fig.legend(handles=all_handles, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(models), fontsize=17)


# Adjust layout to prevent the legend from overlapping with titles
# The 'top' value is lowered to make space.
# plt.tight_layout(rect=[0, 0, 1, 0.92]) 

# plt.suptitle("F1 Scores with Different Sampling Techniques", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.90]) 
# plt.tight_layout(rect=[0, 0, 1, 0.92]) 
plt.savefig("balancing.png", dpi=300)
plt.show()