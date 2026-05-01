from autocvd import autocvd
autocvd(num_gpus = 1)
import os
import optuna
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'axes.labelsize': 14,   # x and y labels
    'axes.titlesize': 16,   # plot title
    'xtick.labelsize': 12,  # x tick labels
    'ytick.labelsize': 12,  # y tick labels
    'legend.fontsize': 12,  # legend text
})

# Load the study from the SQLite database
study_name = 'study_full_adv2'  # Unique identifier of the study.
storage_name = 'sqlite:///'+str(study_name)+'.db'
study = optuna.load_study(study_name=study_name, storage=storage_name)

# Retrieve all completed trials
completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
print(completed_trials[20])

#best combine: trial 14
# params={'num_conv_blocks': 3, 'fc_dim': 64, 'base_channels': 16, 
#     'temporal_pool_output': 1, 'temporal_num_layers': 1, 'temporal_stride_first': 2,
#      'hidden_features': 30, 'num_transforms': 20}

# best TARP: trial 20
# params: {'num_conv_blocks': 3, 'fc_dim': 128, 'base_channels': 32,
#      'temporal_pool_output': 2, 'temporal_num_layers': 2, 'temporal_stride_first': 1,
#       'hidden_features': 7, 'num_transforms': 20}

# find trial with smallest values[0]
trials_with_values = [t for t in completed_trials if t.values]
if not trials_with_values:
    print("No completed trials with objective values.")
else:
    # best = min(trials_with_values, key=lambda t: t.values[1])
    sorted_trials = sorted(trials_with_values, key=lambda t: t.values[1])
    for i in range(1):
        best = sorted_trials[i]
        print("trial.number:", best.number)
        print("values:", best.values)
        print("params:", best.params)

# Extract objective values
objective_values = [trial.values for trial in completed_trials if trial.values is not None]

# Assuming a 2-objective study for simplicity in visualization
objective_1 = [values[0] for values in objective_values]
objective_2 = [values[1] for values in objective_values]


# # Plotting the objectives
plt.figure(figsize=(8, 6))
x = np.linspace(0,len(objective_1)-1, len(objective_1))
plt.scatter(objective_1, objective_2, c='blue', label='Other trials')

# Add trial numbers as annotations
# for i, (x_val, y_val) in enumerate(zip(objective_1, objective_2)):
#     plt.annotate(str(i), (x_val, y_val), fontsize=8, alpha=0.7,
#                  xytext=(3, 3), textcoords="offset points")

# Highlight trials 
for trial_num in [0, 20, 14, 98, 124]:
    if trial_num < len(objective_1):
        plt.scatter(objective_1[trial_num], objective_2[trial_num], c='red', s=40)
plt.scatter(objective_1[20], objective_2[20], c='red', s=40, label='Best trials')

plt.scatter(objective_1[20], objective_2[20], c='green', s=60, zorder=5)
plt.scatter(objective_1[14], objective_2[14], c='green', s=60, zorder=5, label='Chosen trials')
plt.annotate(str(14), (objective_1[14], objective_2[14]), fontsize=10, alpha=0.9, xytext=(-18, 0), textcoords="offset points")
plt.annotate(str(20), (objective_1[20], objective_2[20]), fontsize=10, alpha=0.9, xytext=(-18, 0), textcoords="offset points")
# plt.title('Objective Trade-offs')
plt.ylabel('mean TARP deviation')
plt.xlabel('NLL')
# plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("optuna_NLLvsTARP_adv.png")
plt.clf()


# # plot nsf (red) and maf (blue)
# nsf_x, nsf_y = [], []
# maf_x, maf_y = [], []
# mdn_x, mdn_y = [], []

# plt.figure(figsize=(8,6))
# k=0

# for i, t in enumerate(completed_trials):
#     if not t.values:
#         continue

#     label = f"{t.params['hidden_features']},{t.params['num_transforms']}"
#     m = t.params.get('model')

#     if m == 'nsf':
#         nsf_x.append(i); nsf_y.append(t.values[k])
#         # plt.scatter(i, t.values[k], c='red')
#         plt.annotate(label, (i, t.values[k]),
#                      fontsize=7, alpha=0.7,
#                      xytext=(3,3), textcoords="offset points")
#     elif m == 'maf':
#         maf_x.append(i); maf_y.append(t.values[k])
#         # plt.scatter(i, t.values[k], c='blue')
#         plt.annotate(label, (i, t.values[k]),
#                      fontsize=7, alpha=0.7,
#                      xytext=(3,3), textcoords="offset points")
#     elif m == 'mdn':
#         mdn_x.append(i); mdn_y.append(t.values[k])
#         # plt.scatter(i, t.values[k], c='green')
#         plt.annotate(label, (i, t.values[k]),
#                      fontsize=7, alpha=0.7,
#                      xytext=(3,3), textcoords="offset points")

# plt.xlabel('trial index')
# plt.scatter(mdn_x, mdn_y, c='green', label="mdn")
# plt.scatter(maf_x, maf_y, c='blue', label="maf")
# plt.scatter(nsf_x, nsf_y, c='red', label="nsf")
# plt.legend()
# plt.grid(True)
# if k==0:
#     plt.title('NLL value by model')
#     plt.ylabel('NLL')
#     plt.savefig('optuna_NLL_2log.png')
# elif k==1:
#     plt.title('TARP value by model')
#     plt.ylabel('TARP')
#     plt.savefig('optuna_TARP_2log.png')


# (Pareto front selection as an example)
pareto_front = optuna.visualization.matplotlib.plot_pareto_front(study, target_names=[r"NLL", r"mean TARP dev"])
plt.savefig("optuna_study_adv.png")

