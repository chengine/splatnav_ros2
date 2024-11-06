
#%%
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('ytick', labelsize=10) 

exp_name = 'stonehenge'

methods = ['gs_probabilistic', 'gs_simple', 'astar', 'rrt_simple', 'nerf_nav']
colors = ['g', 'b', 'chocolate', 'm', 'mediumpurple']

with open(f'{exp_name}_data.json', 'r') as f:
    meta = json.load(f)

metrics = ['SDF', 'Path Length']

fig, ax = plt.subplots(len(metrics), figsize=(4, 8), dpi=1000)

counter = 0
for c, method in zip(colors, methods):
    # Value is list of list containing all trajectories and all points in trajectories
    data = meta[method]

    sdfs = data['sdfs']
    min_sdfs = np.array(data['sdfs_min'])

    sdf_mean = np.mean(min_sdfs)
    sdf_errors = np.array([sdf_mean - min_sdfs.min(), min_sdfs.max() - sdf_mean]).reshape(-1, 1)

    # p0 = ax[0].bar(counter, sdf_mean, align='center', alpha=0.5, ecolor='black', capsize=10, yerr=sdf_errors)
    # p1 = ax[1].bar(counter, vol_mean, align='center', alpha=0.5, ecolor='black', capsize=10, yerr=vol_errors)

    ax[0].errorbar(counter, sdf_mean, yerr=sdf_errors, color=c, capsize=15)
    ax[0].scatter(counter*np.ones_like(min_sdfs), min_sdfs, color=c, alpha=0.05)
    ax[0].scatter(counter, sdf_mean, s=250, marker='_', color=c)
    counter += 1

counter = 0
for c, method in zip(colors, methods):
    # Value is list of list containing all trajectories and all points in trajectories
    data = meta[method]

    lengths = np.array(data['path_lengths'])

    lengths_mean = np.mean(lengths)
    lengths_error = np.array([lengths_mean - lengths.min(), lengths.max() - lengths_mean]).reshape(-1, 1)

    # p0 = ax[0].bar(counter, sdf_mean, align='center', alpha=0.5, ecolor='black', capsize=10, yerr=sdf_errors)
    # p1 = ax[1].bar(counter, vol_mean, align='center', alpha=0.5, ecolor='black', capsize=10, yerr=vol_errors)

    ax[1].errorbar(counter, lengths_mean, yerr=lengths_error, color=c, capsize=15)
    ax[1].scatter(counter*np.ones_like(lengths), lengths, color=c, alpha=0.05)
    ax[1].scatter(counter, lengths_mean, s=250, marker='_', color=c)

    counter += 1

#%%
ax[0].axhline(y = 0., color = 'r', linestyle = '--', alpha=0.3) 
ax[0].set(xticklabels=[])
ax[0].tick_params(bottom=False)
ax[1].set(xticklabels=[])
ax[1].tick_params(bottom=False)
# ax[0].set_ylabel('Distance to Mesh Surface')
# ax[1].set_ylabel('Path Length')
#%%
# plt.show()
#%%
plt.savefig(f'statistics_{exp_name}.png')
# %%
