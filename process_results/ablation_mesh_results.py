#%% 
import os
import numpy as np
import json
import open3d as o3d
import scipy 

name = 'stonehenge'
mesh_fp = f'{name}.ply'
# mesh_fp = f'{name}.obj'

mesh_ = o3d.io.read_triangle_mesh(mesh_fp)
kdtree = scipy.spatial.KDTree(mesh_.vertices)

# Create robot body
r = 0.03   # For Statues and Flightroom

methods = ['mpc', 'mpc_astar', 'mpc_basic']

full_data = {}

for method in methods:
    print('Calculating results for ', method)

    fp = f'{method}/traj.json'
    with open(fp, 'r') as f:
        meta = json.load(f)
    traj = meta['traj']

    if method == 'mpc_basic':
        stuck = meta['stuck']

    sdfs = []
    sdfs_min = []
    path_lengths = []
    for i, sub_traj in enumerate(traj):

        sub_traj = np.array(sub_traj)[..., :3]

        if np.isnan(sub_traj).sum() > 0:
            print('Warning: Nans')
            continue

        sdf = kdtree.query(sub_traj, workers=-1)[0] - r 

        sdf = sdf.astype(np.float64)
        sdf_min = np.min(sdf)

        # calculate path length
        length = np.sum(np.linalg.norm(sub_traj[:-1] - sub_traj[1:], axis=-1))

        sdfs.append(sdf.tolist())
        sdfs_min.append(sdf_min)

        if method == 'mpc_basic' and stuck[i]:
            continue
        path_lengths.append(length)
        print('Trajectory ', i) 

    data = {
        'sdfs': sdfs,
        'sdfs_min': sdfs_min,
        'path_lengths': path_lengths
    }

    full_data[method] = data

with open(f'ablation_data.json', 'w') as f:
    json.dump(full_data, f, indent=4)
