#%%
import os
import torch
from pathlib import Path    
import time
import numpy as np
import json

from splat.splat_utils import GSplatLoader
from initialization.grid_utils import GSplatVoxel
from polytopes.collision_set import GSplatCollisionSet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters for robot
radius = 0.02
vmax = 0.1
amax = 0.1

scene_name = 'flight'
save_path = scene_name + '_voxelized.obj'
lower_bound = torch.tensor([-1.33, -0.5, -0.17], device=device)
upper_bound = torch.tensor([1, 0.5, 0.26], device=device)
resolution = 100

if scene_name == 'old_union':
    path_to_gsplat = Path('outputs/old_union2/splatfacto/2024-09-02_151414/config.yml') # points to where the gsplat params are stored

elif scene_name == 'stonehenge':
    path_to_gsplat = Path('outputs/stonehenge/splatfacto/2024-09-11_100724/config.yml')

elif scene_name == 'statues':
    path_to_gsplat = Path('outputs/statues/splatfacto/2024-09-11_095852/config.yml')

elif scene_name == 'flight':
    path_to_gsplat = Path('outputs/flight/splatfacto/2024-09-12_172434/config.yml')

elif scene_name == 'flight-low-res':
    path_to_gsplat = 'flightroom_gaussians_sparse_deep.json'
    path_to_gsplat_high_res = Path('outputs/flight/splatfacto/2024-09-12_172434/config.yml')

print(f"Running {scene_name}")

tnow = time.time()
gsplat = GSplatLoader(path_to_gsplat, device)
print('Time to load GSplat:', time.time() - tnow)

tnow = time.time()
torch.cuda.synchronize()
gsplat_voxel = GSplatVoxel(gsplat, lower_bound=lower_bound, upper_bound=upper_bound, resolution=resolution, radius=radius, device=device)
torch.cuda.synchronize()
print('Time to create GSplatVoxel:', time.time() - tnow)

#%% Save the mesh
# gsplat_voxel.create_mesh(save_path=save_path)
# %%
# gsplat.save_mesh(scene_name + '_gsplat.obj')
# %%

x0 = torch.tensor([0.39, 0.08, -0.03], device=device)
xf = torch.tensor([0., 0., 0.], device=device)

tnow = time.time()
path = gsplat_voxel.create_path(x0, xf)
print('Time to create path:', time.time() - tnow)

#%% Visualize bounding boxes
collision_set = GSplatCollisionSet(gsplat, vmax, amax, radius, device)
output = collision_set.compute_set(torch.tensor(path, device=device), save_path=scene_name)
#%%
total_data = []
data = {
'traj': path.tolist()
}

total_data.append(data)

# Save trajectory
data = {
    'scene': scene_name,
    'radius': radius,
    'total_data': total_data,
}

# create directory if it doesn't exist
os.makedirs('trajs', exist_ok=True)

# write to the file
with open(f'trajs/{scene_name}_astar_test.json', 'w') as f:
    json.dump(data, f, indent=4)

#%%