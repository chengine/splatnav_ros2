#%%
import os
import torch
from pathlib import Path    
import time
import pickle 
import numpy as np
from tqdm import tqdm
import json
from SFC.corridor_utils import Corridor
from splat.splat_utils import GSplatLoader
from splatplan.splatplan import SplatPlan
from splatplan.spline_utils import SplinePlanner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Methods for the simulation
n = 100         # number of different configurations
n_steps = 10   # number of time discretizations

# Creates a circle for the configuration
t = np.linspace(0, 2*np.pi, n)
t_z = 10*np.linspace(0, 2*np.pi, n)

# flag for using saved planner / corridor data
use_saved = False
config_path_base = 'configs'
os.makedirs(config_path_base, exist_ok=True)

# Using sparse representation?
sparse = False

### ----------------- Possible Methods ----------------- ###
# method = 'splatplan'
# method = 'sfc'
# TODO: splatplan-single-step, A*
### ----------------- Possible Distance Types ----------------- ###

for scene_name in ['flight', 'statues', 'old_union']: #['stonehenge', 'statues', 'flight', 'old_union']:
    for method in ['splatplan']:

        # NOTE: POPULATE THE UPPER AND LOWER BOUNDS FOR OTHER SCENES!!!
        if scene_name == 'old_union':
            radius_z = 0.01     # How far to undulate up and down
            radius_config = 1.35/2  # radius of xy circle
            mean_config = np.array([0.14, 0.23, -0.15]) # mean of the circle

            if sparse:
                path_to_gsplat = Path('outputs/old_union2/sparse-splat/2024-10-25_113753/config.yml')
            else:
                path_to_gsplat = Path('outputs/old_union2/splatfacto/2024-09-02_151414/config.yml') # points to where the gsplat params are stored

            radius = 0.01       # radius of robot
            amax = 0.1
            vmax = 0.1

            lower_bound = torch.tensor([-.8, -.7, -0.2], device=device)
            upper_bound = torch.tensor([1., 1., -0.1], device=device)

            resolution = 75

        elif scene_name == 'stonehenge':
            radius_z = 0.01
            radius_config = 0.784/2
            mean_config = np.array([-0.08, -0.03, 0.05])

            if sparse:
                path_to_gsplat = Path('outputs/stonehenge/sparse-splat/2024-10-25_120323/config.yml')
            else:
                path_to_gsplat = Path('outputs/stonehenge/splatfacto/2024-09-11_100724/config.yml')

            radius = 0.015
            amax = 0.1
            vmax = 0.1

            lower_bound = torch.tensor([-5., -.5, -0.], device=device)
            upper_bound = torch.tensor([5., .5, 0.1], device=device)

            resolution = 40

        elif scene_name == 'statues':
            radius_z = 0.03    
            radius_config = 0.475
            mean_config = np.array([-0.064, -0.0064, -0.025])

            if sparse:
                path_to_gsplat = Path('outputs/statues/sparse-splat/2024-10-25_114702/config.yml')
            else:
                path_to_gsplat = Path('outputs/statues/splatfacto/2024-09-11_095852/config.yml')

            radius = 0.03
            amax = 0.1
            vmax = 0.1

            lower_bound = torch.tensor([-.5, -.5, -0.1], device=device)
            upper_bound = torch.tensor([.5, .5, 0.2], device=device)

            resolution = 60

        elif scene_name == 'flight':
            radius_z = 0.06
            radius_config = 0.545/2
            mean_config = np.array([0.19, 0.01, -0.02])

            if sparse:
                path_to_gsplat = Path('outputs/flight/sparse-splat/2024-10-25_115216/config.yml')
            else:
                path_to_gsplat = Path('outputs/flight/splatfacto/2024-09-12_172434/config.yml')

            radius = 0.02
            amax = 0.1
            vmax = 0.1

            lower_bound = torch.tensor([-1.33, -0.5, -0.17], device=device)
            upper_bound = torch.tensor([1, 0.5, 0.26], device=device)

            resolution = 100

        print(f"Running {scene_name} with {method}")

        # Robot configuration
        robot_config = {
            'radius': radius,
            'vmax': vmax,
            'amax': amax,
        }

        # Environment configuration (specifically voxel)
        voxel_config = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'resolution': resolution,
        }

        tnow = time.time()
        gsplat = GSplatLoader(path_to_gsplat, device)
        print('Time to load GSplat:', time.time() - tnow)

        spline_planner = SplinePlanner(spline_deg=6, device=device)
        
        if method == 'splatplan' or method == 'splatplan-single-step':
            
            # load voxel config to save time if exists
            if os.path.exists(f'{config_path_base}/{scene_name}_splatplan.pkl') and use_saved: 
                print("Loading Splat-Plan Planner from file")
                with open(f'{config_path_base}/{scene_name}_splatplan.pkl', 'rb') as f:
                    planner = pickle.load(f)
            else: 
                planner = SplatPlan(gsplat, robot_config, voxel_config, spline_planner, device)
                with open(f'{config_path_base}/{scene_name}_splatplan.pkl', 'wb') as f:
                    pickle.dump(planner, f)
       
        elif method == "sfc":
            # load corridor config to save time if exists
            if os.path.exists(f'{config_path_base}/{scene_name}_sfc.pkl') and use_saved:
                print("Loading SFC from file")
                with open(f'{config_path_base}/{scene_name}_sfc.pkl', 'rb') as f:
                    sfc = pickle.load(f)
            else: 
                sfc = Corridor(gsplat, robot_config, voxel_config, spline_planner, device)
                with open(f'{config_path_base}/{scene_name}_sfc.pkl', 'wb') as f:
                    pickle.dump(sfc, f)

        else:
            raise ValueError(f"Method {method} not recognized")
        
        ### Create configurations in a circle
        x0 = np.stack([radius_config*np.cos(t), radius_config*np.sin(t), radius_z * np.sin(t_z)], axis=-1)     # starting positions
        x0 = x0 + mean_config

        xf = np.stack([radius_config*np.cos(t + np.pi), radius_config*np.sin(t + np.pi), radius_z * np.sin(t_z + np.pi)], axis=-1)     # goal positions
        xf = xf + mean_config

        # Run simulation
        total_data = []

        for trial, (start, goal) in enumerate(zip(x0, xf)):

            # State is 6D. First 3 are position, last 3 are velocity. Set initial and final velocities to 0
            x = torch.tensor(start).to(device).to(torch.float32)
            goal = torch.tensor(goal).to(device).to(torch.float32)

            # We only do this for the single-step SplatPlan
            if method == 'splatplan':
                output = planner.generate_path(x, goal)

            elif method == 'sfc':
                output = sfc.generate_path(x, goal)

            else:
                raise ValueError(f"Method {method} not recognized")

            total_data.append(output)
            print(f"Trial {trial} completed")

        # Save trajectory
        data = {
            'scene': scene_name,
            'method': method,
            'radius': radius,
            'amax': amax,
            'vmax': vmax,
            'radius_z': radius_z,
            'radius_config': radius_config,
            'mean_config': mean_config.tolist(),
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'resolution': resolution,
            'n_steps': n_steps,
            'total_data': total_data,
        }

        # create directory if it doesn't exist
        os.makedirs('trajs', exist_ok=True)

        # write to the file
        if sparse:
            save_path = f'trajs/{scene_name}_sparse_{method}.json'
        else:
            save_path = f'trajs/{scene_name}_{method}.json'
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)
        
# %%
