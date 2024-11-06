#%%
import os
import torch
torch.cuda.empty_cache()
from pathlib import Path    
import time
import numpy as np
import json
import open3d as o3d    

from SFC.corridor_utils import Corridor
from splat.splat_utils import GSplatLoader, PointCloudLoader
from initialization.grid_utils import GSplatVoxel
from polytopes.collision_set import GSplatCollisionSet, compute_bounding_box
from polytopes.polytopes_utils import h_rep_minimal, find_interior
from ellipsoids.mesh_utils import create_gs_mesh


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters for robot
radius = 0.02
vmax = 0.1
amax = 0.1
rs = vmax**2 / (2 * amax) # safety radius for bounding box

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

# load point cloud
gsplat = GSplatLoader(path_to_gsplat, device)

# set covariance to spherical near zero
point_scale = 1

#gsplat.covs = (point_scale * torch.eye(3)).unsqueeze(0).expand(gsplat.covs.shape[0], -1, -1).to(gsplat.covs.device)

point_cloud = gsplat.means

#%%

# create voxel grid
gsplat_voxel = GSplatVoxel(gsplat, lower_bound=lower_bound, upper_bound=upper_bound, resolution=resolution, radius=radius, device=device)

# run A*
x0 = torch.tensor([0.39, 0.08, -0.03], device=device)
xf = torch.tensor([0., 0., 0.], device=device)
path = gsplat_voxel.create_path(x0, xf)


'''
# CROPPED PATH FROM A* SOLUTION FOR DEBUGGING FLIGHT ROOM
path = torch.tensor([[ 3.82550001e-01,  8.50000158e-02, -3.02499980e-02],
       [ 3.82550001e-01,  8.50000158e-02, -2.59499978e-02],
       [ 3.82550001e-01,  8.50000158e-02, -2.16499977e-02],
       [ 3.82550001e-01,  8.50000158e-02, -1.73499975e-02]], device=device)
'''
#%%

cor = Corridor(radius_robot=radius, vmax=vmax, amax=amax)
gsplat_collision = GSplatCollisionSet(gsplat, vmax, amax, radius, device)

#for line_end in path[1:]: 

line_start = path[9]
line_end = path[10] 

line_segment = torch.tensor([line_start, line_end], device=device)
line_start = line_end

# generate collision set
collision_set = gsplat_collision.compute_set(line_segment)
box_As = collision_set[0]['A_bb']
box_bs = collision_set[0]['b_bb']

collision_means = collision_set[0]['means']
# find ellipsoid
ellipsoid, d, p_stars = cor.find_ellipsoid(line_segment, collision_means)

# find polyhedron
As, bs, ps_star = cor.find_polyhedron(collision_means, d, ellipsoid)


# transform box constraints to local frame of ellipsoid
box_As_tf = box_As * (1. / ellipsoid['S'])[None, :] @ ellipsoid['R'].T    # M x 3
box_bs_tf = box_bs + torch.sum(d[None, :] * box_As_tf, dim=-1)    # M

'''
# transform polytope constraints back to world frame
As_tf = As * (1. / ellipsoid['S'])[None, :] @ ellipsoid['R'].T    # M x 3
bs_tf = bs + torch.sum(d[None, :] * As_tf, dim=-1)    # M
ps_star_tf = ps_star * ellipsoid['S'][None, :] @ ellipsoid['R'] + d

# viz collision set
As_tf = torch.cat((As_tf, box_As))
bs_tf = torch.cat((bs_tf, box_bs))
As_tf = As_tf.cpu()
bs_tf = bs_tf.cpu()
cc = find_interior(As_tf, bs_tf)
minimal_A, minimal_b, qhull_pts = h_rep_minimal(As_tf, bs_tf, cc)

polytope_pcd = o3d.geometry.PointCloud()
polytope_pcd.points = o3d.utility.Vector3dVector(qhull_pts)
polytope_hull, _ = polytope_pcd.compute_convex_hull() 
polytope_hull.compute_vertex_normals()
'''
# shrink
As = torch.cat((As, box_As), dim=0)
bs_shrunk = cor.shrink_corridor(bs, box_bs)


#%%
# viz collision set
As = As.cpu()
bs = bs_shrunk.cpu()
cc = find_interior(As, bs)
minimal_A, minimal_b, qhull_pts = h_rep_minimal(As, bs, cc)

polytope_pcd_shrunk = o3d.geometry.PointCloud()
polytope_pcd_shrunk.points = o3d.utility.Vector3dVector(qhull_pts)
polytope_hull_shrunk, _ = polytope_pcd_shrunk.compute_convex_hull() 
polytope_hull_shrunk.paint_uniform_color([1.0, 0.0, 0.0])   
polytope_hull_shrunk.compute_vertex_normals()
#success = o3d.io.write_triangle_mesh('polytope_shrunk.obj', polytop`  e_hull)


#%%

# viz ellipsoid
means = np.array([d.cpu().numpy()])
rotations = np.array([ellipsoid['R'].cpu().numpy()])
scalings = np.array([ellipsoid['S'].cpu().numpy()])
colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]]))
ellipsoid_mesh = create_gs_mesh(means, rotations, scalings, colors)
#success = o3d.io.write_triangle_mesh('ellipsoid.obj', ellipsoid_mesh)

#%% 
point_cloud_pcd = o3d.geometry.PointCloud()
point_cloud_pcd.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())



vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(ellipsoid_mesh) # ellipsoid 
#vis.add_geometry(polytope_hull_shrunk) # polytope
vis.add_geometry(polytope_hull_shrunk) # polytope
vis.add_geometry(point_cloud_pcd) # point cloud
vis.get_render_option().mesh_show_back_face = True
vis.get_render_option().background_color = np.array([1.0, 1.0, 1.0])  # Black background for better contrast
vis.get_view_control().set_zoom(0.8)
vis.update_renderer()
vis.run()


# %%
