#%% 
import os
import numpy as np
import json
import open3d as o3d
import trimesh
import scipy 

name = 'flightroom'
# mesh_fp = f'{name}.ply'
mesh_fp = f'{name}.obj'

mesh_ = o3d.io.read_triangle_mesh(mesh_fp)
kdtree = scipy.spatial.KDTree(mesh_.vertices)
# # mesh_ = mesh_.filter_smooth_taubin(number_of_iterations=100)
# mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_)

# # Create a scene and add the triangle mesh
# scene = o3d.t.geometry.RaycastingScene()
# _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

# # signed distance is a [32,32,32] array
# sdf_query = lambda x: scene.compute_signed_distance(x).numpy()

# Create robot body
r = 0.03   # For Statues and Flightroom

# mesh = trimesh.load(mesh_fp, force='mesh')
# print('Mesh is watertight? ' , mesh.is_watertight)

# proximity = trimesh.proximity.ProximityQuery(mesh)

methods = ['rrt_simple', 'astar', 'gs_probabilistic', 'gs_simple', 'nerf_nav']

full_data = {}

for method in methods:
    print('Calculating results for ', method)
    if method == 'nerf_nav':
        traj = []
        for i in range(100):
            try:
                fp = f'{method}/{name}/100.0_iter{i}/init_costs/19.json'
                with open(fp, 'r') as f:
                    meta = json.load(f)
                sub_traj = meta["pos"]
                traj.append(sub_traj)
            except:
                print(f'Traj number {i} does not exist.')

    elif method == 'rrt_simple'  or method == 'rrt_probabilistic':
        fp = f'{method}/{name}.json'
        with open(fp, 'r') as f:
            meta = json.load(f)

        traj = [element["Path"] for element in meta]

    else:
        # for astar, gs methods
        fp = f'{method}/{name}.json'
        with open(fp, 'r') as f:
            meta = json.load(f)
        traj = meta['traj']

    sdfs = []
    sdfs_min = []
    path_lengths = []
    for i, sub_traj in enumerate(traj):

        if len(sub_traj) > 0:
            sub_traj = np.array(sub_traj)[..., :3]

            if np.isnan(sub_traj).sum() > 0:
                continue

            if method == 'rrt_simple' or method == 'rrt_probabilistic':
                # Need to interpolate more points
                sub_traj1 = sub_traj[:-1]
                sub_traj2 = sub_traj[1:]

                t = np.linspace(0, 1, 20)

                sub_traj_ = []
                for (pt1, pt2) in zip(sub_traj1, sub_traj2):
                    sub_traj_.append(pt1[None] + (pt2 - pt1)[None]*t[..., None])

                sub_traj = np.concatenate(sub_traj_, axis=0)

            # sdf = sdf_query(sub_traj.astype(np.float32)) - r
            # sdf = proximity.signed_distance(sub_traj.astype(np.float32)) + r
                
            # (closest_points,
            # distances,
            # triangle_id) = mesh.nearest.on_surface(sub_traj.tolist())

            sdf = kdtree.query(sub_traj, workers=-1)[0] - r 

            sdf = sdf.astype(np.float64)
            sdf_min = np.min(sdf)

            # calculate path length
            length = np.sum(np.linalg.norm(sub_traj[:-1] - sub_traj[1:], axis=-1))

            sdfs.append(sdf.tolist())
            sdfs_min.append(sdf_min)
            path_lengths.append(length)
            print('Trajectory ', i) 

    data = {
        'sdfs': sdfs,
        'sdfs_min': sdfs_min,
        'path_lengths': path_lengths
    }

    full_data[method] = data

with open(f'{name}_data.json', 'w') as f:
    json.dump(full_data, f, indent=4)
