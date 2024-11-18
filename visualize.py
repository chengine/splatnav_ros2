import os
import time
from pathlib import Path
from splat.splat_utils import GSplatLoader
import torch
import numpy as np
import trimesh
import imageio
import json
import viser
import viser.transforms as tf
import matplotlib as mpl
import open3d as o3d
import scipy
from polytopes.polytopes_utils import find_interior

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_polytope_trimesh(polytopes, colors=None):
    for i, (A, b) in enumerate(polytopes):
        # Transfer all tensors to numpy
        pt = find_interior(A, b)

        halfspaces = np.concatenate([A, -b[..., None]], axis=-1)
        hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)
        qhull_pts = hs.intersections

        output = trimesh.convex.convex_hull(qhull_pts)

        if colors is not None:
            output.visual.face_colors = colors[i]
            output.visual.vertex_colors = colors[i]

        if i == 0:
            mesh = output
        else:
            mesh += output
    
    return mesh

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

# path_to_gsplat = Path('outputs/configs/ros-depth-splatfacto/2024-10-30_140655/config.yml')
# lower_bound = torch.tensor([-4.5, -3., -2.], device=device)
# upper_bound = torch.tensor([3., 4.5, 0.], device=device)
# bounds = torch.stack([lower_bound, upper_bound], dim=-1)
#is_flipped = True


path_to_gsplat = Path('outputs/statues/splatfacto/2024-09-11_095852/config.yml')
scene_name = 'statues'
bounds = None
is_flipped = False


gsplat = GSplatLoader(path_to_gsplat, device)


if is_flipped:
    rotation = tf.SO3.from_x_radians(np.pi).wxyz
else:
    rotation = tf.SO3.from_x_radians(0.0).wxyz

server = viser.ViserServer()

if bounds is not None:
    mask = torch.all((gsplat.means - bounds[:, 0] >= 0) & (bounds[:, 1] - gsplat.means >= 0), dim=-1)
else:
    mask = torch.ones(gsplat.means.shape[0], dtype=torch.bool, device=device)

means = gsplat.means[mask]
covs = gsplat.covs[mask]
colors = gsplat.colors[mask]
opacities = gsplat.opacities[mask]

if not os.path.exists(f"assets/{scene_name}.obj"):
    gsplat.save_mesh(f"assets/{scene_name}.obj", bounds=bounds, res=4)

mesh = trimesh.load_mesh(str(Path(__file__).parent / f"assets/{scene_name}.obj"))
assert isinstance(mesh, trimesh.Trimesh)
vertices = mesh.vertices
faces = mesh.faces
print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

voxels = trimesh.load_mesh(str(Path(__file__).parent / f"assets/{scene_name}_voxel.obj"))
robot = trimesh.load_mesh(str(Path(__file__).parent / f"assets/drone/Assembly1.obj"))

robot = as_mesh(robot)

bounding_box = trimesh.bounds.oriented_bounds(robot)

# transform robot to origin
robot.apply_translation(bounding_box[0][:3, -1])
# robot.apply_transform(bounding_box[0])

# calculate diagonal of bounding box
diagonal = np.linalg.norm(bounding_box[1])

# scale robot down 
radius = 0.03
scale = radius / diagonal
robot.apply_scale(scale)

server.scene.add_gaussian_splats(
    name="/splats",
    centers= means.cpu().numpy(),
    covariances= covs.cpu().numpy(),
    rgbs= colors.cpu().numpy(),
    opacities= opacities.cpu().numpy(),
    wxyz=rotation,
)

server.scene.add_mesh_simple(
    name="/simple",
    vertices=vertices,
    faces=faces,
    material='standard',
    color=np.array([0.5, 0.5, 0.5]),
    wxyz=rotation,
    opacity=0.5,
    wireframe=True
)

# server.scene.add_mesh_trimesh(
#     name="/trimesh",
#     mesh=mesh,
#     wxyz=rotation,
# )

# server.scene.add_mesh_simple(
#     name="/voxel",
#     vertices=voxels.vertices,
#     faces=voxels.faces,
#     material='standard',
#     wireframe=True,
#     opacity=0.2,
#     wxyz=rotation
# )

# server.scene.add_mesh_simple(
#     name="/robot",
#     vertices=robot.vertices,
#     faces=robot.faces,
#     material='standard',
#     # wireframe=True,
#     color=np.array([.75, 75., 0.1]),
#     opacity=0.75,
#     wxyz=rotation,
# )


### FLIGHTROOM ###
# for i in range(4):
#     # Load in trajectories
#     traj_filepath = f'traj/data_closed-loop_{i}.json'

#     with open(traj_filepath, 'r') as f:
#         meta = json.load(f)

#     data = meta['total_data']
#     data = data[0]
#     traj = np.array(data['traj'])[:, :3]

#     points = np.stack([traj[:-1], traj[1:]], axis=1)

#     cmap = mpl.cm.get_cmap('turbo')
#     colors = np.array([cmap(i) for i in np.linspace(0, 1, len(points))])[..., :3]
#     colors = colors.reshape(-1, 1, 3)

#     # Add trajectory to scene
#     server.scene.add_line_segments(
#         name=f"/traj_{i}",
#         points=points,
#         colors=colors,
#         line_width=10,
#         wxyz=rotation,
#     )

#     # add polytopes to scene
#     polytopes = data['polytopes']
#     polytopes = [(np.array(polytope)[..., :3], np.array(polytope)[..., 3]) for polytope in polytopes]
    
#     colors = np.array([cmap(i) for i in np.linspace(0, 1, len(polytopes))])[..., :3]
#     colors = colors.reshape(-1, 3)
#     colors = np.concatenate([colors, 0.1*np.ones((len(polytopes), 1))], axis=-1)
#     colors = (255*colors).astype(np.uint8)
#     corridor_mesh = create_polytope_trimesh(polytopes, colors=colors)

    # server.scene.add_mesh_trimesh(
    #     name=f"/corridor_{i}",
    #     mesh=corridor_mesh,
    #     wxyz=rotation,
    #     #position=(0.0, 5.0, 0.0),
    # )

#     vertices = corridor_mesh.vertices
#     faces = corridor_mesh.faces

#     server.scene.add_mesh_simple(
#     name=f"/corridor_{i}",
#     vertices=vertices,
#     faces=faces,
#     color=np.array([0., 1., 0.]),
#     #material='standard',
#     #wireframe=True,
#     opacity=0.3,
#     wxyz=rotation
# )

# # Load in trajectories
traj_filepath = f'assets/statues_splatplan_processed.json'

with open(traj_filepath, 'r') as f:
    meta = json.load(f)

datas = meta['total_data']


### PROGRESSION RENDER ###

# not_done = True
# while not_done:
#     # Get all currently connected clients.
#     client = server.get_clients()
#     print("Connected client IDs", client.keys())

#     if len(client) == 0:
#         time.sleep(.5)
#         continue

#     # for id, client in client.items():
#     #     output = client.get_render(height=800, width=800)
#     #     print("Got image with shape", output.shape)

#     #    imageio.imwrite(f"renders/mesh_trimesh/{id}_position{pos_id}.png", output)

#     for id, client in client.items():
#         for i in range(len(datas[0:1])):
#             data = datas[91]        #91
#             traj = np.array(data['traj'])[:, :3]

#             points = np.stack([traj[:-1], traj[1:]], axis=1)
#             margins = np.array(data['safety_margin'])
#             margins = (margins - np.min(margins)) / (np.max(margins) - np.min(margins))

#             # Progression color
#             astar = np.array(data['path'])
#             astar_points = np.stack([astar[:-1], astar[1:]], axis=1)

#             cmap = mpl.cm.get_cmap('turbo')
#             colors = np.array([cmap(i) for i in np.linspace(0, 1, len(astar_points))])[..., None, :3]

#             # for j in range(len(astar_points)):
#             #     # Add trajectory to scene
#             #     server.scene.add_line_segments(
#             #         name="/astar",
#             #         points=astar_points[:j],
#             #         colors=colors[:j],
#             #         line_width=10,
#             #         wxyz=rotation,
#             #     )

#             #     output = client.get_render(height=1080, width=1920)
#             #     imageio.imwrite(f"renders/astar_draw/{id}_{j}.png", output)

#             # server.scene.add_line_segments(
#             #     name="/astar",
#             #     points=astar_points,
#             #     colors=colors,
#             #     line_width=10,
#             #     wxyz=rotation,
#             # )

#             # add polytopes to scene
#             polytopes = data['polytopes']
#             polytopes = [(np.array(polytope)[..., :3], np.array(polytope)[..., 3]) for polytope in polytopes]

#             # colors = np.array([cmap(i) for i in np.linspace(0, 1, len(polytopes))])[..., :3]
#             # colors = colors.reshape(-1, 3)
#             # colors = np.concatenate([colors, 0.1*np.ones((len(polytopes), 1))], axis=-1)
#             # colors = (255*colors).astype(np.uint8)

#             # for j in range(len(polytopes)):
#             #     if j == 0:
#             #         continue

#             #     corridor_mesh = create_polytope_trimesh(polytopes[0:j])

#             #     vertices = corridor_mesh.vertices
#             #     faces = corridor_mesh.faces

#             #     server.scene.add_mesh_simple(
#             #     name=f"/corridor_{i}",
#             #     vertices=vertices,
#             #     faces=faces,
#             #     color=np.array([0.1, 0.5, 0.5]),
#             #     material='standard',
#             #     # wireframe=True,
#             #     opacity=0.65,
#             #     wxyz=rotation
#             #     )

#             #     output = client.get_render(height=1080, width=1920)
#             #     imageio.imwrite(f"renders/polytope_draw/{id}_{j}.png", output)

#             corridor_mesh = create_polytope_trimesh(polytopes)

#             vertices = corridor_mesh.vertices
#             faces = corridor_mesh.faces

#             server.scene.add_mesh_simple(
#             name=f"/corridor_{i}",
#             vertices=vertices,
#             faces=faces,
#             color=np.array([0.1, 0.5, 0.5]),
#             material='standard',
#             # wireframe=True,
#             opacity=0.65,
#             wxyz=rotation
#             )

#             # Safety margin color
#             cmap = mpl.cm.get_cmap('jet')
#             colors = np.array([cmap(1.- margin) for margin in margins[1:]])[..., :3]
#             colors = colors.reshape(-1, 1, 3)

#             # Add trajectory to scene
#             server.scene.add_line_segments(
#                 name=f"/traj_{i}",
#                 points=points,
#                 colors=colors,
#                 line_width=10,
#                 wxyz=rotation,
#             )
            
#         # client.camera.position = np.array([-0.66611781, -0.05211388,  0.18852854])
#         # client.camera.wxyz = np.array([-0.38550102,  0.61630354, -0.58218947,  0.3641625 ])
#         # output = client.get_render(height=1080, width=1920)
#         # imageio.imwrite(f"renders/traj.png", output)

#     not_done = False
#     # kill the server
#     time.sleep(1.0)

for i, data in enumerate(datas):
    traj = np.array(data['traj'])[:, :3]

    points = np.stack([traj[:-1], traj[1:]], axis=1)
    margins = np.array(data['safety_margin'])
    margins = (margins - np.min(margins)) / (np.max(margins) - np.min(margins))

    # Safety margin color
    cmap = mpl.cm.get_cmap('jet')
    colors = np.array([cmap(1.- margin) for margin in margins[1:]])[..., :3]
    colors = colors.reshape(-1, 1, 3)

    # Add trajectory to scene
    server.scene.add_line_segments(
        name=f"/traj_{i}",
        points=points,
        colors=colors,
        line_width=10,
        wxyz=rotation,
    )

    # yaw = 0.0
    # yaws = []
    # for idx, pt in enumerate(points[:, 0]):
    #     if idx < len(points[:, 0]) - 1:

    #         diff = points[:, 0][idx + 1] - points[:, 0][idx]

    #         prev_yaw = yaw
    #         yaw = np.arctan2(diff[1], diff[0])

    #         closest_k = np.round(-(yaw - prev_yaw) / (2*np.pi))
    #         yaw = yaw + 2*np.pi*closest_k     

    #     yaws.append(yaw)

    # yaws = np.stack(yaws) + np.pi/2

    # for j, (point, vec) in enumerate(zip(points[::5], yaws[::5])):
    #     server.scene.add_mesh_simple(
    #         name=f"/robot_{j}",
    #         vertices=robot.vertices,
    #         faces=robot.faces,
    #         material='standard',
    #         # wireframe=True,
    #         color=np.array([.75, 75., 0.1]),
    #         opacity=0.75,
    #         wxyz=tf.SO3.from_z_radians(vec).wxyz,
    #         position=point[0],
    #     )


### SPIRAL RENDER ###
# N = 500
# t = np.linspace(0, 6*np.pi, N)
# scaling = np.linspace(.5, 0.3, N)
# positions = np.stack([scaling*np.sin(t), scaling*np.cos(t), 0.5*np.linspace(0., 1., N)], axis=-1)

# time.sleep(1.0)
# not_done = True
# while not_done:
#     # Get all currently connected clients.
#     client = server.get_clients()
#     print("Connected client IDs", client.keys())

#     if len(client) == 0:
#         time.sleep(1.0)
#         continue

#     for id, client in client.items():
#         for pos_id, pos in enumerate(positions):

#             client.camera.position = pos
#             client.camera.look_at = np.array([0.0, 0.0, 0.0])
#             output = client.get_render(height=int(0.9*1080), width=int(0.9*1920))
#             print("Got image with shape", output.shape)

#             imageio.imwrite(f"renders/mesh_trimesh/{id}_position{pos_id}.png", output)

#     # kill the server
#     time.sleep(1.0)
#     not_done = False

### ONE TIME RENDER ###

# not_done = True
# while not_done:
#     # Get all currently connected clients.
#     client = server.get_clients()
#     print("Connected client IDs", client.keys())

#     if len(client) == 0:
#         time.sleep(1.0)
#         continue

#     for id, client in client.items():
#         output = client.get_render(height=1080, width=1920)
#         print("Got image with shape", output.shape)

#         imageio.imwrite(f"renders/corridor_traj.png", output)

#     not_done = False
#     # kill the server
#     time.sleep(1.0)

while True:
    time.sleep(10.0)