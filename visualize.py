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

path_to_gsplat = Path('outputs/configs/ros-depth-splatfacto/2024-10-30_140655/config.yml')
# path_to_gsplat = Path('outputs/statues/splatfacto/2024-09-11_095852/config.yml')

# scene_name = 'statues'
gsplat = GSplatLoader(path_to_gsplat, device)

lower_bound = torch.tensor([-4.5, -3., -2.], device=device)
upper_bound = torch.tensor([3., 4.5, 0.], device=device)
bounds = torch.stack([lower_bound, upper_bound], dim=-1)

# bounds = None

is_flipped = True

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

# if not os.path.exists(f"assets/{scene_name}.obj"):
#     gsplat.save_mesh(f"assets/{scene_name}.obj", bounds=bounds, res=4)

# mesh = trimesh.load_mesh(str(Path(__file__).parent / f"assets/{scene_name}.obj"))
# assert isinstance(mesh, trimesh.Trimesh)

# vertices = mesh.vertices
# faces = mesh.faces
# print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

server.scene.add_gaussian_splats(
    name="/splats",
    centers= means.cpu().numpy(),
    covariances= covs.cpu().numpy(),
    rgbs= colors.cpu().numpy(),
    opacities= opacities.cpu().numpy(),
    wxyz=rotation,
)

# server.scene.add_mesh_simple(
#     name="/simple",
#     vertices=vertices,
#     faces=faces,
#     material='standard',
#     wxyz=rotation
# )

# server.scene.add_mesh_trimesh(
#     name="/trimesh",
#     mesh=mesh,
#     wxyz=rotation,
#     #position=(0.0, 5.0, 0.0),
# )

for i in range(4):
    # Load in trajectories
    traj_filepath = f'traj/data_closed-loop_{i}.json'

    with open(traj_filepath, 'r') as f:
        meta = json.load(f)

    data = meta['total_data']
    data = data[0]
    traj = np.array(data['traj'])[:, :3]

    points = np.stack([traj[:-1], traj[1:]], axis=1)

    cmap = mpl.cm.get_cmap('turbo')
    colors = np.array([cmap(i) for i in np.linspace(0, 1, len(points))])[..., :3]
    colors = colors.reshape(-1, 1, 3)

    # Add trajectory to scene
    server.scene.add_line_segments(
        name=f"/traj_{i}",
        points=points,
        colors=colors,
        line_width=10,
        wxyz=rotation,
    )

    # add polytopes to scene
    polytopes = data['polytopes']
    polytopes = [(np.array(polytope)[..., :3], np.array(polytope)[..., 3]) for polytope in polytopes]
    
    colors = np.array([cmap(i) for i in np.linspace(0, 1, len(polytopes))])[..., :3]
    colors = colors.reshape(-1, 3)
    colors = np.concatenate([colors, 0.1*np.ones((len(polytopes), 1))], axis=-1)
    colors = (255*colors).astype(np.uint8)
    corridor_mesh = create_polytope_trimesh(polytopes, colors=colors)

    # server.scene.add_mesh_trimesh(
    #     name=f"/corridor_{i}",
    #     mesh=corridor_mesh,
    #     wxyz=rotation,
    #     #position=(0.0, 5.0, 0.0),
    # )

    vertices = corridor_mesh.vertices
    faces = corridor_mesh.faces

    server.scene.add_mesh_simple(
    name=f"/corridor_{i}",
    vertices=vertices,
    faces=faces,
    color=np.array([0., 1., 0.]),
    #material='standard',
    #wireframe=True,
    opacity=0.3,
    wxyz=rotation
)

# @server.on_client_connect
# def _(client: viser.ClientHandle) -> None:
#     print("new client!")

#     # This will run whenever we get a new camera!
#     @client.camera.on_update
#     def _(_: viser.CameraHandle) -> None:
#         print(f"New camera on client {client.client_id}!")

#     # Show the client ID in the GUI.
#     gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
#     gui_info.disabled = True

# N = 500
# t = np.linspace(0, 6*np.pi, N)
# scaling = np.linspace(1., 0.5, N)
# positions = np.stack([scaling*np.sin(t), scaling*np.cos(t), 0.5*np.linspace(0., 1., N)], axis=-1)

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
#             output = client.get_render(height=800, width=800)
#             print("Got image with shape", output.shape)

#             imageio.imwrite(f"renders/mesh_trimesh/{id}_position{pos_id}.png", output)

#     not_done = False
#     # kill the server
#     time.sleep(1.0)

while True:
    time.sleep(10.0)