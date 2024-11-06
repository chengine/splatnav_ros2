#%%
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from plotting.plot_utils import plot_ellipse, plot_polytope
import polytope
from polytopes.decomposition import compute_polytope
from ellipsoids.intersection_utils import compute_intersection_linear_motion
from polytopes.polytopes_utils import compute_path_in_polytope
from SFC.corridor_utils import Corridor
from matplotlib import pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The point of this script is to test if the ellipsoid line test and the polytope generation is working.
# We will initialize Gaussians in a C shape with the robot starting at the center and moving outward.

# Parameters
n_gaussians = 20        # per side
n_dim = 3
t = np.linspace(0, 1, n_gaussians)

# First side
quats1 = np.random.rand(n_gaussians, 4)
# quats1[:, :2] = 0.      # Only rotates about z-axis
quats1 = quats1 / np.linalg.norm(quats1, axis=1)[:, None]
rots1 = Rotation.from_quat(quats1).as_matrix()
scales1 = np.random.rand(n_gaussians, n_dim) * 0.1 + 0.05

delta_x = np.array([1., 0., 0.])
means1 = np.array([-0.5, 0.5, 0.])[None, :] + delta_x[None, :] * t[:, None]

# second side
quats2 = np.random.rand(n_gaussians, 4)
# quats2[:, :2] = 0.      # Only rotates about z-axis
quats2 = quats2 / np.linalg.norm(quats2, axis=1)[:, None]
rots2 = Rotation.from_quat(quats2).as_matrix()
scales2 = np.random.rand(n_gaussians, n_dim) * 0.1 + 0.05

delta_x = np.array([0., 1., 0.])
means2 = np.array([-0.5, -0.5, 0.])[None, :] + delta_x[None, :] * t[:, None]

# third side
quats3 = np.random.rand(n_gaussians, 4)
# quats3[:, :2] = 0.      # Only rotates about z-axis
quats3 = quats3 / np.linalg.norm(quats3, axis=1)[:, None]
rots3 = Rotation.from_quat(quats3).as_matrix()
scales3 = np.random.rand(n_gaussians, n_dim) * 0.1 + 0.05

delta_x = np.array([-1., 0., 0.])
means3 = np.array([0.5, -0.5, 0.])[None, :] + delta_x[None, :] * t[:, None]

quats = np.concatenate([quats1, quats2, quats3], axis=0)
rots = np.concatenate([rots1, rots2, rots3], axis=0)[..., :n_dim, :n_dim]
scales = np.concatenate([scales1, scales2, scales3], axis=0)
means = np.concatenate([means1, means2, means3], axis=0)

# Convert to tensors
rots = torch.tensor(rots, dtype=torch.float32, device=device)
scales = torch.tensor(scales, dtype=torch.float32, device=device)
means = torch.tensor(means, dtype=torch.float32, device=device)

#%%

delta_x = torch.tensor([0.75, 0.0, 0.], device=device)
x0 = torch.tensor([-0.25, 0., 0.], device=device)

# delta_x = 1.5*torch.tensor([1., 1.], device=device)
# x0 = torch.tensor([-0.75, -0.75], device=device)

# x0 = torch.tensor([0., 0.], device=device)
# delta_x = 0.25*torch.tensor([1., 1.], device=device)

# robot parameters
radius = 0.15

R_B = np.random.rand(4)
R_B /= np.linalg.norm(R_B)
R_B = Rotation.from_quat(R_B).as_matrix()
R_B = torch.tensor(R_B, dtype=torch.float32, device=device)[:n_dim, :n_dim]
S_B = torch.rand(n_dim, device=device) * 0.1 + 0.05

torch.cuda.synchronize()
tnow = time.time()
output = compute_intersection_linear_motion(x0, delta_x, rots, scales, means, 
                                   R_B=None, S_B=radius, collision_type='sphere', 
                                   mode='bisection', N=10)
# output = compute_intersection_linear_motion(x0, delta_x, rots, scales, means, 
#                                    R_B=R_B, S_B=S_B, collision_type='ellipsoid', 
#                                    mode='bisection', N=10)
torch.cuda.synchronize()
print('Time to compute intersections:', time.time() - tnow)

segment = torch.stack([x0, x0 + delta_x], dim=0)
check1 = torch.einsum('bij, bjk, bkl->bil', (segment[0][None] - output['mu_A'])[..., None, :], output['Q_opt'], 
                (segment[0][None] - output['mu_A'])[..., None] ).squeeze()
check2 = torch.einsum('bij, bjk, bkl->bil', (segment[1][None] - output['mu_A'])[..., None, :], output['Q_opt'], 
        (segment[1][None] - output['mu_A'])[..., None] ).squeeze()

try:
    assert torch.all( check1 - output['K_opt'] >= -1e-4)
    assert torch.all( check2 - output['K_opt'] >= -1e-4)
except:
    print(f"Check failed", check1, check2, output['K_opt'])

torch.cuda.synchronize()
tnow = time.time()
A, b, pts = compute_polytope(output['deltas'], output['Q_opt'], output['K_opt'], output['mu_A'])
torch.cuda.synchronize()
print('Time to compute polytopes:', time.time() - tnow)
#%% Plot ellipsoids
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))

robot_plot_kwargs = {
    'facecolor': 'blue',
    'edgecolor': None,
    'alpha': 0.5
}
t = np.linspace(0, 1, 100)
robot_line = x0[None, :] + torch.tensor(t[:, None], device=device) * delta_x[None, :]
Sigma_B = torch.eye(n_dim) * (radius**2)
# Sigma_B = R_B * S_B[None, :]
# Sigma_B = Sigma_B @ Sigma_B.T

# plot robot body along the line
for pt in robot_line:
    plot_ellipse(pt, Sigma_B, 1., ax1, **robot_plot_kwargs)

# plot ellipsoids and colored red if they intersect
not_intersects = output['is_not_intersect']
for (rot, scale, mean, not_intersect) in zip(rots, scales, means, not_intersects):
    
    if not_intersect:
        color = 'green'
    else:
        color = 'red'

    ellipsoid_plot_kwargs = {
    'facecolor': color,
    'edgecolor': None,
    'alpha': 0.5
    }
    # Compute covariances
    Q = rot * scale[None, :]
    Sigma = Q @ Q.T

    plot_ellipse(mean, Sigma, 1., ax1, **ellipsoid_plot_kwargs)

# plot polytope
poly = polytope.Polytope(A[:,:2].cpu().numpy(), b.cpu().numpy())
plot_polytope(poly, ax1)
ax1.set_title('Splat-Nav')
# plot the pivot points
ax1.scatter(pts[:, 0].cpu().numpy(), pts[:, 1].cpu().numpy(), color='black', s=10)
ax1.scatter(pts[:, 0].cpu().numpy(), pts[:, 1].cpu().numpy(), color='black', s=10)

# plot config
ax1.scatter(x0[0].cpu().numpy(), x0[1].cpu().numpy(), color='red', s=10)
ax1.scatter(x0[0].cpu().numpy() + delta_x[0].cpu().numpy(), x0[1].cpu().numpy() + delta_x[1].cpu().numpy(), color='red', s=10)

ax1.set_xlim(-1, 3)
ax1.set_ylim(-2, 2)
# %% Using SFC

# init corridor
vmax = 0.1
amax = 0.01
# Robot configuration
robot_config = {
    'radius': radius,
    'vmax': vmax,
    'amax': amax,
}

#cor = Corridor(radius_robot=radius, vmax=vmax, amax=amax)

point_cloud = torch.column_stack((means, torch.zeros(means.shape[0], device=device).view(-1,1)))

x0 = torch.tensor([0., 0., 0.], device=device)
delta_x = 0.25*torch.tensor([1., 1., 0.], device=device)
line_segment = torch.stack([x0, x0 + delta_x])

# build sfc
ellipsoid, d, p_stars = cor.find_ellipsoid(line_segment, point_cloud)
As, bs, ps_star = cor.find_polyhedron(point_cloud, d, ellipsoid)


#%% 
#fig, ax = plt.subplots(1, figsize=(10, 10))

for pt in robot_line:
    plot_ellipse(pt, Sigma_B, 1., ax2, **robot_plot_kwargs)

for mu in point_cloud:    
    if not_intersect:
        color = 'green'
    else:
        color = 'red'

    ellipsoid_plot_kwargs = {
    'facecolor': color,
    'edgecolor': None,
    'alpha': 0.5
    }
    ax2.scatter(mu[0].cpu().numpy(), mu[1].cpu().numpy(), color=color)
bs_shrunk = bs - cor.rad_robot * torch.norm(As, dim=1)
poly= polytope.Polytope(As[:,:2].cpu().numpy(), bs_shrunk.cpu().numpy())
plot_polytope(poly, ax2)
ax2.set_title('SFC')
ax2.set_xlim(-1, 3)
ax2.set_ylim(-2, 2)

#%% Setup quasi-point clouds for splat-nav test

x0 = torch.tensor([0., 0.], device=device)
delta_x = 0.25*torch.tensor([1., 1.], device=device)

pc_rots = torch.eye(2, device=device).unsqueeze(0).repeat(means.shape[0], 1, 1)
scale_down = 0
pc_scales = torch.zeros_like(scales) 
output_pc = compute_intersection_linear_motion(x0, delta_x, pc_rots, pc_scales, means, 
                                   R_B=R_B, S_B=radius, collision_type='sphere', 
                                   mode='bisection', N=10)

A_pc, b_pc, pts_pc = compute_polytope(output_pc['deltas'], output_pc['Q_opt'], output_pc['K_opt'], output_pc['mu_A'])
b_pc = b_pc - radius

#fig, ax = plt.subplots(1, figsize=(10, 10))

for pt in robot_line:
    plot_ellipse(pt, Sigma_B, 1., ax3, **robot_plot_kwargs)

for mu in point_cloud:    
    if not_intersect:
        color = 'green'
    else:
        color = 'red'

    ellipsoid_plot_kwargs = {
    'facecolor': color,
    'edgecolor': None,
    'alpha': 0.5
    }
    ax3.scatter(mu[0].cpu().numpy(), mu[1].cpu().numpy(), color=color)
poly = polytope.Polytope(A_pc.cpu().numpy(), b_pc.cpu().numpy())
plot_polytope(poly, ax3)
ax3.set_title('Splat-Plan W/ Point Cloud Obstacles')
ax3.set_xlim(-1, 3)
ax3.set_ylim(-2, 2)

plt.show()
