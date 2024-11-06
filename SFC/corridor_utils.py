from .astar_utils import *
import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import time

from initialization.grid_utils import GSplatVoxel
from polytopes.collision_set import GSplatCollisionSet
from polytopes.polytopes_utils import h_rep_minimal, find_interior, compute_segment_in_polytope

# All functions are adapted and named accordingly to the "Planning Dynamically Feasible Trajectories for Quadrotors Using Safe Flight Corridors in 3-D Complex Environments"
#  paper (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7839930). These functions have been adapted into Pytorch.
class Corridor():
    def __init__(self, gsplat, robot_config, env_config, spline_planner, device) -> None:
        # Rs is the radius around the path, determined by the max velocity and acceleration of the robot.

        self.gsplat = gsplat
        self.device = device
        self.env_config = env_config
        self.spline_planner = spline_planner
         # Robot configuration
        self.radius = robot_config['radius']
        self.vmax = robot_config['vmax']
        self.amax = robot_config['amax']
        self.collision_set = GSplatCollisionSet(self.gsplat, self.vmax, self.amax, self.radius, self.device)
        
        
        # Environment configuration (specifically voxel)
        self.lower_bound = env_config['lower_bound']
        self.upper_bound = env_config['upper_bound']
        self.resolution = env_config['resolution']

        tnow = time.time()
        torch.cuda.synchronize()
        self.gsplat_voxel = GSplatVoxel(self.gsplat, lower_bound=self.lower_bound, upper_bound=self.upper_bound, resolution=self.resolution, radius=self.radius, device=device)
        torch.cuda.synchronize()
        print('Time to create GSplatVoxel:', time.time() - tnow)

        
        self.rs = robot_config['vmax']**2 / (2 * robot_config['amax'])


        self.up = torch.tensor([[0, 0, 1]], dtype=torch.float32, device=device)
        
        # init bounding box constriants values as empty
        self.box_As = torch.tensor([])
        self.box_bs = torch.tensor([])


        # Record times
        self.times_collision = []
        self.times_ellipse = []
        self.times_polytope = []        

    def bounding_box_and_points(self, path, point_cloud):
        # path: N+1 x 3
        # point_cloud: M x 3

        # Find the bounding box hyperplanes of the path

        # Without loss of generality, let us assume the local coordinate system is at the midpoint of the path and local x is along the path
        # We artibrarily choose y and z accordingly.
        # NOTE: Might have to keepdims to the norms
        midpoints = 0.5 * (path[1:] + path[:-1])        # N x 3
        lengths = torch.linalg.norm(path[1:] - path[:-1], dim=1)        # N
        lengths_x = lengths + self.rs
        local_x = (path[1:] - path[:-1]) / torch.linalg.norm(path[1:] - path[:-1], dim=-1, keepdim=True)      # This is the pointing direction of the path (N x 3)
        local_y = torch.cross(self.up, local_x)   # This is the direction perpendicular to the path and the z-axis
        local_y = local_y / torch.norm(local_y, dim=-1, keepdim=True)
        local_z = torch.cross(local_x, local_y)    # This is the direction perpendicular to the path and the y-axis

        rotation_matrix = torch.stack([local_x, local_y, local_z], dim=-1)    # This is the local x,y,z to world frame rotation (N x 3 x 3)

        # These vectors form the normal of the hyperplanes. We simply need to find their intercepts. 
        # We are basically trying to find a_i.T * (x_0 + l_i * a_i) = b_i = a_i.T * x_0 + l_i (since a_i.T * a_i = 1)
        xyz_lengths = torch.stack([lengths_x, lengths, lengths], dim=-1)
        intercepts_pos = torch.bmm(midpoints[..., None, :], rotation_matrix).squeeze() + torch.stack([lengths_x, lengths, lengths], dim=-1)    # N x 3
        intercepts_neg = -(intercepts_pos) + 2*xyz_lengths    # N x 3

        # Represent as x.T A <= b
        A = torch.cat([rotation_matrix, -rotation_matrix], dim=-1)    # N x 3 x 6
        b = torch.cat([intercepts_pos, intercepts_neg], dim=-1)    # N x 6

        # Now return points that are only within the bounding box
        xA = torch.einsum('mk, nkl -> nml', point_cloud, A)    # N x M x 6

        # In order for a pt to be within the bounding box, it must satisfy xA <= b, i.e. xA-b <= 0
        mask = (xA - b[:, None, :] <= 0.)    # N x M x 6
        keep_mask = mask.all(dim=-1)    # N x M      # For every path segment N, keep_pts is a boolean of every point in M if it is in the bounding box or not

        # Returning just the points is not parallelizable because there can be different number of points for each bounding box, so we need to loop, selecting the points
        # where the mask is True.
        keep_points = []
        for keep_per_n in keep_mask:
            keep_points.append(point_cloud[keep_per_n])

        # saves bounding box constraints for polytope definition
        self.box_As = torch.transpose(A[0], 0, 1)
        self.box_bs = b[0]

        # Return both the bounding box half-space representation and the relevant points
        output = {'A': A, 'b': b, 'midpoints': midpoints, 'keep_points': keep_points}
        return output

    def create_corridor(self, path, point_cloud):
        output = self.bounding_box_and_points(path, point_cloud)

        line_segments = torch.stack([path[:-1], path[1:]], dim=1)    # N x 2 x 3

        A_box = torch.transpose(output['A'], 1, 2)
        b_box = output['b']
        midpoints = output['midpoints']
        keep_points = output['keep_points']

        # NOTE: This may be parallelizable, but likely not because different number of points for each bounding box (unless we pad the tensor). 
        As = []
        bs = []
        for (mid, keep_points_per_n, A_box_per_n, b_box_per_n, line_segment) in zip(midpoints, keep_points, A_box, b_box, line_segments):

            # If the number of points in the box is 0, we don't need to do anything.
            if len(keep_points_per_n) == 0:
                As.append(A_box_per_n)
                bs.append(b_box_per_n)

            # otherwise we need to find the ellipsoid and polyhedron
            else:
                ellipsoid = self.find_ellipsoid(line_segment, keep_points_per_n)
                A, b, pivots = self.find_polyhedron(keep_points_per_n, mid, ellipsoid)

                A, b = self.shrink_corridor(A, b, line_segment, pivots)

                if A is None:
                    raise ValueError("No feasible polytope found.")

                # NOTE: Might need to squeeze some dimensions
                As.append(torch.cat([A, A_box_per_n], dim=0))
                bs.append(torch.cat([b, b_box_per_n], dim=0))

        return As, bs

    def find_ellipsoid(self, line_segment, point_cloud):
        # line_segment: 2 x 3
        # point_cloud: M x 3

        # Frame1: Local frame with x-axis along the line segment, y-z are aligned with world
        # Frame2: After first iteration, x-y plane is aligned with p* and z is perpendicular to x-y plane (y-z are no longer aligned with frame1)

        # returns the ellipsoid dictionary {'R': R, 'S': S}, R is the rotation matrix from local frame to world frame. S is the scaling matrix.
        # So E = R @ S @ S.T @ R.T (this is flipped from the paper, but follows the Gaussian Splatting decomposition). 
        # For convenience, R is a matrix, but S is a vector that parametrized the diagonal scaling matrix.
 
        # find the ellipsoid center       
        p0, p1 = line_segment[0], line_segment[1]
        d = (p0 + p1) / 2
        # find length of segment
        direction = p1 - p0
        L = torch.norm(direction)

        # create initial rotation matrix
        world_x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=direction.dtype, device=direction.device)
        R = rotation_matrix_from_vectors(direction, world_x_axis)

        # initialize scale as half the length of the line segment
        S_orig = torch.tensor([L/2, L/2, L/2], dtype=direction.dtype, device=direction.device)

        # transform points to local frame
        shifted_points = point_cloud - d[None, :]
        rotated_points = shifted_points @ R.T 
        scaled_points = rotated_points / S_orig
        S = S_orig / S_orig # scale to unity

        # grab points inside the spheroid 
        dists = torch.sum(scaled_points**2, axis=1) 
        inside = dists < 1.0 - 1e-6

        closest_point = None
        
        while inside.any():     
       
            # grab closest point
            closest_id = torch.argmin(dists[inside])
            scaled_points_inside = scaled_points[inside]
            closest_point = scaled_points_inside[closest_id]
            # find scale factor for y-z axes
            distance_yz = closest_point[1]**2 + closest_point[2]**2
            denominator = 1.0 - closest_point[0]**2
            scale_factor = torch.sqrt(distance_yz / denominator)
            # update scales
            S[1] = scale_factor
            S[2] = scale_factor
            
            # redefine sphere using new scales
            scaled_points_norm = scaled_points / S
            
            # find any points inside new sphere
            dists = torch.sum(scaled_points_norm**2, axis=1)
            inside = dists < 1.0 - 1e-6

        if closest_point is not None: 
            p_star_xy = closest_point
            # remove p* from list of points
            scaled_points = scaled_points[torch.norm(scaled_points - p_star_xy[None, :], dim=1) > 0]

            # project p* onto yz plane
            p_yz = p_star_xy - p_star_xy[0] * torch.tensor([1, 0, 0], dtype=torch.float32, device=p_star_xy.device)
            p_yz_norm = p_yz / torch.norm(p_yz)
            
            # define current y-axis
            y_axis_R = torch.tensor([0, 1, 0], dtype=torch.float32, device=p_star_xy.device)
            
    
            cos_theta = torch.dot(y_axis_R, p_yz_norm)
            sin_theta = torch.sqrt(1 - cos_theta**2)  
            # rotation from original line segment align to final 
            R_2_1 = torch.tensor([
                [1, 0, 0],
                [0, cos_theta, -sin_theta],
                [0, sin_theta, cos_theta]
            ], dtype=torch.float32, device=p_star_xy.device)

            scaled_points = scaled_points @ R_2_1   
            # get p_star in world frame
            p_star_xy = ((p_star_xy * S_orig) @ R) + d

        else:  # no collision found
            # set p_star on ellipsoid boundary
            p_star_xy = torch.tensor([0, 0, 0], dtype=torch.float32, device=d.device)
            R_2_1 = torch.eye(3, dtype=torch.float32, device=d.device)

        # prep variables for z-axis shrink
        S[2] = 1 # reset z-axis scale to x-axis scale
        S_temp = S.clone()
        scaled_points = (scaled_points) / S_temp
        
        dists = torch.sum(scaled_points**2, axis=1) 
        inside = dists < 1.0 - 1e-6
        closest_point = None # reset closest point

        
        while inside.any():

            # Grab closest point
            closest_id = torch.argmin(dists[inside])
            scaled_points_inside = scaled_points[inside]
            closest_point = scaled_points_inside[closest_id]

            # find scale factor for z-axis
            distance_z = closest_point[2]**2
            denominator = 1.0 - (closest_point[0]**2) - (closest_point[1]**2)
            scale_factor = torch.sqrt(distance_z / denominator)

            # update scales
            S[2] = scale_factor
            # update points inside
            scaled_points_norm = scaled_points / S
            dists = torch.sum(scaled_points_norm**2, axis=1)
            inside = dists < 1.0 - 1e-6

        if closest_point is not None: 
            p_star_z = closest_point
            # find p* in world frame
            p_star_z = ((p_star_z * S_temp * S_orig) @ R_2_1.T @ R) + d
        else:  # no collision found
            # set p_star on ellipsoid boundary
            p_star_z = torch.tensor([0, 0, 0], dtype=torch.float32, device=d.device)
        
        # return final values in world frame
        R_final = R.T @ R_2_1
        S_final = S * S_orig
        p_stars = torch.stack([p_star_xy, p_star_z], axis=0) 

        ellipsoid = {'R': R_final, 'S': S_final}
        center = d  # in world frame 
        return ellipsoid, center, p_stars

    def find_polyhedron(self, point_cloud, midpoint, ellipsoid):
        # point_cloud: M x 3
        # midpoint: 3
        # ellipsoid: {'R': R, 'S': S}
        R = ellipsoid['R']
        S = ellipsoid['S']
        
        # We basically want to compute the Mahalanobis distance of each point from the ellipsoid. And sort them.
        # If it is less than 1, it is inside the ellipsoid.

        # However, it is actually faster (when doing this iteratively) to transform the entire coordinate frame
        # into the local frame of the ellipsoid, scale it so that it becomes a unit sphere, and then do the checking. This is
        # without loss of generality because intersections are preserved under linear transformations.

        # Shift the coordinate system to the midpoint
        shifted_points = point_cloud - midpoint[None, :]    # M x 3
        rotated_points = shifted_points @ R    # M x 3
        scaled_points = rotated_points / S    # M x 3
        # NOTE: Do we need a sanity check to make sure the ellipsoid is collision-free?
        #assert (torch.linalg.norm(scaled_points, dim=-1) > 1 - 1e-3).all()
        As = []
        bs = []
        ps_star = []
        while len(scaled_points) > 0:
            # Calculate the distance to the origin
            dists = torch.linalg.norm(scaled_points, dim=-1) # M
            # Find the closest distance and which point it corresponds to
            where_min = torch.argmin(dists)
            min_val = dists[where_min] # this is distance squared
            p_star = scaled_points[where_min]
            p_star_norm = p_star / torch.linalg.norm(p_star)

            # After finding the closest point, we compute the tangent hyperplane and delete those points that are outside this hyperplane
            # For spheres it is easy, the normal is just the point, and the intercept is the squared distance.
            # convert p_star to world frame and set half-space constraints
            As.append(p_star)
            bs.append(min_val**2)
            ps_star.append(p_star)

            # We now find and delete all points that are on or outside this hyperplane
            mask = torch.sum(scaled_points * p_star, dim=-1) < min_val**2 - 1e-2  # M_i, true if on the right side of the hyperplane
            scaled_points = scaled_points[mask] 

        # If we only performed one iteration, we can just return the hyperplane

        if len(As) == 0:
            box_As = self.box_As
            box_bs = self.box_bs
            return self.box_As, self.box_bs, None

        elif len(As) == 1:
            As = As[0][None]        # 1 x 3
            bs = bs[0].reshape(1,)    # 1
            ps_star = ps_star[0].reshape(1, -1)

        else:
        # otherwise we stack them
            As = torch.stack(As, dim=0)
            bs = torch.stack(bs)
            ps_star = torch.stack(ps_star, dim=0)


        # Now we need to transform the hyperplanes back to the world frame
        As = As * (1. / S)[None, :] @ R.T    # M x 3
        bs = bs + torch.sum(midpoint[None, :] * As, dim=-1)    # M

        return As, bs, ps_star
    

    def shrink_corridor(self, bs, box_bs):
        # As: M x 3
        # bs: M

        # this only shrinks the corridor, and does not pivot it
        bs = torch.cat([bs, box_bs])

        # find norm of each plane
        bs_new = bs - self.radius

        return bs_new

    # def shrink_and_pivot_corridor(self, As, bs, line_segment, pivots):
    #     # As: M x 3
    #     # bs: M
    #     # line_segment: 2 x 3
    #     # pivots: M x 3

    #     point1 = line_segment[0]
    #     point2 = line_segment[1]

    #     # NOTE: We need to segment out the hyperplanes that are within some radius of the robot. 
    #     # All hyperplanes should contain the line segment by design, and so the hyperplane
    #     # will not intersect the middle of the line segment, thus the min distance must occur at 
    #     # one of the end points.
    #     signed_distance_to_point1 = torch.sum( torch.matmul(As, torch.transpose((point1[None, :] - pivots), 0, 1)), dim=-1) / torch.linalg.norm(As, dim=-1)     # M
    #     signed_distance_to_point2 = torch.sum( torch.matmul(As, torch.transpose((point2[None, :] - pivots), 0, 1)), dim=-1) / torch.linalg.norm(As, dim=-1)     # M

    #     print(f'Signed Distance to Point 1: {signed_distance_to_point1}')
    #     print(f'Signed Distance to Point 2: {signed_distance_to_point2}')
    #     # As a sanity check, we should check if these signed distances are all negative (meaning the hyperplane is on the correct side).
    #     # TODO:
    #     hyperplane_ok = torch.logical_and( (signed_distance_to_point1 < -self.radius),  (signed_distance_to_point2 < -self.radius) )

    #     hyperplanes_to_adjust = As[~hyperplane_ok]
    #     pivots_to_adjust = pivots[~hyperplane_ok]

    #     # For the hyperplanes that we don't have to adjust, we need to shift them by the robot radius.
    #     # To do this, we basically move the pivot point by the robot radius in the direction opposite of the normal.
    #     hyperplane_ok_A = As[hyperplane_ok]

    #     deflated_pivots = pivots[hyperplane_ok] - self.radius * hyperplane_ok_A / torch.linalg.norm(hyperplane_ok_A, dim=-1, keepdim=True)    # M x 3

    #     hyperplane_ok_b = torch.sum( hyperplane_ok_A * deflated_pivots, dim=-1)    # M

    #     # The closest distance between a line segment and a plane has to be at one of the endpoints
    #     # if the line segment is not parallel to the plane and the line segment does not intersect the plane.
    #     # Note that the outgoing hyperplane is already pushed some R away from the pivot, and so is ready to go
    #     # in terms of using it for point-based planning.
    #     if hyperplanes_to_adjust.shape[0] > 0:  
    #         adjusted_A1, adjusted_b1, objs1 = find_closest_hyperplane(pivots_to_adjust, hyperplanes_to_adjust, point1, self.radius)        # M x 2 x 3, M x 2
    #         adjusted_A2, adjusted_b2, objs2 = find_closest_hyperplane(pivots_to_adjust, hyperplanes_to_adjust, point2, self.radius)

    #     if adjusted_A1 is None or adjusted_A2 is None:
    #         return None, None

    #     adjusted_A = torch.cat([adjusted_A1, adjusted_A2], dim=1)    # M x 4 x 3
    #     adjusted_b = torch.cat([adjusted_b1, adjusted_b2], dim=1)    # M x 4
    #     objs = torch.cat([objs1, objs2], dim=-1)    # M x 4

    #     # Test if the closest hyperplane using one point is still on the same side as the other point
    #     plane1_to_point2 = torch.sum( adjusted_A1 * point2[None, None, :], dim=-1) - adjusted_b1  # M x 2
    #     plane2_to_point1 = torch.sum( adjusted_A2 * point1[None, None, :], dim=-1) - adjusted_b2  # M x 2

    #     # planei_to_pointj should be less than 0 for at least one (i,j) pair / solution.
    #     # basically pick the option that has the greatest non-positive objective value
    #     plane_to_point = (torch.cat([plane1_to_point2, plane2_to_point1], dim=-1) <= 0.).to(int)    # M x 4
    #     plane_to_point_positive = (plane_to_point == 0)

    #     # If all solutions are on the wrong side, we quit
    #     if torch.any(torch.all(plane_to_point_positive, dim=-1)):
    #         return None, None
        
    #     plane_to_point[plane_to_point_positive] = int(1e6)
    #     plane_to_point_obj = plane_to_point * (objs + 1e-3)   # M x 4
    #     min_obj, min_idx = torch.min(plane_to_point_obj, dim=-1)    # M

    #     # Now we select the closest hyperplane
    #     adjusted_A = adjusted_A[torch.arange(adjusted_A.shape[0]), min_idx]    # M x 3
    #     adjusted_b = adjusted_b[torch.arange(adjusted_b.shape[0]), min_idx]    # M

    #     # Now we stack the good hyperplanes with the adjusted ones
    #     As = torch.cat([hyperplane_ok_A, adjusted_A], dim=0)
    #     bs = torch.cat([hyperplane_ok_b, adjusted_b], dim=0)

    #     return As, bs


    def generate_path(self, x0, xf):

        # Part 1: Computes the path seed using A*
        tnow = time.time()
        path = self.gsplat_voxel.create_path(x0, xf)
        torch.cuda.synchronize()
        time_astar = time.time() - tnow

        times_collision_set = 0
        times_ellipsoid = 0
        times_polytope = 0
        times_shrink = 0

        polytopes = []      # List of polytopes (A, b)
        segments = torch.tensor(np.stack([path[:-1], path[1:]], axis=1), device=self.device)

        for it, segment in enumerate(segments):
            # If this is the first line segment, we always create a polytope. Or subsequently, we only instantiate a polytope if the line segment
            midpoint = 0.5 * (segment[0] + segment[1])

            if it == 0 or it == len(segments) - 1:
                # Part 2: Computes the collision set
                tnow = time.time()
                output = self.collision_set.compute_set_one_step(segment)
                torch.cuda.synchronize()

                point_cloud = output['means']
                self.box_As = output['A_bb']
                self.box_bs = output['b_bb_shrunk']
                times_collision_set += time.time() - tnow


                if len(output['gaussian_ids']) == 0:
                    polytope = (self.box_As, self.box_bs)
                
                else:

                    # Part 3: Compute the ellipsoid
                    tnow = time.time()
                    ellipsoid, d, p_stars = self.find_ellipsoid(segment, point_cloud)
                    torch.cuda.synchronize()
                    times_ellipsoid += time.time() - tnow

                    # Part 4: Compute the polytope
                    tnow = time.time()
                    As, bs, ps_star = self.find_polyhedron(point_cloud, d, ellipsoid)
                    
                    # Part 5: Shrink the polytope
                    bs = bs - self.radius * torch.norm(As, dim=-1)

                    As = torch.cat([As, self.box_As], dim=0)
                    bs = torch.cat([bs, self.box_bs], dim=0)
                    
                    # save polytope
                    # polytope = (As, bs)

                    # Concatenate the bounding box constraints with the point cloud polyhedron
                    # As = torch.cat([As, self.box_As], dim=0)
                    # bs_shrunk = torch.cat([bs_shrunk, self.box_bs], dim=0)

                    norm_A = torch.linalg.norm(As, dim=-1, keepdims=True)
                    As = As / norm_A
                    bs = bs / norm_A.squeeze()

                    # By manageability, the midpoint should always be clearly within the polytope
                    # NOTE: Let's hope there are no errors here.
                    #A, b, qhull_pts = h_rep_minimal(As.cpu().numpy(), bs.cpu().numpy(), midpoint.cpu().numpy())

                    polytope = (torch.tensor(As, device=self.device), torch.tensor(bs, device=self.device))

                    torch.cuda.synchronize()
                    times_polytope += time.time() - tnow

            else:
                # Test if the line segment is within the polytope
                # If the segment is within the polytope, we proceed to next segment
                if compute_segment_in_polytope(polytope[0], polytope[1], segment):

                    continue

                else:
                    # Part 2: Computes the collision set
                    tnow = time.time()
                    output = self.collision_set.compute_set_one_step(segment)
                    torch.cuda.synchronize()
                    point_cloud = output['means']
                    self.box_As = output['A_bb']
                    self.box_bs = output['b_bb_shrunk']
                    times_collision_set += time.time() - tnow

                    if len(output['gaussian_ids']) == 0:
                        polytope = (self.box_As, self.box_bs)
                    
                    else:

                        # Part 3: Compute the ellipsoid
                        tnow = time.time()
                        ellipsoid, d, p_stars = self.find_ellipsoid(segment, point_cloud)
                        torch.cuda.synchronize()
                        times_ellipsoid += time.time() - tnow

                        # Part 4: Compute the polytope
                        tnow = time.time()
                        As, bs, ps_star = self.find_polyhedron(point_cloud, d, ellipsoid)
                        
                        # Part 5: Shrink the polytope
                        bs = bs - self.radius * torch.norm(As, dim=-1)

                        As = torch.cat([As, self.box_As], dim=0)
                        bs = torch.cat([bs, self.box_bs], dim=0)

                        # save polytope
                        # polytope = (As, bs)

                        # Concatenate the bounding box constraints with the point cloud polyhedron
                        # As = torch.cat([As, self.box_As], dim=0)
                        # bs_shrunk = torch.cat([bs_shrunk, self.box_bs], dim=0)

                        norm_A = torch.linalg.norm(As, dim=-1, keepdims=True)
                        As = As / norm_A
                        bs = bs / norm_A.squeeze()

                        # By manageability, the midpoint should always be clearly within the polytope
                        # NOTE: Let's hope there are no errors here.
                        #A, b, qhull_pts = h_rep_minimal(As.cpu().numpy(), bs.cpu().numpy(), midpoint.cpu().numpy())
        
                        polytope = (torch.tensor(As, device=self.device), torch.tensor(bs, device=self.device))

                        torch.cuda.synchronize()
                        times_polytope += time.time() - tnow
                
            polytopes.append(polytope)

        # Step 6: Perform Bezier spline optimization
        tnow = time.time()
        traj, feasible = self.spline_planner.optimize_b_spline(polytopes, x0, xf)
        if not feasible:
            traj = torch.stack([x0, xf], dim=0)
        torch.cuda.synchronize()
        times_opt = time.time() - tnow

        # save outputs 
        traj_data = {
            'path': path.tolist(),
            'polytopes': [torch.cat([polytope[0], polytope[1].unsqueeze(-1)], dim=-1).tolist() for polytope in polytopes],
            'num_polytopes': len(polytopes),
            'traj': traj.tolist(),
            'times_astar': time_astar,
            'times_collision_set': times_collision_set,
            'times_ellipsoid': times_ellipsoid,
            'times_polytope': times_polytope,
            'times_opt': times_opt,
            'feasible': feasible
        }

        #self.save_polytope(polytopes, 'sfc_corridor.obj')
        return traj_data
    
    def save_polytope(self, polytopes, save_path):
        # Initialize mesh object
        mesh = o3d.geometry.TriangleMesh()

        for (A, b) in polytopes:
            # Transfer all tensors to numpy
            A = A.cpu().numpy()
            b = b.cpu().numpy()

            pt = find_interior(A, b)

            halfspaces = np.concatenate([A, -b[..., None]], axis=-1)
            hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)
            qhull_pts = hs.intersections

            pcd_object = o3d.geometry.PointCloud()
            pcd_object.points = o3d.utility.Vector3dVector(qhull_pts)
            bb_mesh, qhull_indices = pcd_object.compute_convex_hull()
            mesh += bb_mesh
        
        success = o3d.io.write_triangle_mesh(save_path, mesh, print_progress=True)

        return success





def rotation_matrix_from_vectors(vec1, vec2):
    # Reference: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    dim = vec1.shape[0]

    if dim == 2:
        # 2D rotation matrix
        theta = torch.atan2(vec2[1], vec2[0]) - torch.atan2(vec1[1], vec1[0])
        rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]], device=vec1.device)
        return rotation_matrix
    a, b = (vec1 / torch.norm(vec1)).reshape(3), (vec2 / torch.norm(vec2)).reshape(3)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    s = torch.norm(v)
    # if parallel vectors
    if s == 0:
        return torch.eye(3, device=vec1.device) if c > 0 else torch.eye(3, device=vec1.device) * -1  
    
    kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=vec1.device)
    rotation_matrix = torch.eye(3, device=vec1.device) + kmat + kmat.mm(kmat) * ((1 - c) / (s ** 2))
    
    return rotation_matrix


def find_closest_hyperplane(pivots, As, point, rad):
    # Tries to find the closest hyperplane to the current one that is anchored at the pivot
    # and is some R away from the point. 
    # In fact, it is equivalent to finding the hyperplane that is normal to the sphere of radius R
    # centered about the pivot point and passes through our desired point.
    # pivot: 3
    # As: M x 3

    # As is typically outward facing (facing away from the line segment). For this algorithm,
    # we've assumed it is inward facing. We will flip it back at the end.
    As = -As

    # We need to make sure that the polytope constraints are normalized. We also want it
    # to have length R.
    As = rad * As / torch.linalg.norm(As, dim=-1, keepdim=True)    # M x 3

    # Preliminary variables
    delta = point[None, :] - pivots    # M x 3
    delta_norm = torch.linalg.norm(delta, dim=-1)    # M
    a0 = As
    a0_delta = torch.sum( a0 * delta, dim=-1)    # M

    # Coefficients of the quadratic equation
    coeff1 = (delta_norm**2 - delta_norm**4 / (rad**2) ) / 4      # M
    coeff2 = - a0_delta * (1 - ( delta_norm / rad ) **2  ) 
    coeff3 = torch.sum( As**2, dim=-1 ) - (a0_delta / rad )**2

    # Solve the quadratic equation
    # first find if it is even solvable
    discriminant = coeff2**2 - 4 * coeff1 * coeff3

    # If the discriminant is negative, there are no solutions
    if torch.any(discriminant < 0):
        return None, None, None
    
    # Otherwise, we find the two solutions
    l1_1 = (-coeff2 + torch.sqrt(discriminant)) / (2 * coeff1)      # M
    l1_2 = (-coeff2 - torch.sqrt(discriminant)) / (2 * coeff1)      # M

    l2_1 = (l1_1 * delta_norm**2 / 2 - a0_delta) / rad**2 - 1
    l2_2 = (l1_2 * delta_norm**2 / 2 - a0_delta) / rad**2 - 1

    # We now compute the closest hyperplane candidates
    a_1 = ( As - l1_1 * delta / 2 ) / (1 + l2_1)        # M x 3
    a_2 = ( As - l1_2 * delta / 2 ) / (1 + l2_2)        # M x 3

    # Find the closest hyperplane by plugging back into objctive
    obj1 = torch.sum( (a_1 - As)**2, dim=-1)
    obj2 = torch.sum( (a_2 - As)**2, dim=-1)

    objs = torch.stack([obj1, obj2], dim=-1)    # M x 2

    new_pivot1 = a_1 + pivots
    new_pivot2 = a_2 + pivots

    new_pivots = torch.stack([new_pivot1, new_pivot2], dim=1)    # M x 2 x 3

    # Recall that we flipped the As to point inward, so we need to flip it back
    closest_A = -torch.stack([a_1, a_2], dim=-2) # M x 2 x 3
    closest_b = torch.sum(closest_A * new_pivots, dim=-1)    # M x 2

    return closest_A, closest_b, objs

