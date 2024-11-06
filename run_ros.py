#%%
import os
import torch
from pathlib import Path    
import time
import numpy as np
from splat.splat_utils import GSplatLoader
from splatplan.splatplan import SplatPlan
from splatplan.spline_utils import SplinePlanner

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header, String
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
#from px4_msgs.msg import VehicleOdometry  # Adjusted to use the PX4-specific odometry message
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.duration import Duration
import time
import json

import threading
import sys
import select
import tty
import termios

from ros_utils import make_point_cloud

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ControlNode(Node):

    def __init__(self, mode='open-loop'):
        super().__init__('control_node')
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_c = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_incoming = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        # Set the map name for visualization in RVIZ
        self.map_name = "camera_link"

        self.mode = mode

        # Subscribe to the estimated pose topic. In open-loop experiments, this is not used. In the closed-loop experiments,
        # this can be from the VIO or from Splat-Loc.
        # Subscribe to the odometry topic
        # self.odometry_subscriber = self.create_subscription(
        #     VehicleOdometry,
        #     '/fmu/out/vehicle_odometry',
        #     self.odometry_callback,
        #     qos_profile
        # )
        
        # Publish to the control topic
        self.control_publisher = self.create_publisher(
            Float32MultiArray,
            '/control',
            qos_profile_c
        )

        # Publishes the static voxel grid as a point cloud
        self.pcd_publisher = self.create_publisher(
            PointCloud2, "/gsplat_pcd", 10
        )
        self.timer = self.create_timer(1.0, self.pcd_callback)
        self.pcd_msg = None

        # The state of SplatPlan. This is used to trigger replanning. 
        self.state_publisher = self.create_publisher(String, "/splatplan_state", 10)

        # This publishes the goal pose
        self.goal_publisher = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.goal_timer = self.create_timer(1.0, self.goal_callback)

        # self.position_subscriber = self.create_subscription(
        #     PoseStamped,
        #     '/republished_pose',
        #     self.odometry_callback,
        #     10)

        # When rebpulished pose is updated, we update the trajectory too! ONLY WORKS FOR SLOW REBPUB RATES
        self.position_subscriber = self.create_subscription(
            PoseStamped,
            '/republished_pose',
            self.trajectory_callback,
            10)
        
        # if mode == 'closed-loop':
        #     # This is the timer that triggers replanning
        #     self.replan_timer = self.create_timer(1.0, self.replan)

        ### Initialize variables  ###
        self.fmu_pos = [0.0, 0.0, 0.0]
        
        self.velocity_output = [0.0, 0.0, 0.0]
        self.position_output = [0.0, 0.0, -0.75]
        self.current_position = [0.0, 0.0, -0.75]
        self.acceleration_output = [0.0, 0.0, 0.0]


        idx = 3
        self.goals = [ [2.5, 3., -0.75],
                      [0., 4., -0.75],
                      [-2., 4., -0.9],
                      [-3.5, 3.0, -0.75]
        ]

        self.goal = self.goals[idx]


        self.des_yaw_rate = 0.0
        self.yaw = (90.0) * 3.14/ 180.0
        self.outgoing_waypoint = [0.0, 0., -0.75, 0., 0., 0., 0., 0., 0., 0., 0., 0., self.yaw]

        self.timer = self.create_timer(1.0 / 10.0, self.publish_control)

        self.start_mission = False

        # Start the keyboard listener thread
        self.keyboard_thread = threading.Thread(target=self.key_listener)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

        ### SPLATPLAN INITIALIZATION ###
        ############# Specify scene specific data here  #############
        # Points to the config file for the GSplat
        path_to_gsplat = Path('outputs/configs/ros-depth-splatfacto/2024-10-30_140655/config.yml')

        radius = 0.2       # radius of robot
        amax = 1.
        vmax = 1.

        self.distance_between_points = 0.25

        lower_bound = torch.tensor([-4.5, -3., -1.], device=device)
        upper_bound = torch.tensor([3., 4.5, 0.], device=device)
        resolution = 50

        #################
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
        self.gsplat = GSplatLoader(path_to_gsplat, device)
        print('Time to load GSplat:', time.time() - tnow)

        print(f'There are {len(self.gsplat.means)} Gaussians in the Splat.')

        spline_planner = SplinePlanner(spline_deg=6, device=device)
        self.planner = SplatPlan(self.gsplat, robot_config, voxel_config, spline_planner, device)
        # self.traj = self.plan_path(self.position_output, self.goal)

        # Publishes the trajectory as a Pose Array
        self.trajectory_publisher = self.create_publisher(PoseArray, "/trajectory", 10)
        # self.trajectory_timer = self.create_timer(1.0, self.trajectory_callback)

        self.traj = None
        self.do_replan = True

        self.outputs = []
        self.data_savepath = f'traj/data_{self.mode}_{idx}.json'

        print("SplatPlan Initialized...")

    def trajectory_callback(self, msg):
        if msg is not None:
            self.current_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        
        if not self.do_replan:
            self.do_replan = True

            return

        if self.traj is None or self.mode == 'closed-loop':
            try:
                traj, output = self.plan_path(self.current_position, self.goal)

                # NOTE: !!! self.traj is the np array of traj( a list)
                self.traj = np.array(traj)

                yaw = 0.0
                yaws = []
                for idx, pt in enumerate(self.traj):
                    if idx < len(self.traj) - 1:

                        diff = self.traj[idx + 1] - self.traj[idx]

                        prev_yaw = yaw
                        yaw = np.arctan2(diff[1], diff[0])

                        closest_k = np.round(-(yaw - prev_yaw) / (2*np.pi))
                        yaw = yaw + 2*np.pi*closest_k     

                    yaws.append(yaw)

                yaws = np.stack(yaws)

                self.traj = np.concatenate([self.traj, yaws.reshape(-1, 1)], axis=-1)

                output['traj'] = self.traj.tolist()

                self.outputs.append(output)

                # IF we are close to goal
                if np.linalg.norm(self.current_position - self.goal) < 0.1:
                    print('Reached Goal!')

            except:
                print('Error with SplatPlan.')

        poses = []
        for idx, pt in enumerate(self.traj):
            msg = Pose()
            # msg.header.frame_id = self.map_name
            msg.position.x, msg.position.y, msg.position.z = pt[0], pt[1], pt[2]
            yaw = pt[-1]
            quat = Rotation.from_euler("z", yaw).as_quat()
            (
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ) = (quat[0], quat[1], quat[2], quat[3])

            poses.append(msg)

        print('Traj Start:', self.traj[0])
        print('Traj End:', self.traj[-1])

        msg = PoseArray()
        msg.header.frame_id = self.map_name
        msg.poses = poses

        self.trajectory_publisher.publish(msg)

        self.do_replan = False


        # waypoint [pos, vel, accel, jerk]
                    # Find the closest position to the current position
        distance = np.linalg.norm( self.traj[:, :3] - np.array(self.current_position)[None] , axis=-1 ).squeeze()
                    
        # Find all distances within a ball to the drone
        within_ball = distance <= self.distance_between_points

        # If no points exist within the ball, choose the closest point
        if not np.any(within_ball):
            min_index = np.argmin(distance)
            # self.outgoing_waypoint = ( self.traj[min_index] + self.outgoing_waypoint ) / 2
            self.traj = self.traj[min_index:]
        
        #else
        else:
            # find the point that makes the most progress
            indices = np.arange(len(distance))[within_ball]
            max_index = np.max(indices)
            # self.outgoing_waypoint = ( self.traj[max_index] + self.outgoing_waypoint ) / 2
            self.traj = self.traj[max_index:]                   

        return

    def goal_callback(self):
        msg = PoseStamped()
        msg.header.frame_id = self.map_name

        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = self.goal[0], self.goal[1], self.goal[2]
        yaw = 0.
        quat = Rotation.from_euler("z", yaw).as_quat()
        (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ) = (quat[0], quat[1], quat[2], quat[3])

        self.goal_publisher.publish(msg)

    def pcd_callback(self):
        if self.pcd_msg is None:
            points = self.gsplat.means.cpu().numpy()
            colors = (255 * torch.clip(self.gsplat.colors, 0., 1.).cpu().numpy()).astype(np.uint32)

            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgba", offset=12, datatype=PointField.UINT32, count=1),
            ]

            self.pcd_msg = make_point_cloud(points, colors, self.map_name, fields)

        self.pcd_publisher.publish(self.pcd_msg)

    def odometry_callback(self, msg):
        # # Extract velocity and position in the x and y directions (assuming NED frame)
        # if msg.velocity_frame == VehicleOdometry.VELOCITY_FRAME_NED:
        #     #self.current_velocity[0] = -msg.velocity[0]  # Velocity in x direction
        #     #self.current_velocity[1] = -msg.velocity[1]  # Velocity in y direction
        #     # self.current_position[0] = msg.position[0]  # Position in x direction
        #     # self.current_position[1] = msg.position[1]  # Position in y direction
        #     self.fmu_pos[0] = msg.position[0]
        #     self.fmu_pos[1] = msg.position[1]
        #     self.fmu_vel[0] = msg.velocity[0]
        #     self.fmu_vel[1] = msg.velocity[1]
        #     # print("we are getting odom")
        #     pass

        if msg is not None:
            self.current_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

        # TODO: Set the yaw?
        # yaw = 0.
        # quat = Rotation.from_euler("z", yaw).as_quat()
        # (
        #     msg.pose.orientation.x,
        #     msg.pose.orientation.y,
        #     msg.pose.orientation.z,
        #     msg.pose.orientation.w,
        # ) = (quat[0], quat[1], quat[2], quat[3])

    def key_listener(self):
        print("Press the space bar to start the mission.")
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.start_mission:
                dr, dw, de = select.select([sys.stdin], [], [], 0)
                if dr:
                    c = sys.stdin.read(1)
                    if c == ' ':
                        self.start_mission = True
                        print("Space bar pressed. Starting trajectory.")
                        break
        except Exception as e:
            print(e)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def publish_control(self):
        # current_time = self.get_clock().now().to_msg()
        # current_time_f = current_time.sec + current_time.nanosec * 1e-9
        # print(current_time_f)
        dt = 1.0 / 30.0  # Assuming the timer runs at 30 Hz
        control_msg = Float32MultiArray()

        # pop from the first element of the trajectory 
        if (self.start_mission) and self.traj is not None:

            if self.traj.shape[0] > 0:

                # If open loop, pop the trajectory
                #if self.mode == 'open-loop' or not self.do_replan:
                self.outgoing_waypoint = (self.traj[0] + self.outgoing_waypoint) / 2
                self.traj = self.traj[1:]

                # else:
                #     # waypoint [pos, vel, accel, jerk]
                #     # Find the closest position to the current position
                #     distance = np.linalg.norm( self.traj[:, :3] - np.array(self.current_position)[None] , axis=-1 ).squeeze()
                    
                #     # Find all distances within a ball to the drone
                #     within_ball = distance <= self.distance_between_points

                #     # If no points exist within the ball, choose the closest point
                #     if not np.any(within_ball):
                #         min_index = np.argmin(distance)
                #         self.outgoing_waypoint = ( self.traj[min_index] + self.outgoing_waypoint ) / 2
                    
                #     #else
                #     else:
                #         # find the point that makes the most progress
                #         indices = np.arange(len(distance))[within_ball]
                #         max_index = np.max(indices)
                #         self.outgoing_waypoint = ( self.traj[max_index] + self.outgoing_waypoint ) / 2                        
                    
                self.position_output = [self.outgoing_waypoint[0], self.outgoing_waypoint[1], self.outgoing_waypoint[2]]
                self.velocity_output = [self.outgoing_waypoint[3], self.outgoing_waypoint[4], self.outgoing_waypoint[5]]
                acceleration_output = [self.outgoing_waypoint[6], self.outgoing_waypoint[7], self.outgoing_waypoint[8]]        # We set this to 0 for now
                self.jerk = [self.outgoing_waypoint[9], self.outgoing_waypoint[10], self.outgoing_waypoint[11]]
                self.yaw_output = self.outgoing_waypoint[-1]

            else:
                print("Trajectory complete.")
        else:
            self.yaw_output = self.yaw

        control_msg.data = [
            self.acceleration_output[0], self.acceleration_output[1], self.acceleration_output[2],
            self.velocity_output[0], self.velocity_output[1], self.velocity_output[2],
            self.position_output[0], self.position_output[1], self.position_output[2], self.yaw_output
        ]
        # control_msg.data = [
        #     self.acceleration_output[0], self.acceleration_output[1], self.acceleration_output[2],
        #     self.velocity_output[0], self.velocity_output[1], self.velocity_output[2],
        #     self.position_output[0], self.position_output[1], self.position_output[2], 0.
        # ]

        self.control_publisher.publish(control_msg)
        self.publish_control_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9  
        # print("control message: ", control_msg.data)

    def plan_path(self, start, goal):
        start = torch.tensor(start).to(device).to(torch.float32)
        goal = torch.tensor(goal).to(device).to(torch.float32)
        output = self.planner.generate_path(start, goal)

        return output['traj'], output
    
    def save_data(self):
        out_data = {
            'total_data': self.outputs
        }

        with open(self.data_savepath, 'w') as f:
            json.dump(out_data, f, indent=4)

        print('Saved data!')

def main(args=None):
    rclpy.init(args=args)
    control_node = ControlNode(mode='open-loop')
    
    try:
        rclpy.spin(control_node)
    except KeyboardInterrupt:
        pass
    finally:
        control_node.save_data()
        control_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
