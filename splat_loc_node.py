# %%
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber, TimeSynchronizer
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
import sensor_msgs.msg as sensor_msgs
from scipy.spatial.transform import Rotation
import json

import struct
from std_msgs.msg import Header, String
from sensor_msgs.msg import PointCloud2, PointField, Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
import cv2

import numpy as np
import open3d as o3d
from pathlib import Path
import torch
import time

from ns_utils.nerfstudio_utils import GaussianSplat
from pose_estimator.utils import SE3error, PrintOptions
from pose_estimator.pose_estimator import SplatLoc


TRIAL_IDX = 3


class SplatLocNode(Node):
    def __init__(
        self,
        config_path: Path,
        use_compressed_image: bool = True,
        drone_config=None,
        colmap2mocap=None,
        result_dir: Path = './results',
        timestart: str = None,
        enable_UKF: bool = False,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__(node_name="splatlocnode")
        # map name
        self.map_name = "camera_link"

        self.result_dir = result_dir / f"{timestart}_{TRIAL_IDX}"
            
        # create directory, if needed
        Path(self.result_dir).parent.mkdir(parents=True, exist_ok=True)
        
        # device
        self.device = device

        # slam mode
        self.slam_method = 'modal'

        # load the Gaussian Splat
        self.gsplat = GaussianSplat(config_path=config_path,
                                    dataset_mode="train",
                                    device=self.device)

        # applied to data
        self.T_data2gsplat = torch.eye(4).cuda()
        self.T_data2gsplat[
            :3, :
        ] = self.gsplat.pipeline.datamanager.train_dataparser_outputs.dataparser_transform
        self.s_data2gsplat = (
            self.gsplat.pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        )
        # applied to gsplat poses
        self.T_gsplat2data = torch.linalg.inv(self.T_data2gsplat)
        self.s_gsplat2data = 1 / self.s_data2gsplat

        # drone config parameters
        self.drone_config = drone_config

        # set the camera intrinsics
        self.update_camera_intrinsics()

        # distortion
        self.dparams = torch.tensor([drone_config["k1"], drone_config["k2"], 
                                     drone_config["k3"], drone_config["p1"], 
                                     drone_config["p2"]]).float().to(self.device)
        
        # option to undistort
        self.undistort_images = True

        # blur
        self.blur_threshold = 1.0

        # self.T_handeye = torch.eye(4).cuda()

        if drone_config is not None:
            # camera-to-body-frame
            # self.T_handeye = torch.linalg.inv(torch.from_numpy(np.array(drone_config["pose_to_camera"]))).to(self.device).float()
            self.T_handeye = torch.from_numpy(np.array(drone_config["pose_to_camera"])).to(self.device).float()
            # self.T_handeye[:3, :3] = torch.from_numpy(np.array(handeye["R"]))
            # self.T_handeye[:3, 3] = torch.from_numpy(np.array(handeye["t"])).squeeze()

        # inverse of the handeye
        # body-frame-to-camera
        self.T_inv_handeye = torch.linalg.inv(self.T_handeye)

        # from COLMAP data to mocap
        # self.T_colmap2mocap = torch.from_numpy(np.array(colmap2mocap["transform_matrix"])).cuda().float()

        # from mocap to COLMAP data
        # self.T_mocap2colmap = torch.linalg.inv(self.T_colmap2mocap)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Splat-Loc Node
        self.splat_loc = SplatLoc(self.gsplat)

        # setting to enable UKF
        self.enable_UKF = enable_UKF

        if self.enable_UKF:
            # set the parameters: kappa and dt for the UKF
            self.splat_loc.set_kappa(kappa=2.0)
            self.splat_loc.set_dt(dt=1e-3)

        # computation time
        self.computation_time = []

        # pose error
        self.pose_error = []

        # success rate
        self.cache_success_flag = []

        if self.enable_UKF:
            # computation time
            self.computation_time_ukf = []

            # pose error
            self.pose_error_ukf = []

            # success rate
            self.cache_success_flag_ukf = []

        # OpenCV brideTimeSynchronizer
        self.opencv_bridge = CvBridge()

        # QOS Profile
        qos_profile_incoming = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # self.cam_info_sub = self.create_subscription(
        #     CameraInfo,
        #     "/camera/color/camera_info",
        #     self.cam_info_callback,
        #     qos_profile=qos_profile_incoming,
        # )

        # subscriber to the ground-truth pose
        self.gt_pose_subscription = Subscriber(
            self, PoseStamped, "/republished_pose"
            , qos_profile=qos_profile_incoming
        )


        # self.odometry_subscriber = self.create_subscription(
        #     VehicleOdometry,
        #     '/fmu/out/vehicle_odometry',
        #     self.odometry_callback,
        #     qos_profile
        # )

        # option to use compressed images
        self.use_compressed_image = use_compressed_image
        self.print_options = PrintOptions()

        # sensor message type (for images)
        sensor_img_type = CompressedImage if self.use_compressed_image else Image

        # subscriber to the RGB Image topic
        self.cam_rgb_subscription = Subscriber(
            self,
            sensor_img_type,
            "/republished_image",
            qos_profile=qos_profile_incoming,
        )

        self.ts = ApproximateTimeSynchronizer(
            [self.gt_pose_subscription, self.cam_rgb_subscription], 40, 0.1
        )
        self.ts.registerCallback(self.ts_callback)

        # topic_sync = "exact"

        # if topic_sync == "approx":
        #     self.ts = ApproximateTimeSynchronizer(self.subs, 40, topic_slop)
        # elif topic_sync == "exact":
        #     self.ts = TimeSynchronizer(self.subs, 40)
        # else:
        #     raise NameError("Unsupported topic sync method. Must be {approx, exact}.")

        # self.ts.registerCallback(self.ts_callback)

        # self.update_RGB_image,
        # self.pose_callback,

        # publisher for the current pose estimate
        self.est_pose_publisher = self.create_publisher(
            PoseStamped, "/estimated_pose", 10
        )

        self.state_publisher = self.create_publisher(String, "state", 10)

        # current RGB image
        self.cam_rgb = None
        # self.cam_K = None

        # initial guess of the pose
        self.init_guess = None

        # self.replan_timer = self.create_timer(5., self.replan)

        # debug mode
        self.debug_mode = True

    def ts_callback(self, msg_pose, msg_rgb):
        if self.init_guess is None:
            self.update_init_guess(msg_pose)

        self.gt_pose = torch.eye(4, device=self.device)
        # translation
        self.gt_pose[:3, -1] = torch.tensor(
            [
                msg_pose.pose.position.x,
                msg_pose.pose.position.y,
                msg_pose.pose.position.z,
            ]
        )
        # orientation
        R_msg = Rotation.from_quat(
            np.array(
                [
                    msg_pose.pose.orientation.x,
                    msg_pose.pose.orientation.y,
                    msg_pose.pose.orientation.z,
                    msg_pose.pose.orientation.w,
                ]
            )
        ).as_matrix()
        self.gt_pose[:3, :3] = torch.from_numpy(R_msg).to(self.device)

        self.update_RGB_image(msg_rgb)

    # def cam_info_callback(self, msg):
    #     if self.cam_K is None:
    #         self.cam_K = msg.k
    #         self.cam_K = torch.from_numpy(self.cam_K.reshape((3, 3)))

    def update_camera_intrinsics(self):
        # focal lengths and optical centers
        fx = self.drone_config["fx"]
        fy = self.drone_config["fy"]
        cx = self.drone_config["cx"]
        cy = self.drone_config["cy"]
        
        # cmaera intrinsics matrix
        self.cam_K = np.array([[fx, 0,  cx],
                               [0,  fy, cy],
                               [0,  0,  1]])
        self.cam_K = torch.from_numpy(self.cam_K).to(self.device).float()

    def update_RGB_image(self, msg):
        # update the current image of the camera
        if self.use_compressed_image:
            # image in OpenCV format
            img_cv = self.opencv_bridge.compressed_imgmsg_to_cv2(msg)
            img_tensor = torch.from_numpy(img_cv).to(dtype=torch.float32) / 255.0

            # convert from BGR to RGB
            img_tensor = img_tensor[..., [2, 1, 0]]
        else:
            im_cv = self.opencv_bridge.imgmsg_to_cv2(msg, msg.encoding)
            if self.slam_method == "modal":
                im_cv = cv2.cvtColor(im_cv, cv2.COLOR_YUV2RGB_Y422)
            else:
                im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)

            # img_cv = self.opencv_bridge.imgmsg_to_cv2(msg, msg.encoding)
            # img_tensor = torch.from_numpy(im_cv).to(dtype=torch.float32) / 255.0


        if self.undistort_images:
            im_cv = cv2.undistort(im_cv, self.cam_K.cpu().numpy(), self.dparams.cpu().numpy(), None, None)

        # Filter motion blur
        if self.blur_threshold > 0:
            laplacian = cv2.Laplacian(im_cv, cv2.CV_64F).var()
            if laplacian < self.blur_threshold:
                return False, None

        img_tensor = torch.from_numpy(im_cv).to(dtype=torch.float32) / 255.0

        # cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("frame", im_cv)
        # import matplotlib.pyplot as plt
        # img = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # plt.figure()
        # plt.imshow(img)
        # plt.show()

        self.cam_rgb = img_tensor

        # breakpoint()
        # estimate the pose
        self.estimate_pose()

    def pose_callback(self, msg):
        # set the initial guess
        if self.init_guess is None:
            self.update_init_guess(msg)

    def update_init_guess(self, msg):
        # This is MOCAP z-up frame
        if self.debug_mode:
            print("Updating the Initial Guess!")

        init_guess = torch.eye(4, device=self.device)

        # translation
        init_guess[:3, -1] = torch.tensor(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )

        # orientation
        # init_guess[:3, :3] = quaternion_to_rotation_matrix(torch.tensor([msg.pose.orientation.x,
        #                                                                       msg.pose.orientation.y,
        #                                                                       msg.pose.orientation.z,
        #                                                                       msg.pose.orientation.w])[None])
        R_msg = Rotation.from_quat(
            np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ]
            )
        ).as_matrix()
        init_guess[:3, :3] = torch.from_numpy(R_msg).to(self.device)
        init_guess = init_guess @ self.T_handeye

        # init_guess = init_guess[:, [1,2,0,3]]
        init_guess[:, [1, 2]] *= -1

        # transform from Mocap to COLMAP
        # init_guess = self.T_mocap2colmap @ init_guess
        # print(self.T_data2gsplat)

        init_guess = self.T_data2gsplat @ init_guess
        init_guess[:3, 3] *= self.s_data2gsplat

        # breakpoint()

        self.init_guess = init_guess
        # init_guess = self.gsplat.get_poses()[0]
        # self.init_guess = torch.eye(4).cuda()
        # self.init_guess[:3, :] = init_guess

        if self.enable_UKF:
            # Set the prior distribution of the UKF
            self.splat_loc.set_prior_distribution(mu=init_guess,
                                                sigma=torch.eye(6, device=self.device))


    def estimate_pose(self):
        # estimate the pose
        if (
            (self.cam_rgb is not None)
            and (self.init_guess is not None)
            and (self.cam_K is not None)
        ):
            print(self.print_options.sep_0)
            print(self.print_options.sep_1)
            print(f"Running Pose Estimator...")
            print(self.print_options.sep_1)

            if self.enable_UKF:
                # estimate the pose (UKF)
                # try:
                    # computation time
                    start_time = time.perf_counter()

                    est_pose_ukf = self.splat_loc.estimate_ukf(
                        init_guess=None,
                        cam_rgb=self.cam_rgb.to(self.device),
                        cam_K=self.cam_K,
                    )

                    # total computation time (for this iteration)
                    self.computation_time_ukf.append(time.perf_counter() - start_time)

                    # cache the success flag
                    self.cache_success_flag_ukf.append(1)
                # except:
                #     print("Failed UKF!")
                #     # cache the success flag
                #     self.cache_success_flag_ukf.append(0)
            
            # estimate the pose (PnP-RANSAC only)
            try:
                # computation time
                start_time = time.perf_counter()

                est_pose = self.splat_loc.estimate(
                    init_guess=self.init_guess,
                    cam_rgb=self.cam_rgb.to(self.device),
                    cam_K=self.cam_K,
                )

                # total computation time (for this iteration)
                self.computation_time.append(time.perf_counter() - start_time)

                # cache the success flag
                self.cache_success_flag.append(1)
            except Exception as excp:
                print("Failed PnP!")
                # cache the success flag
                self.cache_success_flag.append(0)
                return

            # update the initial guess
            self.init_guess = est_pose
            pub_pose = est_pose.clone()
            pub_pose[:3, 3] *= self.s_gsplat2data
            pub_pose = self.T_gsplat2data @ pub_pose

            # transform from COLMAP to Mocap
            # pub_pose = self.T_colmap2mocap @ pub_pose

            # convert from opengl frame ros
            pub_pose[:, [1, 2]] *= -1
            pub_pose = pub_pose @ self.T_inv_handeye
            # pub_pose = pub_pose[:, [2,0,1,3]]

            if self.debug_mode:
                # pose error
                error = SE3error(self.gt_pose.cpu().numpy(), pub_pose.cpu().numpy())

                print(
                    f"{self.print_options.sep_space} PnP-RANSAC --- SE(3) Estimation Error -- Rotation: {error[0]}, Translation: {error[1]}"
                )

                # store the pose error
                self.pose_error.append(error)

            if self.enable_UKF:
                if self.debug_mode:
                    ukf_pose = est_pose_ukf.clone()
                    ukf_pose[:3, 3] *= self.s_gsplat2data
                    ukf_pose = self.T_gsplat2data @ ukf_pose

                    # convert from opengl frame ros
                    ukf_pose[:, [1, 2]] *= -1
                    ukf_pose = ukf_pose @ self.T_inv_handeye
                    # pub_pose = pub_pose[:, [2,0,1,3]]

                    # pose error
                    error = SE3error(self.gt_pose.cpu().numpy(), ukf_pose.cpu().numpy())

                    print(
                        f"{self.print_options.sep_space} UKF SE(3) --- Estimation Error -- Rotation: {error[0]}, Translation: {error[1]}"
                    )

                    # store the pose error
                    self.pose_error.append(error)

            # Publish the estimated pose
            msg = PoseStamped()
            # msg.header.frame_id = self.map_name

            # translation
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (
                pub_pose[0, -1].item(),
                pub_pose[1, -1].item(),
                pub_pose[2, -1].item(),
            )

            # orientation
            (
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ) = Rotation.from_matrix(pub_pose[:3, :3].cpu().numpy()).as_quat()

            # header
            msg.header.frame_id = self.map_name

            self.est_pose_publisher.publish(msg)

            print(self.print_options.sep_0)
            print(self.print_options.sep_1)
            print(f"Finished Estimating the Pose...")
            print(self.print_options.sep_1)

    def save_results(self):
        # save the results

        # computation time
        save_dir = self.result_dir / "estimator"
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            f"{save_dir}/computation_time.npy",
            self.computation_time,
        )

        # success rate
        np.save(
            f"{save_dir}/success_flag.npy",
            self.cache_success_flag,
        )

        # pose error
        np.save(
            f"{save_dir}/pose_error.npy", self.pose_error
        )

        if self.enable_UKF:
            # computation time
            save_dir = self.result_dir / "estimator"
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                f"{save_dir}/computation_time_ukf.npy",
                self.computation_time,
            )

            # success rate
            np.save(
                f"{save_dir}/success_flag_ukf.npy",
                self.cache_success_flag,
            )

            # pose error
            np.save(
                f"{save_dir}/pose_error_ukf.npy", self.pose_error
            )


if __name__ == "__main__":
    # init
    rclpy.init()
    timestart = time.strftime("%Y%m%d-%H%M")

    # config path
    # config_path = Path(f"./data/Flightroom hard/outputs/flightroom_colmap/gemsplat/2024-04-16_002525/config.yml")
    config_path = Path(
        "outputs/configs/ros-depth-splatfacto/2024-10-30_140655/config.yml"
    )

    # drone info
    drone_config_path = "configs/modal.json"

    if drone_config_path is not None:
        with open(drone_config_path, "r") as f:
            drone_config = json.load(f)

    # # Handeye Calibration
    # handeye_transforms_path = "configs/modal.json" # "transforms/handeye.json"

    # if handeye_transforms_path is not None:
    #     with open(handeye_transforms_path, "r") as f:
    #         handeye_dict = json.load(f)

    # COLMAP to Mocap
    colmap2mocap_transforms_path = None # "transforms/colmap2mocap.json"

    if colmap2mocap_transforms_path is not None:
        with open(colmap2mocap_transforms_path, "r") as f:
            colmap2mocap_dict = json.load(f)
    else:
        colmap2mocap_dict = None

    # result directory
    goal_queries = ["red cup", "water gallon", "keyboard", "teddy bear"]
    
    # index for the goal query
    goal_idx = 0

    # result directory
    result_dir = Path(f"results/{goal_queries[goal_idx]}")

    # enable UKF
    enable_UKF = False # True
    
    # Pose Estimator
    splatnavnode = SplatLocNode(
        config_path=config_path,
        use_compressed_image=False,
        drone_config=drone_config,
        colmap2mocap=colmap2mocap_dict,
        result_dir=result_dir,
        timestart=timestart,
        enable_UKF=enable_UKF
    )

    # Run
    try:
        rclpy.spin(splatnavnode)
    except KeyboardInterrupt:
        pass

    # save the results
    splatnavnode.save_results()

    # Clean-up
    splatnavnode.destroy_node()

    # shutdown
    # rclpy.shutdown()
