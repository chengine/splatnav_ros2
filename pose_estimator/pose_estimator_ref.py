from pose_estimator.utils import *
from nerfstudio.models.base_model import Model as EnvModel
from lightglue import rbd

from colorama import Fore, Style
from traceback import print_exc

class Pose_Estimator:
    '''
    Pose Estimator Class for online pose estimation in Gaussian Splatting Environments.
    '''
    def __init__(self,
                 camera_K: torch.Tensor,
                 device: torch.device = 'cuda:0',
                 compute_pose_error: bool = False,
                 feature_detector: POI_Detector = POI_Detector.LIGHTGLUE,
                 visualization_options: Dict = {
                     'enable_registration_visualization': False,
                     'enable_downsampled_visualization': True,
                     'downsampled_voxel_size': 0.01,
                     'visualize_RGBD_point_clouds': False,
                     'render_RGB_at_estimated_pose': False,
                     'visualize_kp_and_matches': False,
                     'filepath_figures': 'figures',
                     'print_options': PrintOptions(),
                     'print_stats': True,
                     },
                 detector_parameters: Dict = {
                    'extractor': LIGHTGLUE_Extractor.SUPERPOINT,
                    'max_num_keypoints': 2048
                    },
                 ):
        '''
        Initializes the instance of the class.
        Arguments:
            camera_K: torch.Tensor - camera intrinsics matrix
            device: torch.device - device for tensors,
            compute_pose_error: bool - option to compute the pose error,
            feature_detector: POI_Detector - feature detector,
            visualization_options: Dict with items:
                enable_registration_visualization: bool - option to visualize registration results,
                enable_downsampled_visualization: bool - option to downsample the point cloud for visualization,
                downsampled_voxel_size: float - voxel size for downsampling the point cloud for visualization,
                visualize_RGBD_point_clouds: bool - option to visualize point clouds,
                render_RGB_at_estimated_pose: bool - option to render the RGB image at the estimated pose,
                visualize_kp_and_matches: bool - option to visualize the keypoints and matches,
                filepath_figures: Path or str - filepath to save the figures,
                print_options: PrintOptions - options for printing to the CONSOLE,
                print_stats: bool - option to print the error stats.
        '''
        # camera intrinsics
        self.camera_K = camera_K
        
        # device
        self.device = device
        
        # option to compute the pose error
        self.compute_pose_error = compute_pose_error

        # pose error for global registration
        self.pose_error_global_reg = []

        # computation time for global registration
        self.comp_time_global_reg = []
        
        # pose error for recursive pose estimation
        self.pose_error = []

        # computation time for recursive pose estimation
        self.comp_time = []
        
        # visualization options
        self.visualization_options = visualization_options
        
        # set up the feature dectector and matcher
        self.feature_detector = feature_detector
        
        if feature_detector == POI_Detector.LIGHTGLUE:
            # setup LightGlue
            extractor, matcher = setup_lightglue(feature_extractor=detector_parameters['extractor'],
                                                 max_num_keypoints=detector_parameters['max_num_keypoints'])
            
            # detector parameters
            self.feature_extractor = extractor
            self.feature_matcher = matcher
            
        # initialize timestep
        self.timestep = 0
                        
    def initialize(self,
                   env_pcd: o3d.geometry.PointCloud,
                   cam_pcd: o3d.geometry.PointCloud,
                   voxel_size: float,
                   perturb_init_guess: bool = False,
                   perturb_init_guess_params: Dict = {},
                   global_registration_params: Dict = {},
                   ground_truth_pose: np.ndarray = None
                   ):
        '''
        Computes an initial guess of the camera's pose.
        Arguments:
            env_pcd: o3d.geometry.PointCloud - point cloud of the environment,
            cam_pcd: o3d.geometry.PointCloud - point cloud obtained from the camera,
            voxel_size: float - voxel size for preprocessing point clouds,
            perturb_init_guess: bool - option to add noise to the initial guess of the pose,
            perturb_init_guess_params: Dict - parameters with which to perturb the initial guess,
            ground_truth_pose: np.ndarray - ground-truth pose, only needed for evaulation purposes.
        '''
        # compute an initial guess of the pose
        init_guess, global_est_transformation, global_time_taken, target_down = self._compute_intial_guess(env_pcd=env_pcd,
                                                                                                           cam_pcd=cam_pcd,
                                                                                                           voxel_size=voxel_size,
                                                                                                           perturb_init_guess=perturb_init_guess,
                                                                                                           perturb_init_guess_params=perturb_init_guess_params,
                                                                                                           global_registration_params=global_registration_params,
                                                                                                           visualization_options=self.visualization_options,
                                                                                                           device=self.device)
        
        # initial guess in OPENGL camera convention
        self.init_guess = init_guess
        
        # convert from OPENCV Camera convention to OPENGL convention
        self.init_guess[:, 1] = -init_guess[:, 1]
        self.init_guess[:, 2] = -init_guess[:, 2]
        
        self.init_guess = torch.tensor(self.init_guess, device=self.device).float()

        if self.compute_pose_error and ground_truth_pose is not None:
            # estimated pose
            # convert from OPENCV Camera convention to OPENGL convention
            global_est_pose = global_est_transformation.transformation.copy()
            global_est_pose[:, 1] = -global_est_pose[:, 1]
            global_est_pose[:, 2] = -global_est_pose[:, 2]

            # pose error from global registration
            self.pose_error_global_reg.append(SE3error(ground_truth_pose, global_est_pose))
            
            # pose error from initial guess
            self.pose_error_init_guess.append(SE3error(ground_truth_pose, self.init_guess))
                
            if self.visualization_options['print_stats']:
                print(self.visualization_options['print_options'].sep_0)
                print(f"Global Registration")
                print(self.visualization_options['print_options'].sep_0)
                print(f"SE(3) Estimation Error -- [Rotation (deg), Translation (m)]: {self.pose_error_global_reg[-1]}")
                print(self.visualization_options['print_options'].sep_1)
                
                print(self.visualization_options['print_options'].sep_0)
                print(f"Initial Guess")
                print(self.visualization_options['print_options'].sep_0)
                print(f"SE(3) Estimation Error -- [Rotation (deg), Translation (m)]: {self.pose_error_init_guess[-1]}")
                print(self.visualization_options['print_options'].sep_1)
            
        # computation time for global registration
        self.comp_time_global_reg.append(global_time_taken)

        # detector parameters
        detector_params = {}

        return self.init_guess
       
    def _compute_intial_guess(self,
                              env_pcd: o3d.geometry.PointCloud,
                              cam_pcd: o3d.geometry.PointCloud,
                              voxel_size: float,
                              perturb_init_guess: bool = False,
                              perturb_init_guess_params: Dict = {},
                              global_registration_params: Dict = {},
                              visualization_options: Dict = {
                                  'visualize_RGBD_point_clouds': False,
                                  'enable_registration_visualization': False,
                                  'enable_downsampled_visualization': True,
                                  'downsampled_voxel_size': 0.01,
                                  'print_options': PrintOptions(),
                                  'print_stats': True,
                                  },
                              device: torch.device = 'cuda:0'
                             ):
        '''
        Computes an initial guess of the camera's pose.
        Arguments:
            env_pcd: o3d.geometry.PointCloud - point cloud of the environment,
            cam_pcd: o3d.geometry.PointCloud - point cloud obtained from the camera,
            voxel_size: float - voxel size for preprocessing point clouds,
            perturb_init_guess: bool - option to add noise to the initial guess of the pose,
            perturb_init_guess_params: Dict - parameters with which to perturb the initial guess,
            visualization_options: Dict with items:
                enable_registration_visualization: bool - option to visualize registration results,
                enable_downsampled_visualization: bool - option to downsample the point cloud for visualization,
                downsampled_voxel_size: float - voxel size for downsampling the point cloud for visualization,
                print_options: PrintOptions - options for printing to the CONSOLE,
                print_stats: bool - option to print the error stats.
            device: torch.device - device for tensors.
        '''
        
        # preprocess source and target point clouds
        source_down, target_down, source_fpfh, target_fpfh = preprocess_point_clouds(source=cam_pcd,
                                                                                     target=env_pcd,
                                                                                     voxel_size=voxel_size,
                                                                                     downsample_pcd=True)            
        # #
        # # Global Registration
        # #
        
        if global_registration_params['method'] == Global_Registration.RANSAC:
            # start time
            t0 = time.perf_counter()

            # execute global registration
            global_est_transformation = execute_global_registration(source_down=source_down, target_down=target_down,
                                                                    source_fpfh=source_fpfh, target_fpfh=target_fpfh,
                                                                    voxel_size=voxel_size)

            # end time
            t1 = time.perf_counter()

            # computation time
            # eval_comp_time[eval_param['name']][0] = t1 - t0
            
            global_time_taken = t1 - t0

            print(visualization_options['print_options'].sep_0)
            print(f"RANSAC Global Registration")
            print(visualization_options['print_options'].sep_0)
        elif global_registration_params['method'] == Global_Registration.FGR:  
            
            # # #
            # # # FAST GLOBAL REGISTRATION
            # # #
            
            # FAST GLOBAL REGISTRATION

            # start time
            t0 = time.perf_counter()

            # execute global registration
            global_est_transformation = execute_fast_global_registration(source_down=source_down, target_down=target_down,
                                                                         source_fpfh=source_fpfh, target_fpfh=target_fpfh,
                                                                         voxel_size=voxel_size)

            # end time
            t1 = time.perf_counter()

            # computation time
            global_time_taken = t1 - t0

            print(visualization_options['print_options'].sep_0)
            print(f"Fast Global Registration")
            print(visualization_options['print_options'].sep_0)

        if visualization_options['print_stats']:
            print(f"Global Registration took {global_time_taken} seconds!")
        
        print(visualization_options['print_options'].sep_0)
        print(visualization_options['print_options'].sep_0)

        if visualization_options['enable_registration_visualization']:
            visualize_registration_result(source=cam_pcd, target=env_pcd, 
                                          transformation=global_est_transformation.transformation,
                                          enable_downsampled_visualization=visualization_options['enable_downsampled_visualization'],
                                          downsampled_voxel_size=visualization_options['downsampled_voxel_size'])

        # initial guess
        init_guess = global_est_transformation.transformation.copy()
            
        # add noise to the initial guess of the pose
        if perturb_init_guess:
            # generate a random rotation axis
            rand_rot_axis = torch.nn.functional.normalize(torch.rand(3, device=device), dim=-1)
            
            # random rotation matrix
            rand_rot = vec_to_rot_matrix(perturb_init_guess_params['rotation'] * rand_rot_axis)
            
            # initial guess
            init_guess[:3, :3] = rand_rot.cpu().numpy() @ init_guess[:3, :3]
            init_guess[:3, 3] += (perturb_init_guess_params['translation'] 
                                * torch.nn.functional.normalize(torch.rand(3, device=device), dim=-1).cpu().numpy())


        return init_guess, global_est_transformation, global_time_taken, target_down
    
    def execute_PnP_RANSAC(self, nerf: EnvModel, init_guess,
                           rgb_input,
                           camera_intrinsics_K,
                           feature_detector: POI_Detector,
                           save_image: bool=False,
                           pnp_figure_filename: Optional[str]='/',
                           print_stats: bool = False,
                           visualize_PnP_matches: bool = False,
                           pnp_matches_figure_filename: Optional[str]='/',
                           detector_params: Optional[Dict[str, Any]] = {}):
        '''
        Executes the Perspective-n-Point Algorithm.
        Arguments:
            nerf: EnvModel - radiance field representing the environment,
            init_guess: torch.Tensor - initial guess of the camera pose,
            camera_intrinsics_K: torch.Tensor - camera intrinsics matrix
            device: torch.device - device for tensors,
            compute_pose_error: bool - option to compute the pose error,
            feature_detector: POI_Detector - feature detector,
            save_image: bool - option to save the rendered image from the EnvModel,
            pnp_figure_filename: Optional[str] - filepath to save the rendered image from the EnvModel,
            print_stats: bool - option to print the error stats,
            visualize_PnP_matches: bool - option to visualize the PnP-Matches,
            pnp_matches_figure_filename: Optional[str] - filepath to save the PnP-Matches figure,
            detector_params: Optional[Dict[str, Any]] - optional parameters for the feature detector.
        '''
        # generate RGB-D point cloud
        target_rgb, target_pcd_pts, _, depth_mask = generate_RGBD_point_cloud(nerf, 
                                                                              init_guess,
                                                                              save_image=save_image,
                                                                              filename=pnp_figure_filename,
                                                                              return_pcd=False)
      
        # target point cloud
        target_pcd_cam = target_pcd_pts.view(-1, 3)
        pts_shape = target_pcd_cam.shape

        # convert from OPENGL Camera convention to OPENCV convention
        init_guess_gl = init_guess.detach().clone()
        init_guess_gl[:, 1] = -init_guess_gl[:, 1]
        init_guess_gl[:, 2] = -init_guess_gl[:, 2]

        # transform to the world space
        target_pcd = init_guess_gl @ torch.cat((target_pcd_cam,
                                        torch.ones(pts_shape[0], 1, device=nerf.device)),
                                        axis=-1).T
        target_pcd = target_pcd.T.view((*target_pcd_pts.shape[:2], target_pcd.shape[0]))[..., :3]
        
        # source image (Image obtained from the camera.)
        source_img = (rgb_input.cpu().numpy() * 255).astype(np.uint8)

        # target image (Image rendered at the initial guess.)
        target_img = (target_rgb.cpu().numpy() * 255).astype(np.uint8)

        # start time
        t0 = time.perf_counter()

        # feature matching
        if feature_detector == POI_Detector.LIGHTGLUE:
            # extract local features
            source_feats = self.feature_extractor.extract(rgb_input.permute(2, 0, 1))
            target_feats = self.feature_extractor.extract(target_rgb.permute(2, 0, 1))

            # match the features
            matches = self.feature_matcher({'image0': source_feats, 'image1': target_feats})

            # remove batch dimension
            source_feats, target_feats, matches = [rbd(x) for x in [source_feats, target_feats, matches]]

            # macthes: indices with shape (K, 2)
            matches = matches['matches']

            # matched points
            # coordinates in the source image with shape (K, 2)
            source_xy_matches = source_feats['keypoints'][matches[..., 0]].long().cpu().numpy()

            # coordinates in target image with shape (K, 2)
            target_xy_matches = target_feats['keypoints'][matches[..., 1]].long()
        else:
            source_xy, source_descriptors, source_kp_img = find_POI(source_img, 1000, viz=False, detector=feature_detector)
            target_xy, target_descriptors, target_kp_img = find_POI(target_img, 1000, viz=False, detector=feature_detector)
            matches, Matches = feature_matching(np.array(source_descriptors), np.array(target_descriptors), detector=feature_detector)

            # number of matches
            num_matches = len(matches)

            if print_stats:
                print(f'Found {num_matches} matches!')
                
            # Start matching as points (x, y)
            source_xy_matches = []
            target_xy_matches = []
            
            for match in matches:
                target_xy_matches.append(target_xy[match.trainIdx])
                source_xy_matches.append(source_xy[match.queryIdx])
                
            # source and target matches
            source_xy_matches = np.stack(source_xy_matches, axis=0)
            target_xy_matches = np.stack(target_xy_matches, axis=0)
            
        # apply depth mask
        source_xy_matches[depth_mask[source_xy_matches[:, 1], source_xy_matches[:, 0]].cpu().numpy()]
        target_xy_matches[depth_mask[target_xy_matches[:, 1], target_xy_matches[:, 0]].cpu().numpy()]

        # Select matches from target point cloud
        target_3d_matches = target_pcd[target_xy_matches[:, 1], target_xy_matches[:, 0]]
        target_3d_matches = target_3d_matches.cpu().numpy()

        # Perform pnp ransac
        guess = init_guess.cpu().numpy()
        guess_w2c = np.linalg.inv(guess)
        t_guess = guess_w2c[:3, -1].reshape(-1, 1)
        r_guess = guess_w2c[:3, :3]
        R_vec_guess = cv2.Rodrigues(r_guess)[0]

        # PnP-RANSAC
        success, R_vec, t, inliers = cv2.solvePnPRansac(
                target_3d_matches, 
                source_xy_matches.astype(np.float32), 
                camera_intrinsics_K.cpu().numpy(), distCoeffs=None,
                rvec=R_vec_guess, tvec=t_guess,
                flags=cv2.SOLVEPNP_EPNP, #SOLVEPNP_ITERATIVE
                confidence=0.99,
                reprojectionError=8.0,
                # useExtrinsicGuess=True
                )

        t1 = time.perf_counter()
        
        if print_stats:
            print(f'time PnP: {t1 - t0} secs.')

        # Retrieve the camera to world transform
        # openCV camera frame is x to right, y down, z forward
        # Also, r and t rotate world to camera
        est_pose = np.eye(4)
        r = cv2.Rodrigues(R_vec)[0]
        w2c_cv = np.hstack([r, t])

        est_pose[:3] = w2c_cv
        est_pose = np.linalg.inv(est_pose)
        est_pose[:, 1] = -est_pose[:, 1]
        est_pose[:, 2] = -est_pose[:, 2]

        if not success:
            print(f"PNP RANSAC FAILED!")
            raise RuntimeError(f"PNP RANSAC FAILED!")
        else:
            if print_stats:
                print(f"PNP RANSAC SUCCEEDED!")
        
        if visualize_PnP_matches:
            fig = plt.figure()
            # flann_matches =cv2.drawMatchesKnn(source_img, source_kp_img,
                                              # target_img, target_kp_img,
                                              # Matches, None)
            # plt.imshow(flann_matches)
            plt.imshow(np.concatenate([source_img, target_img], axis=1))
            plt.show()

            # save figure
            # fig.savefig(pnp_matches_figure_filename)
            
        return est_pose

    def estimate(self,
                 nerf: EnvModel,
                 cam_rgb: torch.Tensor,
                 ground_truth_pose: np.ndarray = None
                ):
        '''
        Evaluates pose estimation algorithms.
        Arguments:
            nerf: EnvModel - radiance field representing the environment,
            cam_rgb: torch.Tensor - RGB image from the camera,
            ground_truth_pose: np.ndarray - ground-truth pose, only needed for evaulation purposes.
        '''
        # #
        # # Perspective-n-Point (PnP)
        # #
        
        try:
            if self.visualization_options['print_stats']:
                print(self.visualization_options['print_options'].sep_0)
                print(f"Recursive Pose Estimation")
                print(self.visualization_options['print_options'].sep_0)
                
            # start time
            t0 = time.perf_counter()
            
            # estimated pose
            est_pose = self.execute_PnP_RANSAC(nerf, self.init_guess,
                                               camera_intrinsics_K=self.camera_K,
                                               rgb_input=cam_rgb,
                                               feature_detector=self.feature_detector,
                                               save_image=False,
                                               pnp_figure_filename=f'{self.visualization_options["filepath_figures"]}/pnp_init_guess.png',
                                               print_stats=self.visualization_options['print_stats'],
                                               visualize_PnP_matches=self.visualization_options['visualize_kp_and_matches'],
                                               pnp_matches_figure_filename=f'{self.visualization_options["filepath_figures"]}/pnp_matches.png',
                                               detector_params={})

            # end time
            t1 = time.perf_counter()

            # computation time
            comp_time = t1 - t0
            
            if self.compute_pose_error and ground_truth_pose is not None:  
                # pose error
                error = SE3error(ground_truth_pose, est_pose)

                if self.visualization_options['print_stats']:
                    print(f"SE(3) Estimation Error -- Rotation (deg): {error[0]}, Translation (m): {error[1]}")
                    print(f"Recursive Pose Estimation took {t1 - t0} seconds!")
                    
                    print(self.visualization_options['print_options'].sep_1)
                    print(self.visualization_options['print_options'].sep_1) 
            
                # store the pose error
                self.pose_error.append(error)  
                 
            if self.visualization_options['render_RGB_at_estimated_pose']:
                # generate RGB-D image at the estimated pose
                cam_rgb, _, cam_pcd, *_ = generate_RGBD_point_cloud(nerf,
                                                                    pose=torch.tensor(est_pose, device=self.device).float(),
                                                                    save_image=True, 
                                                                    filename=f'{self.visualization_options["filepath_figures"]}/est_pose_{self.timestep}.png')

            # close figures
            plt.close('all')
            
            # computation time
            self.comp_time.append(comp_time)
            
            # advance the timestep
            self.timestep += 1
            
            # update the initial guess
            self.init_guess = torch.tensor(est_pose, device=self.device).float()
            
            return est_pose
            
        except (RuntimeError, cv2.error, KeyError) as err:
            print(f"{Fore.RED}{self.visualization_options['print_options'].sep_1}{Style.RESET_ALL}")
            print(f"Recursive Pose Estimation")
            print(f"An exception {Fore.RED}{err}{Style.RESET_ALL} occurred during execution.")
            print(self.visualization_options['print_options'].sep_1)
            print_exc()
            print(self.visualization_options['print_options'].sep_1)
            
