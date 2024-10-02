import os
import yaml

from collections import deque

import cv2
import numpy as np
import re

import rospy
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel

import tf
import tf2_ros

from shapely import LineString

import imghdr
import exifread

from std_msgs.msg import Float32
from sensor_msgs.msg import CameraInfo, Image, LaserScan 
from geometry_msgs.msg import Twist, Vector3Stamped

from global_planner.data.mapping import MapReader
from global_planner.models.global_planner_onnx import GlobalPlannerOnnx
from global_planner.viz.global_planner_viz import GlobalPlannerViz

from utils.viz_util import show_trajectories_projection, show_laser_scan_projection, show_info_overlay, show_next_frame
from utils.preprocessing_images import center_crop_and_resize, prepare_image
from utils.batching import batch_obs_plus_context

from waypoint_planner.model.nomad_onnx import NomadOnnx


TRAJECTORIES = np.array([((0,0), (0.93969262, -0.34202014), (2.59807621, -1.5), (3.83022222, -3.21393805)),
                         ((0,0), (0.98480775, -0.17364818), (2.89777748, -0.77645714), (4.6984631, -1.71010072)),
                         ((0,0), (1, 0),  (3, 0), (5, 0)),
                         ((0,0), (0.98480775, 0.17364818), (2.89777748, 0.77645714), (4.6984631, 1.71010072)),
                         ((0,0), (0.93969262, 0.34202014), (2.59807621, 1.5), (3.83022222, 3.21393805))])

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


def extract_gps_info(image_stream):
    tags = exifread.process_file(image_stream)
    
    try:
        lat = [float(x.num) / float(x.den) for x in tags['GPS GPSLatitude'].values]
        lon = [float(x.num) / float(x.den) for x in tags['GPS GPSLongitude'].values]
    except KeyError:
        return None
    return lat, lon


class JackalDrive:

    def __init__(self):

        # Fetch parameters
        self.goal_image_dir = rospy.get_param('~goal_image_dir')

        self.config_dir_path = rospy.get_param('~config_dir_path')

        self.local_model_type = rospy.get_param('~local_model_type')
        self.local_model_path = rospy.get_param('~local_model_path')
        
        self.global_model_path = rospy.get_param('~global_model_path')
        self.map_path = rospy.get_param('~map_path')
        self.map_name = rospy.get_param('~map_name')
        
        self.base_link_frame = rospy.get_param('~base_link_frame', 'base_link_frame')
        self.gps_frame = rospy.get_param('~gps_frame')
        self.left_camera_frame = rospy.get_param('~left_camera_frame', 'zed2i_left_camera_frame')
        self.left_camera_optical_frame = rospy.get_param('~left_camera_optical_frame', 'zed2i_left_camera_optical_frame')
        
        self.fps = int(rospy.get_param('~fps'))     
        self.record_video = rospy.get_param('~record_video')
        self.timer_frequency = rospy.get_param('~timer_frequency')
        
        self.nomad_conditioning_threshold = float(rospy.get_param('~nomad_conditioning_threshold'))
        self.gps_conditioning_threshold = float(rospy.get_param('~gps_conditioning_threshold'))

        self.use_laser = bool(rospy.get_param('~use_laser', True))
        self.use_global_planner = bool(rospy.get_param('~use_global_planner', False))
        self.use_nomad = bool(rospy.get_param('~use_nomad', False))
        
        rospy.loginfo(f"use_global_planner: {self.use_global_planner}")
        rospy.loginfo(f"use_laser: {self.use_laser}")
        rospy.loginfo(f"use_nomad: {self.use_nomad}")

        if self.use_nomad:
            self.goal_conditioning_threshold = self.nomad_conditioning_threshold
        else:
            self.goal_conditioning_threshold = self.gps_conditioning_threshold

        # Initialize tf2 listener and CV bridge
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()

        # Load global planner configration
        # global_planner_config_file = os.path.join(self.config_dir_path, 'distance_segment.yaml')
        global_planner_config_file = os.path.join(self.config_dir_path, 'default_segment.yaml')
        with open(global_planner_config_file, "r") as f:
            self.global_planner_config = yaml.safe_load(f)
        rospy.loginfo(f"Loaded global planner config: {global_planner_config_file}")

        # Load global planner map
        map_type = self.global_planner_config["map_type"]
        map_file_path = os.path.join(self.map_path, f"{self.map_name}_{map_type}.tif")
        self.map_reader = MapReader(map_file_path, self.global_planner_config["map_size"])
        rospy.loginfo(f"Loaded global planner map: {map_file_path}")
        rospy.loginfo(f"Map resolution: {self.map_reader.map_resolution}")

        # Load global planner model
        self.global_planner = GlobalPlannerOnnx(self.map_reader, self.global_planner_config, self.global_model_path, convert_to_px=False)        
        rospy.loginfo(f"Loaded global planner model: {self.global_model_path}")

        # Load local planner configuration
        local_planner_config_file = os.path.join(self.config_dir_path, str(self.local_model_type) + '.yaml')
        if os.path.exists(local_planner_config_file):
            with open(local_planner_config_file, 'r') as f:
                self.local_planner_config = yaml.safe_load(f)
        else:
            raise Exception(f"No such file: {local_planner_config_file}")
        rospy.loginfo(f"Loaded local planner config: {local_planner_config_file}")  

        self.img_size = self.local_planner_config['image_size']
        self.waypoint_length = self.local_planner_config['len_traj_pred']
        self.context_length = self.local_planner_config['context_size']     
        self.waypoint_spacing = self.local_planner_config['context_spacing'] # 4

        rospy.loginfo(f"Observation size: {self.img_size}")

        self.local_session = NomadOnnx(self.local_model_path)

        rospy.loginfo(f"Loaded local planner model: {self.local_model_path}")

        if self.fps == 15:
            self.buffer_length = self.context_length * self.waypoint_spacing + 1
        #if fps = 4, buffer--> 6 consecutive images
        elif self.fps == 4:
            self.buffer_length = self.context_length + 1
        else:
            raise Exception(f"FPS {self.fps} not supported")

        self.deque_images = deque(maxlen = self.buffer_length)

        print(f"fps: {self.fps}")

        # Extracting the image meta data from image EXIF
        goal_image_names = [img for img in os.listdir(self.goal_image_dir) if os.path.join(self.goal_image_dir, img)]
        goal_image_names = sorted(goal_image_names, key=natural_sort_key)
        rospy.loginfo(f"goal images retrieved: {goal_image_names}")

        self.goal_images = []
        self.goal_gps = []

        for img in goal_image_names:

            img_filepath = os.path.join(self.goal_image_dir, img)

            img_type = imghdr.what(img_filepath)
            if img_type not in ALLOWED_EXTENSIONS:
                rospy.logerr(f'File type not supported. Please upload an image of supported type. Supported types:{ALLOWED_EXTENSIONS}')
                rospy.signal_shutdown("Image file type not allowed..")
            
            goal_img = cv2.imread(img_filepath, cv2.IMREAD_COLOR)
            self.goal_images.append(goal_img)

            lat_lon = extract_gps_info(open(img_filepath, 'rb'))
            if not lat_lon:
                rospy.logerr('Image does not contain GPS information.')
                rospy.signal_shutdown("No GPS info in image EXIF")

            lat, lon = lat_lon
            latitude = sum(x / 60**n for n, x in enumerate(lat))
            longitude = sum(x / 60**n for n, x in enumerate(lon))

            self.goal_gps.append((latitude, longitude))

        self.goal_images = np.array(self.goal_images)
        self.goal_gps = np.array(self.goal_gps)

        rospy.loginfo(f"goal gps:\n{self.goal_gps}")

        self.goal_id = 0            

        self.current_gps = None
        self.current_position = None
        self.current_heading = None
        
        rospy.loginfo("Goal gps converted to pixel coordinates")

        # Load camera model
        camera_info_msg = rospy.wait_for_message('/zed/zed_node/left_raw/camera_info', CameraInfo)
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(camera_info_msg)
        rospy.loginfo("Loaded pinhole camera model")
        
        self.laser_linestring = None

        self.current_image = None
        self.vel = None

        # Initialize video writer
        if self.record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video = cv2.VideoWriter(self.record_video, fourcc, self.timer_frequency, (640, 360))
            rospy.loginfo(f"Recording video to: {self.record_video}")
        
        self.tf_from_cam_frame_to_base_link = self.create_tf_matrix(source_frame=self.left_camera_frame,
                                                                    target_frame=self.base_link_frame)
        self.tf_from_cam_frame_to_gps_frame = self.create_tf_matrix(source_frame=self.left_camera_frame,
                                                                    target_frame=self.gps_frame)
        self.tf_cam2opt_frame = self.create_tf_matrix(source_frame=self.left_camera_frame,
                                                      target_frame=self.left_camera_optical_frame)

        # Initialize ROS publishers and subscribers
        self.driving_command_publisher = rospy.Publisher('cmd_vel',
                                                          Twist,
                                                          queue_size=1)
        
        rospy.Timer(rospy.Duration(1.0/self.timer_frequency), self.timer_callback)
        
        rospy.Subscriber('/zed/zed_node/left_raw/image_raw_color',
                        Image,
                        self.image_callback,
                        queue_size=1,
                        buff_size=2**24)
        
        rospy.Subscriber('/scan',
                        LaserScan,
                        self.laser_callback,
                        queue_size=1)
        
        # self.current_gps = None
        rospy.Subscriber('/filter/positionlla',
                          Vector3Stamped,
                          self.gps_callback,
                          queue_size=1)

        # self.current_heading = None
        rospy.Subscriber('/current_heading',
                          Float32,
                          self.current_heading_callback,
                          queue_size=1)


    def create_tf_matrix(self, source_frame, target_frame):
        """
        Creates a 4X4 transformation matrix that will transform one/multiple homogeneous coordinates
        from source frame to target frame
        Input  :  source frame, target frame
        Output :  4X4 transformation matrix
        """
        transform = self.tf_buffer.lookup_transform(target_frame=target_frame, 
                                                    source_frame=source_frame, 
                                                    time=rospy.Time(), 
                                                    timeout=rospy.Duration(5.0))

        translation = np.array([transform.transform.translation.x,
                                transform.transform.translation.y,
                                transform.transform.translation.z])
        rotation = np.array([transform.transform.rotation.x,
                             transform.transform.rotation.y,
                             transform.transform.rotation.z,
                             transform.transform.rotation.w])
         
        transformation_matrix = np.dot(
            tf.transformations.translation_matrix(translation),
            tf.transformations.quaternion_matrix(rotation)
        )
        
        return transformation_matrix


    def transform_points(self, input_points, tf_matrix):
        """
        Transforms a point (or a set of points) from one frame to another based on the tf matrix
        Input  :  points in source frame of the tf_matrix
        Output :  points in target frame of the tf_matrix
        """
        input_points = input_points.reshape(-1, 2)
        points_in_source_frame = np.ones(shape=(input_points.shape[0], 4))
        # print(points_in_target_frame.shape)
        
        points_in_source_frame[:, :2] = input_points

        points_in_target_frame = np.dot(points_in_source_frame, tf_matrix.T)
        # points_in_target_frame = np.dot(tf_matrix, points_in_target_frame.T)
        # points_in_target_frame = points_in_target_frame.T

        return np.squeeze(points_in_target_frame[:, :2])
    

    def compute_gps(self, current_gps, waypoints):
        """
        Computes the GPS coordinates of the relative waypoints
        Input  :  current gps, relative waypoints(in gps_frame)
        Output :  GPS coordinates of the relative waypoints
        """
        waypoints = waypoints.reshape(-1, 2)
        relative_distances = np.sqrt((waypoints[:, 1])**2 + (waypoints[:, 0])**2)
        
        # Following right hand rule
        realtive_angles_rad = np.arctan2(waypoints[:, 1], waypoints[:, 0])
        realtive_angles = np.rad2deg(realtive_angles_rad)

        current_north_heading = self.current_heading
        waypoint_angles_north = current_north_heading - realtive_angles

        waypoint_angles_north[waypoint_angles_north < -180] += 360
        waypoint_angles_north[waypoint_angles_north > 180] -= 360

        current_gps_rad = np.deg2rad(current_gps)
        waypoint_angles_north_rad = np.deg2rad(waypoint_angles_north)

        cur_lat_rad = current_gps_rad[0]
        cur_lon_rad = current_gps_rad[1]
        
        R = 6371e3  # Earth's radius in meters

        new_lat = np.degrees(np.arcsin(np.sin(cur_lat_rad)*np.cos(relative_distances/R) + np.cos(cur_lat_rad)*np.sin(relative_distances/R)*np.cos(waypoint_angles_north_rad)))
        new_lon = current_gps[1] + np.degrees(np.arctan2(np.sin(waypoint_angles_north_rad)*np.sin(relative_distances/R)*np.cos(cur_lat_rad), 
                                                  np.cos(relative_distances/R)-np.sin(cur_lat_rad)*np.sin(np.radians(new_lat))))

        new_gps = np.column_stack((new_lat, new_lon))
        return np.squeeze(new_gps)
    

    def distance_from_gps(self, gps_origin, gps_destination):
        # gps are expected to be in angles (degrees)
        # first the degrees need to be converted to radians
        # then the distance is computed 

        # phi --> lat & lambda --> lon

        gps_origin = gps_origin.reshape(-1, 2)
        gps_destination = gps_destination.reshape(-1, 2)

        phi_1 = np.deg2rad(gps_origin[:, 0])           # gps --> (lat, lon); gps[0]=lat & gps[1]=lon 
        phi_2 = np.deg2rad(gps_destination[:, 0]) 

        del_phi = np.deg2rad(gps_destination[:, 0] - gps_origin[:, 0])
        del_lambda = np.deg2rad(gps_destination[:, 1] - gps_origin[:, 1])
        
        a = (np.sin(del_phi/2))**2 + np.cos(phi_1) * np.cos(phi_2) * (np.sin(del_lambda/2))**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        R = 6371e3 # earth's radius in meters
        d = R * c
        
        return np.squeeze(d)
        

    def image_callback(self, img_msg):
        
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        
        # Used only for NoMaD
        if self.use_nomad == True:
            self.deque_images.append(img)
        
        self.current_image = img

        if self.vel is not None:
            self.driving_command_publisher.publish(self.vel)


    def laser_callback(self, laser_msg):
        ranges = laser_msg.ranges
        range_min = laser_msg.range_min
        range_max = laser_msg.range_max
        
        angle_min = laser_msg.angle_min
        angle_increment = laser_msg.angle_increment

        raw_coordinates = []

        ranges_filtered = []
        angles_filtered = []
        for i, laser_range in enumerate(ranges):
            
            if np.isinf(laser_range):
                if laser_range == float("-inf"):
                    ranges_filtered.append(range_min)
                    
                elif laser_range == float("inf"):
                    ranges_filtered.append(range_max)
            
            else:
                ranges_filtered.append(laser_range)
            
            angles_filtered.append(angle_min + i * angle_increment)

        ranges_filtered = np.array(ranges_filtered)
        angles_filtered = np.array(angles_filtered)

        x = ranges_filtered * np.cos(angles_filtered)
        y = ranges_filtered * np.sin(angles_filtered)

        raw_coordinates = np.column_stack((x, y))
        raw_coordinates = raw_coordinates[~np.isnan(raw_coordinates).any(axis=1)]
        
        if len(raw_coordinates) >= 10:
            filtered_coordinates = raw_coordinates[::10]
        else:
            filtered_coordinates = raw_coordinates

        # To form a linestring, 2 points are required
        if len(filtered_coordinates) > 1:
            line = LineString(filtered_coordinates)            
        # Else just ignore those points and consider no laser points were obtained
        else:
            line = None

        self.laser_linestring = line


    def gps_callback(self, msg):
        self.current_gps = np.array([msg.vector.x, msg.vector.y])
        self.current_position = self.map_reader.to_px(self.current_gps)


    def current_heading_callback(self, msg):
        self.current_heading = msg.data    


    def timer_callback(self, event=None):
        
        if self.laser_linestring is None:
            rospy.loginfo_throttle(5, "laser scan not received")
            return
        
        if self.current_heading is None:
            rospy.loginfo("GPS heading not received ..")
            return
        
        if self.current_image is None:
            rospy.loginfo("Image from camera not received ..")
            return
        
        if self.use_nomad == True and len(self.deque_images) < self.buffer_length:
            rospy.loginfo_throttle(5, "Input image buffer filling up ...")
            return
        
        # robot_drive_mode = self.drive_mode
        current_image = self.current_image
        current_goal_gps = self.goal_gps[self.goal_id]       

        if self.use_nomad == False:
            trajectories = TRAJECTORIES
            distance_to_current_goal = self.distance_from_gps(gps_origin=self.current_gps,
                                                          gps_destination=current_goal_gps)
        else:
            # Set nomad action predictions as trajectories
            obs_img = batch_obs_plus_context(buffer_length = self.buffer_length,
                                         waypoint_spacing = self.waypoint_spacing,
                                         deque_images = self.deque_images, 
                                         fps = self.fps, 
                                         target_size=self.img_size)
            
            goal_img_resized = center_crop_and_resize(self.goal_images[self.goal_id], self.img_size)
            goal_img_preprocessed = prepare_image(goal_img_resized)

            local_model_out = self.local_session.predict(obs_tensor=obs_img, goal_tensor=goal_img_preprocessed)

            # print(local_model_out)
            temporal_distance_prediction = local_model_out[0].squeeze()
            action_predictions = local_model_out[1][:, :, :2].squeeze()

            trajectories = action_predictions
            distance_to_current_goal = temporal_distance_prediction

        if self.use_laser == True:

            collision_free_trajectories = []
            laser_linestring = self.laser_linestring

            laser_coordinates = np.array(laser_linestring.coords)

            for trajectory in trajectories: 

                trajectory_linestring = LineString(trajectory)                                 
                trajectory_buffer = trajectory_linestring.buffer(0.5)

                intersects = trajectory_buffer.intersects(laser_linestring)
                
                if not intersects:
                    collision_free_trajectories.append(trajectory)
            
            collision_free_trajectories = np.array(collision_free_trajectories)
        
        else:
            laser_coordinates = None
            collision_free_trajectories = trajectories

        # Compute the pixel coordinate for goal
        goal_position = self.map_reader.to_px(self.goal_gps[self.goal_id])

        # Run the model to get the probability map
        self.global_planner.predict_probabilities(self.current_position, goal_position)
        
        # Initialize the global map visualization
        self.global_planner_viz = GlobalPlannerViz(self.global_planner, adjust_heading=False)
        
        # Create trajectories map
        trajectories_map_img = self.global_planner_viz.plot_trajectories_map(self.current_position,
                                                                        goal_position,
                                                                        self.current_heading)
        
        # Initialize the probability map and candidate pixel coordiantes as as None
        # update it if using global planner model
        probability_map_img = None
        candidate_px = None

        # # If there are valid waypoints (collision free)
        if collision_free_trajectories.shape[0] > 0:

            candidate_waypoints = collision_free_trajectories[:, -1]
            candidate_waypoints_in_gps_frame = self.transform_points(input_points=candidate_waypoints,
                                                                    tf_matrix=self.tf_from_cam_frame_to_gps_frame)
            
            candidate_gps = self.compute_gps(current_gps=self.current_gps,
                                            waypoints=candidate_waypoints_in_gps_frame)
            
            candidate_gps = candidate_gps.reshape(-1, 2)        
            
            # Compute the pixel coordinates of the candidate waypoints
            # Used for plotting candidate waypoints over both maps (trajectories map + probability map)
            candidate_px = self.map_reader.lat_lon_to_pixel(candidate_gps[:, 0], candidate_gps[:, 1])        

            # Compute the individual waypoint's probabilities
            wp_prob = self.global_planner.calculate_probs(candidate_px,
                                                        self.current_position,
                                                        self.current_heading)
            
            # Select the best waypoint from all proposed waypoints
            # Euclidean clustering
            if self.use_global_planner == False:
                distance_goal_to_candidate_waypoints = self.distance_from_gps(gps_origin=candidate_gps,
                                                                            gps_destination=self.goal_gps[self.goal_id])
                best_waypoint_id = np.argmin(distance_goal_to_candidate_waypoints)
            
            # Global planner model based goal heuristic
            else:
                # Create probability map
                probability_map_img = self.global_planner_viz.plot_probability_map(self.current_position, goal_position)

                # Compute the individual waypoint's probabilities
                wp_prob = self.global_planner.calculate_probs(candidate_px,
                                                              self.current_position,
                                                              self.current_heading)
                best_waypoint_id = np.argmax(wp_prob)

            best_trajectory = collision_free_trajectories[best_waypoint_id]

            goal_x, goal_y = best_trajectory[-1]

            dist = np.sqrt((goal_x**2) + (goal_y**2))            
            theta = np.arctan2(goal_y, goal_x)

            v = dist
            w = theta

            v = np.clip(v, 0, 0.3)
            w = np.clip(w, -0.3, 0.3)

            # Publish driving command
            self.vel = Twist()
            self.vel.linear.x = v
            self.vel.angular.z = w
        
        # If there are no valid waypoints, turn counter-clockwise 
        else:
            
            rospy.loginfo("No collision free trajectory detected. Robot turning counter-clockwise...")
            candidate_px = None
            best_waypoint_id = None

            v = 0
            w = 0.2

            self.vel = Twist()
            self.vel.angular.z = w  
        
        # Set the global map image as required
        # If using the euclidean clustering, set it as only the trajectories map with the candidate waypoints
        # If using the global planner mode, set is as the probability map and trajectories map combined with the waypoints
        global_map_img = self.global_planner_viz.prepare_global_map_img(trajectories_map_img=trajectories_map_img, 
                                                                        probability_map_img=probability_map_img, 
                                                                        current_position=self.current_position, 
                                                                        candidate_px=candidate_px, 
                                                                        best_waypoint_id=best_waypoint_id)

        # visualize the global map img on top right corner
        global_map_height, global_map_width, _ = global_map_img.shape
        _, width, _ = current_image.shape

        global_map_start_x = width - global_map_width
        global_map_start_y = 1
        
        global_map_img = cv2.cvtColor(global_map_img, cv2.COLOR_BGR2RGB)
        current_image[global_map_start_y:global_map_start_y+global_map_height, global_map_start_x:width] = global_map_img
        
        # visualize goal img just besides the global map img
        goal_img_resized = cv2.resize(self.goal_images[self.goal_id], (200, 100))
        goal_img_height, goal_img_width, _ = goal_img_resized.shape
        
        goal_img_start_x = global_map_start_x - goal_img_width
        goal_img_start_y = 1
        current_image[goal_img_start_y:goal_img_start_y+goal_img_height, goal_img_start_x:goal_img_start_x+goal_img_width] = goal_img_resized

        # visualize trajectories projection (trajectories over current image and top-down view) if there are collision free trajectories
        if collision_free_trajectories.shape[0] > 0:
            show_trajectories_projection(current_image=current_image,
                                         best_waypoint_id=best_waypoint_id,
                                         collision_free_trajectories=collision_free_trajectories,
                                         camera_model=self.camera_model,
                                         transform=self.tf_cam2opt_frame,
                                         radius=4,
                                         linewidth=2)

        # visualize laser scan projection (laser scans over current image and top-down view)
        show_laser_scan_projection(current_image=current_image,
                                   camera_model=self.camera_model,
                                   transform=self.tf_cam2opt_frame,
                                   laser_coordinates=laser_coordinates)

        # Visualize info overlay (velocities, gps distance to goal, mode, # of disengagements)
        show_info_overlay(frame=current_image,
                          v=v,
                          w=w,
                          goal_distance=distance_to_current_goal)
        
        show_next_frame(img=current_image)
        
        # Add the frames to video stream if recording the frames
        if self.record_video:
                self.video.write(current_image)
        
        # Keyboard interrupt handling
        key = cv2.waitKey(1)
        
        # Shutdown all visualization windows and node if pressed 'ESC'
        if key == 27:
            if self.record_video:
                self.video.release()
            cv2.destroyAllWindows()
            rospy.signal_shutdown("User pressed ESC")
        
        # Switch to next goal image if pressed 'n'
        elif key == ord('n'):
            if self.goal_id < self.goal_images.shape[0] - 1:
                self.goal_id += 1
                print("switched to next goal")
                
        # Switch to previous goal image if pressed 'p'
        elif key == ord('p'):
            if self.goal_id > 0:
                self.goal_id -= 1
                print("switched to previous goal")
                
        if distance_to_current_goal < self.goal_conditioning_threshold:
            
            self.goal_id += 1

            if self.goal_id >= self.goal_images.shape[0]:
                if self.record_video:
                    self.video.release()
                cv2.destroyAllWindows()

                rospy.loginfo("Completed all goals !!")                                
                rospy.signal_shutdown("ALL GOALS REACHED !!")


if __name__ == "__main__":
    rospy.init_node("On_Policy_Robot_Test", log_level=rospy.INFO, anonymous=True)
    node = JackalDrive()    
    rospy.spin()