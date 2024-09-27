import cv2
import numpy as np

import rospy
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs


# top-down overlay + top-down trajectory
# visualize waypoints
# draw waypoint image
# draw rectangle
# show next frame
# rectify image

FONT_SIZE = 0.3

# COLOR code in BGR format
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)

TOP_DOWN_VIEW_SCALE = 10
CURRENT_TOP_DOWN_POS = (75, 190)

def draw_rectangle(frame, x, y, w, h):
    sub_img = frame[y:y+h, x:x+w]
    rect = np.full(sub_img.shape, (0, 0, 0), np.uint8)
    alpha = 0.6
    res = cv2.addWeighted(sub_img, alpha, rect, 1-alpha, 0)
    frame[y:y+h, x:x+w] = res


def show_next_frame(img):
    img_resized = cv2.resize(img, (2*img.shape[1], 2*img.shape[0]), interpolation=(cv2.INTER_AREA))
    cv2.imshow('Live Camera Frame', img_resized)


def rectify_image(camera_model, img):
    rectified_image = np.empty_like(img)
    camera_model.rectifyImage(raw=img, rectified=rectified_image)
    return rectified_image


def show_info_overlay(frame, v, w, goal_distance, font_size=FONT_SIZE):
        
        draw_rectangle(frame, 5, 5, 140, 200)

        cv2.putText(frame, 'Driving Commands:', (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, '  Linear_X: {:.2f}'.format(v), (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, '  Angular_Z: {:.2f}'.format(w), (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, 'Distance to goal: {:.2f}'.format(goal_distance), (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, YELLOW, 1, cv2.LINE_AA)


def show_laser_scan_projection(current_image, laser_coordinates, camera_model, transform, current_top_down_position=CURRENT_TOP_DOWN_POS):        
        
        if laser_coordinates is not None:
            # show laser scan projection over current image
            laser_coordinates_homogeneous = np.ones(shape=(laser_coordinates.shape[0], 4))
            laser_coordinates_homogeneous[:, :2] = laser_coordinates
            laser_coordinates_homogeneous[:, 2] = -0.5 * np.ones(laser_coordinates.shape[0])

            laser_coordinates_homogeneous_optical_frame = np.dot(laser_coordinates_homogeneous, transform.T)
            laser_coordiantes_3d_optical_frame = laser_coordinates_homogeneous_optical_frame[:, :3]

            camera_pixel_coords = np.zeros(shape=(laser_coordiantes_3d_optical_frame.shape[0], 2))

            for id, point in enumerate(laser_coordiantes_3d_optical_frame):
                camera_pixel_coords[id] = camera_model.project3dToPixel(point)

            for i, pixel_coord in enumerate(camera_pixel_coords):            
                current_wp = (int(pixel_coord[0]), int(pixel_coord[1]))
                cv2.circle(current_image, current_wp, 4, GREEN, -1)
                
                if i > 0:
                    prev_wp = (int(camera_pixel_coords[i-1][0]), int(camera_pixel_coords[i-1][1]))
                    cv2.line(current_image, prev_wp, current_wp, GREEN, 2)

            # show laser scans in top-down view        
            for i, point in enumerate(laser_coordinates):
                scaled_down_point = (current_top_down_position[0] - int(TOP_DOWN_VIEW_SCALE * point[1]),
                                    current_top_down_position[1] - int(TOP_DOWN_VIEW_SCALE * point[0]))
                cv2.circle(current_image, scaled_down_point, 2, GREEN, -1)

                if i > 0:
                    prev_point = laser_coordinates[i - 1]
                    scaled_prev_point = (current_top_down_position[0] - int(TOP_DOWN_VIEW_SCALE * prev_point[1]),
                                        current_top_down_position[1] - int(TOP_DOWN_VIEW_SCALE * prev_point[0]))
                    cv2.line(current_image, scaled_prev_point, scaled_down_point, GREEN, 1)


def show_trajectories_projection(current_image, best_waypoint_id, collision_free_trajectories, camera_model, transform, radius, current_pos=CURRENT_TOP_DOWN_POS):
        
        trajectories_in_cam_frame = collision_free_trajectories
        trajectories_in_cam_frame = trajectories_in_cam_frame.reshape(-1, 2)

        # Convert the 2D coordinates to 4D homogeneous coordinates
        trajectories_homogeneous_cam_frame = np.ones(shape=(trajectories_in_cam_frame.shape[0], 4))
        trajectories_homogeneous_cam_frame[:, :2] = trajectories_in_cam_frame
        trajectories_homogeneous_cam_frame[:, 2] = -0.55 * np.ones(trajectories_in_cam_frame.shape[0])

        # Transform the points from camera frame to optical frame
        trajectories_homogeneous_optical_frame = np.dot(trajectories_homogeneous_cam_frame, transform.T)
        
        # Extract the 3D points from the homogeneous coordiantes
        trajectories_3d_optical_frame = trajectories_homogeneous_optical_frame[:, :3]
        camera_pixel_coords = np.zeros(shape=(trajectories_homogeneous_optical_frame.shape[0], 2))
        
        # Convert the 3D points to (u,v) pixel coordinates
        for id, point in enumerate(trajectories_3d_optical_frame):
            camera_pixel_coords[id] = camera_model.project3dToPixel(point)

        # Plot the trajectories waypoints and draw a line throuygh individual trajectories
        camera_pixel_coords = camera_pixel_coords.reshape(collision_free_trajectories.shape[0], collision_free_trajectories.shape[1], collision_free_trajectories.shape[2])        
        for trajectory_pixel_coordinates in camera_pixel_coords:
            for i, pixel_coord in enumerate(trajectory_pixel_coordinates): 
                current_wp = (int(pixel_coord[0]), int(pixel_coord[1]))
                cv2.circle(current_image, current_wp, radius, RED, -1)
                
                if i > 0:
                    prev_wp = (int(trajectory_pixel_coordinates[i-1][0]), int(trajectory_pixel_coordinates[i-1][1]))
                    cv2.line(current_image, prev_wp, current_wp, RED, 2)

        # Plot the best trajectory for the robot to follow
        best_trajectory_pixel_coords = camera_pixel_coords[best_waypoint_id]
        for i, pixel_coord in enumerate(best_trajectory_pixel_coords):            
            current_wp = (int(pixel_coord[0]), int(pixel_coord[1]))
            cv2.circle(current_image, current_wp, radius, BLUE, -1)
            
            if i > 0:
                prev_wp = (int(best_trajectory_pixel_coords[i-1][0]), int(best_trajectory_pixel_coords[i-1][1]))
                cv2.line(current_image, prev_wp, current_wp, BLUE, 2)        
        
        # Top-down projection of trajectories
        for traj in collision_free_trajectories:
            for i, point in enumerate(traj):
                scaled_down_point = (current_pos[0] - int(TOP_DOWN_VIEW_SCALE * point[1]),
                                    current_pos[1] - int(TOP_DOWN_VIEW_SCALE * point[0]))
                cv2.circle(current_image, scaled_down_point, radius, RED, -1)

                if i > 0:
                    prev_point = traj[i - 1]
                    scaled_prev_point = (current_pos[0] - int(TOP_DOWN_VIEW_SCALE * prev_point[1]),
                                        current_pos[1] - int(TOP_DOWN_VIEW_SCALE * prev_point[0]))
                    cv2.line(current_image, scaled_prev_point, scaled_down_point, RED, 2)

                # when i = 0
                else:
                    scaled_first_point = (current_pos[0] - int(TOP_DOWN_VIEW_SCALE * point[1]),
                                        current_pos[1] - int(TOP_DOWN_VIEW_SCALE * point[0]))
                    cv2.line(current_image, scaled_first_point, current_pos, RED, 2)

            best_trajectory = collision_free_trajectories[best_waypoint_id]
            for i, point in enumerate(best_trajectory):
                
                scaled_down_point = (current_pos[0] - int(TOP_DOWN_VIEW_SCALE * point[1]),
                                    current_pos[1] - int(TOP_DOWN_VIEW_SCALE * point[0]))
                cv2.circle(current_image, scaled_down_point, radius, BLUE, -1)

                if i > 0:
                    prev_point = best_trajectory[i - 1]                    
                    scaled_prev_point = (current_pos[0] - int(TOP_DOWN_VIEW_SCALE * prev_point[1]),
                                        current_pos[1] - int(TOP_DOWN_VIEW_SCALE * prev_point[0]))
                    cv2.line(current_image, scaled_prev_point, scaled_down_point, BLUE, 2) 

                else:
                    scaled_first_point = (current_pos[0] - int(TOP_DOWN_VIEW_SCALE * point[1]),
                                          current_pos[1] - int(TOP_DOWN_VIEW_SCALE * point[0]))
                    cv2.line(current_image, scaled_first_point, current_pos, BLUE, 2)

def show_visualization_window(current_image,                                   
                              goal_image,
                              collision_free_trajectories,
                              map_reader,
                              global_planner_viz,
                              current_position,
                              candidate_px,
                              transform,
                              camera_model,                                  
                              best_waypoint_id, 
                              laser_coordinates,
                              trajectories_map_img,
                              probability_map_img,
                              v,
                              w,                                                                     
                              goal_distance):
        
        """
        Function to show the visualiation window
        
        visualize always:
            - Current image
            - Overlay rectangle (left pane)
            - Info overlay
            - Current position in top-down view
            - Laser scan in top-down view
            - Laser scan projection over current image
            - Trajectory map (current position, goal psoition and heading angle)
            - Goal image
        
        visualize when there are valid trajectories (collision-free):
            - Trajectories projection over current image
            - Trajectories in top-down view
            - Trajectory map with waypoint's position
            - Probability map (prob map with robot's current position, goal position and waypoint's position)            
        """
        
        if candidate_px is not None:
            # Plot the waypoints in trajectory map
            for wp_px in candidate_px:
                wp_crop_coords = map_reader.to_crop_coordinates(current_position, wp_px)
                cv2.circle(trajectories_map_img, wp_crop_coords, 3, (0, 0, 0), -1)
            
            # Get the pixel coordinate of the best waypoint and plot over trajectories map
            best_waypoint_px = candidate_px[best_waypoint_id]
            best_waypoint_crop_coords = map_reader.to_crop_coordinates(current_position, best_waypoint_px)
            cv2.circle(trajectories_map_img, best_waypoint_crop_coords, 3, GREEN, -1)

            # plot the wayoints and the best waypoint over probability map
            for wp_px in candidate_px:
                wp_crop_coords = map_reader.to_crop_coordinates(current_position, wp_px)
                cv2.circle(probability_map_img, wp_crop_coords, 3, (0, 0, 0), -1)   
            cv2.circle(probability_map_img, best_waypoint_crop_coords, 3, GREEN, -1)
            
            #  visualize trajectories projection over image and in top-down view
            show_trajectories_projection(current_image=current_image,
                                         best_waypoint_id=best_waypoint_id,
                                         collision_free_trajectories=collision_free_trajectories,
                                         camera_model=camera_model,
                                         transform=transform,
                                         radius=4)

        # Crop the trajectories map and probability map
        trajectories_map_img_cropped = global_planner_viz.crop_map(trajectories_map_img)        
        probability_map_img_cropped = global_planner_viz.crop_map(probability_map_img)

        # Stack the map images vertically
        global_map_img = np.concatenate((trajectories_map_img_cropped, probability_map_img_cropped), axis=1)
        global_map_img_resized = cv2.resize(global_map_img, (200, 100))

        global_map_height, global_map_width, _ = global_map_img_resized.shape
        _, width, _ = current_image.shape
        
        start_x = width - global_map_width
        start_y = 5

        global_map_img_resized = cv2.cvtColor(global_map_img_resized, cv2.COLOR_BGR2RGB)
        
        # Place the global map on the top-right corner
        current_image[start_y:start_y+global_map_height, start_x:width] = global_map_img_resized
        
        # Place the goal image besides the global map
        goal_img_resized = cv2.resize(goal_image, (200, 100))
        draw_goal_image(current_image, goal_img_resized)

        # Place the info-overlay pane on the top-left corner
        show_info_overlay(frame=current_image,                               
                          v=v, 
                          w=w, 
                          goal_distance=goal_distance)
        
        # Project the laser scans over the current camera image, alongside top-down projection with the inf-overlay pane
        show_laser_scan_projection(current_image=current_image,
                                   camera_model=camera_model,
                                   transform=transform,
                                   laser_coordinates=laser_coordinates)

        # show current position in top-down view
        cv2.circle(current_image, CURRENT_TOP_DOWN_POS, 4, BLUE, -1)

        # Visualize everythong over the current image
        show_next_frame(current_image) 


def draw_goal_image(frame, goal_img):
    
    start_x, start_y = 235, 5
    frame[start_y:goal_img.shape[0]+start_y, start_x:goal_img.shape[1]+start_x] = goal_img


