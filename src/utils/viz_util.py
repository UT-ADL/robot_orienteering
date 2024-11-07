import cv2
import numpy as np

FONT_SIZE = 0.32

# COLOR code in BGR format
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)

TOP_DOWN_VIEW_SCALE = 14
CURRENT_TOP_DOWN_POS = (210, 160)

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


def show_info_overlay(frame, v, w, goal_distance, drive_mode, num_goals_completed, total_goals, num_disengagements, manual_drive_time, total_time_elapsed, autonomy_percentage, font_size=FONT_SIZE):
        
        draw_rectangle(frame, 5, 5, 300, 200)

        cv2.putText(frame, 'Driving Commands:', (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, '  Linear_X: {:.2f}'.format(v), (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, '  Angular_Z: {:.2f}'.format(w), (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, 'Distance to goal: {:.2f} m'.format(goal_distance), (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, YELLOW, 1, cv2.LINE_AA)        
        cv2.putText(frame, f'Drive Mode: {drive_mode}', (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, f'Goals completed: {num_goals_completed}/{total_goals}', (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, f'Disengagements: {num_disengagements}', (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, 'Manual Drive Time: {:.2f} Sec'.format(manual_drive_time), (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, 'Total Time Elapsed: {:.2f} Sec'.format(total_time_elapsed), (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, 'Autonomy: {:.2f} %'.format(autonomy_percentage), (10, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, YELLOW, 1, cv2.LINE_AA)


def draw_top_down_overlay(img, coordinates, COLOR, radius, linewidth, current_top_down_position=CURRENT_TOP_DOWN_POS):

    for i, point in enumerate(coordinates):
        scaled_down_point = (current_top_down_position[0] - int(TOP_DOWN_VIEW_SCALE * point[1]),
                            current_top_down_position[1] - int(TOP_DOWN_VIEW_SCALE * point[0]))
        cv2.circle(img, scaled_down_point, radius, COLOR, -1)

        if i > 0:
            prev_point = coordinates[i - 1]
            scaled_prev_point = (current_top_down_position[0] - int(TOP_DOWN_VIEW_SCALE * prev_point[1]),
                                current_top_down_position[1] - int(TOP_DOWN_VIEW_SCALE * prev_point[0]))
            cv2.line(img, scaled_prev_point, scaled_down_point, COLOR, linewidth)


def project_over_camera_image(img, camera_pixel_coords, COLOR, radius, linewidth):
    for i, pixel_coord in enumerate(camera_pixel_coords):                   
        current_wp = (int(pixel_coord[0]), int(pixel_coord[1]))
        cv2.circle(img, current_wp, radius, COLOR, -1)
        
        if i > 0:
            prev_wp = (int(camera_pixel_coords[i-1][0]), int(camera_pixel_coords[i-1][1]))
            cv2.line(img, prev_wp, current_wp, COLOR, linewidth)

def convert_2d_coordinates_to_pixel_coordinates(coordinates, transform, camera_model):
    coordinates_homogeneous = np.ones(shape=(coordinates.shape[0], 4))
    coordinates_homogeneous[:, :2] = coordinates
    coordinates_homogeneous[:, 2] = -0.5 * np.ones(coordinates.shape[0])

    coordinates_homogeneous_optical_frame = np.dot(coordinates_homogeneous, transform.T)
    coordiantes_3d_optical_frame = coordinates_homogeneous_optical_frame[:, :3]

    camera_pixel_coords = np.zeros(shape=(coordiantes_3d_optical_frame.shape[0], 2))

    for id, point in enumerate(coordiantes_3d_optical_frame):
        camera_pixel_coords[id] = camera_model.project3dToPixel(point)
    
    return camera_pixel_coords


def show_laser_scan_projection(current_image, laser_coordinates, camera_model, transform, radius, linewidth):   
    if laser_coordinates is not None:
        camera_pixel_coords = convert_2d_coordinates_to_pixel_coordinates(coordinates=laser_coordinates,
                                                                            transform=transform,
                                                                            camera_model=camera_model)

        project_over_camera_image(img=current_image,
                                  camera_pixel_coords=camera_pixel_coords,
                                  COLOR=GREEN,
                                  radius=radius,
                                  linewidth=linewidth)
        
        draw_top_down_overlay(img=current_image,
                              coordinates=laser_coordinates,
                              COLOR=GREEN,
                              radius=radius,
                              linewidth=linewidth)


def show_trajectories_projection(current_image, best_waypoint_id, collision_free_trajectories, camera_model, transform, radius, linewidth):

    trajectories_in_cam_frame = collision_free_trajectories
    trajectories_in_cam_frame = trajectories_in_cam_frame.reshape(-1, 2)

    camera_pixel_coords = convert_2d_coordinates_to_pixel_coordinates(coordinates=trajectories_in_cam_frame,
                                                                      transform=transform,
                                                                      camera_model=camera_model)

    # Plot the trajectories waypoints and draw a line through individual trajectories
    camera_pixel_coords = camera_pixel_coords.reshape(collision_free_trajectories.shape[0], collision_free_trajectories.shape[1], collision_free_trajectories.shape[2]) 

    # Show trajectories projection over current image
    for trajectory_pixel_coordinates in camera_pixel_coords:
        project_over_camera_image(img=current_image,
                                  camera_pixel_coords=trajectory_pixel_coordinates,
                                  COLOR=RED,
                                  radius=radius,
                                  linewidth=linewidth)

    # Plot the best trajectory for the robot to follow
    best_trajectory_pixel_coords = camera_pixel_coords[best_waypoint_id]
    project_over_camera_image(img=current_image,
                              camera_pixel_coords=best_trajectory_pixel_coords,
                              COLOR=BLUE,
                              radius=radius,
                              linewidth=linewidth)
            
    
    # Top-down projection of trajectories
    for traj in collision_free_trajectories:
        draw_top_down_overlay(img=current_image,
                              coordinates=traj,
                              COLOR=RED,
                              radius=radius,
                              linewidth=linewidth)

    best_trajectory = collision_free_trajectories[best_waypoint_id]
    draw_top_down_overlay(img=current_image,
                          coordinates=best_trajectory,
                          COLOR=BLUE,
                          radius=4,
                          linewidth=linewidth) 


