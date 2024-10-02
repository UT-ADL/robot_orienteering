
import cv2
import numpy as np
from matplotlib import cm

from .util import draw_direction

BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 102, 0)
CYAN = (0, 255, 255)


class GlobalPlannerViz:

    def __init__(self, global_planner, adjust_heading=True):
        self.map_reader = global_planner.map_reader
        # If True, current position and goal position are given in gps lat/lon coordinates and candidate trajectory in
        # relative coordinates and are converted to pixel coordinates. This typicallly used with real robot.
        # If False, all coordinates are assumed to be already in pixel coordinate and no conversion is done.
        # This is typically used with simulation environment.

        self.convert_to_px = global_planner.convert_to_px
        self.probability_map = global_planner.probability_map
        
        self.adjust_heading = adjust_heading

    # def plot_trajectories_map(self, current_pos, goal_pos, north_heading, trajectories, trajectory_colors):
    def plot_trajectories_map(self, current_pos, goal_pos, north_heading):
        if self.adjust_heading:
            north_heading = self.map_reader.adjust_heading(current_pos[0], current_pos[1], north_heading)

        if self.convert_to_px:
            current_pos = self.map_reader.to_px(current_pos)
            goal_pos = self.map_reader.to_px(goal_pos)

        cropped_map = self.map_reader.crop_map_by_position(current_pos)
        
        # Convert pixel coordinates to crop based pixel coordinates
        current_pos_crop = self.map_reader.to_crop_coordinates(current_pos, current_pos)
        goal_pos_crop = self.map_reader.to_crop_coordinates(current_pos, goal_pos)

        # Draw start and end positions
        cv2.circle(cropped_map, (int(current_pos_crop[0]), int(current_pos_crop[1])), 2, BLUE, 2)
        cv2.circle(cropped_map, (int(goal_pos_crop[0]), int(goal_pos_crop[1])), 2, YELLOW, 2)

        draw_direction(current_pos_crop, north_heading, cropped_map, length=15, color=BLUE, thickness=2)
        return cropped_map

    def plot_probability_map(self, current_pos, goal_pos):
        if self.convert_to_px:
            current_pos = self.map_reader.to_px(current_pos)
            goal_pos = self.map_reader.to_px(goal_pos)

        prob_map_colors = self.probability_map_to_img()

        # Convert pixel coordinates to crop based pixel coordinates
        current_pos_crop = self.map_reader.to_crop_coordinates(current_pos, current_pos)
        goal_pos_crop = self.map_reader.to_crop_coordinates(current_pos, goal_pos)

        probability_map_img = prob_map_colors
        cv2.circle(probability_map_img, (int(current_pos_crop[0]), int(current_pos_crop[1])), 2, BLUE, 2)
        cv2.circle(probability_map_img, (int(goal_pos_crop[0]), int(goal_pos_crop[1])), 2, YELLOW, 2)
        return probability_map_img

    def prepare_global_map_img(self, trajectories_map_img, probability_map_img, current_position, candidate_px, best_waypoint_id):
        
        # While using the euclidean clustering
        if probability_map_img is None:
            if candidate_px is not None:
                best_wp_px = candidate_px[best_waypoint_id]
                best_waypoint_crop_coords = self.map_reader.to_crop_coordinates(current_position, best_wp_px)
                # spawn candidate px over trajectories map only
                # set the candidate px spawned trajectories map as global map img
                for wp_px in candidate_px:
                    wp_crop_coords = self.map_reader.to_crop_coordinates(current_position, wp_px)
                    
                    # plot all waypoints in black
                    cv2.circle(trajectories_map_img, wp_crop_coords, 3, (0, 0, 0), -1)

                # plot best waypoint in green
                cv2.circle(trajectories_map_img, best_waypoint_crop_coords, 3, (0, 255, 0), -1)
            
            trajectories_map_img_cropped = self.crop_map(trajectories_map_img)
            
            # set global map img as trajectories map only
            global_map_img = trajectories_map_img_cropped
        
        # While using the global planner model
        else:
            if candidate_px is not None:
                for wp_px in candidate_px:
                    wp_crop_coords = self.map_reader.to_crop_coordinates(current_position, wp_px)
                    cv2.circle(trajectories_map_img, wp_crop_coords, 3, (0, 0, 0), -1)
                    cv2.circle(probability_map_img, wp_crop_coords, 3, (0, 0, 0), -1)
                
                cv2.circle(trajectories_map_img, wp_crop_coords, 3, (0, 255, 0), -1)
                cv2.circle(probability_map_img, wp_crop_coords, 3, (0, 255, 0), -1)

            probability_map_img_cropped = self.crop_map(probability_map_img)
            trajectories_map_img_cropped = self.crop_map(trajectories_map_img)

            # concatenate probability map and trajectories map as global map img
            global_map_img = np.concatenate((probability_map_img_cropped, trajectories_map_img_cropped), axis=1)

        return global_map_img

    def crop_map(self, map_img):
        map_height, map_width, _ = map_img.shape
        map_center_y, map_center_x = map_height // 2, map_width // 2
        map_start_y, map_start_x = map_center_y - 50, map_center_x - 50

        map_img_cropped = map_img[map_start_y:map_start_y + 100, 
                                  map_start_x:map_start_x + 100]
        
        return map_img_cropped

    def probability_map_to_img(self):
        # Convert probabilities into colors
        prob_map_int = (255 * self.probability_map).astype(np.uint8)
        prob_map_colors = cm.Reds(prob_map_int)
        prob_map_colors = (prob_map_colors * 255).astype(np.uint8)
        prob_map_colors = cv2.cvtColor(prob_map_colors, cv2.COLOR_RGBA2RGB)
        return prob_map_colors