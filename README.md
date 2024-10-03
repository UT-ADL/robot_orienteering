# robot_orienteering
Robot orienteering baselines


## Installation

**All packages should be located inside the `src` directory of the catkin workspace.**   

Please follow the following commands to create a catkin worksspace and the subsequent directory structure:
```
mkdir -p robot_orienteering_ws/src
cd robot_orienteering_ws/src
```

Now, all the ros packages should be present in this `src` directory.

### Prerequisite Packages
In order to run the baseline models/nodes, there are a few prerequisite packages that needs to be installed. These necessary pacages are listed below with instructions (or links to the instructions) on how to install them.

#### ZED ROS driver (driver for ZED camera)
To install the ZED ROS driver and use the ZED camera (any ZED family: ZED, ZED_mini, ZED2, ZED2i, etc.), the zed-ros-driver needs to be isntalled.   
To install the ZED ROS driver, please follow the [Getting Started with ROS and ZED](https://www.stereolabs.com/docs/ros)  instructions.

#### xsens_mti_driver (driver for the GNSS/INS device)
To install the xsens_mti_driver, please go through the [offial documentation](https://wiki.ros.org/xsens_mti_driver) of the xsens_mti_driver package. The package can be installed by cloning the github repo for the [xsens_mti_driver](https://github.com/nobleo/xsens_mti_driver).


Once the prerequisite packages and installed, you can move forward to install the main package `robot_orienteering`. 
**Please note that the src directory at this point should already contain some packages**
- zed-ros-wrapper   
- xsens_mti_driver   
- zed-ros-examples (optional)   
        

### Installation of `robot_orienteering` package
This package contains the baseline node/s for the robot orienteering.

In order to install the package, clone the [robot orienteering github repo](https://github.com/UT-ADL/robot_orienteering/tree/main) inside the `src` directory.

Navigate yourself to the `src` directory, and run the following commands:   
```
git clone https://github.com/UT-ADL/robot_orienteering.git
cd ..
catkin build
source devel/setup.bash
```

## Running the baseline node/s

The launch file `jakal_launch.launch` inside the launch directory runs all necessary nodes from the respective pakages. To run this launch file, from the catkin workspace root directory, run the command:   
```
roslaunch robot_orienteering jackal_launch.launch <arg1> <arg2> ...
```

There is a whole list of arguments that can be passed on to this launch file. The main baseline(robot navigation) node takes in some arguments based on which the beahavor of the node might differ, These arguments include:   
- `goal_image_dir`: Points to the directory where all the goal images that form the track for the robot to follow sequentially. Goal images are located inside this directory, namely: "Goal_1.jpg", "Goal_2.jpg", ...   
- `confg_dir_path`: Points to the directory where all configuration files are located   
- `local_model_type`: Sets which local planner model to use if using a neural network based waypoint proposal   
- `local_model_path`: Sets the path to the ONNX model file that will be used (local planner model)   
- `global_model_path`: Sets the path to the global planner ONNX model file which computes the goal heuristic based on robot's current position and goal position   
- `map_path`: Points to the directory where all kinds of maps are located (orienteering, baseelev, etc.)   
- `map_name`: Sets the basename of the map   
- `fps`: Sets the fps for camera (runs either in 15 fps or 4 fps)   
- `timer_frequency`: Sets the frequency in Hz for running the planner and goal heuristic computations   
- `record_video`: If input, records the entire visualization window into a mp4 viedo file. An entire absolute filepath needs to be input if recording the video   
- `nomad_conditioning_threshold`: Sets the threshold value for conditioning according to NoMaD's temporal distance predictions   
- `gps_conditioning_threshold`: Sets the distance threshold for gps distance-based goal conditioning   
- `use_nomad`: If True, runs NoMaD to propose the potential waypoints. Else, uses 5 fixed trajectories for the robot to follow   
- `use_gps`: If True, uses gps distance-based conditioning. Else, uses NoMaD's temporal distance predictions to condition the goal   
- `use_laser`: If True, uses the laser scan projection retrieved from the ZED camera's depth map fo collision avoidance. Else, the collision avoidance is absent   
- `use_global_planner`: If True, uses the global planner model to compute the goal heuristic for proposed waypoints. Else, uses euclidean distance-based goal heuristic   
- `initial_heading`: Sets the initial gps-heading (angle w.r.t. Magnetic North)
- `node_name`: Sets which node to run (can be useful when there are multiple nodes)

In general, the navigation node can be run without even setting these arguments manually, as they are launched with some default values. However, these can be tweaked while running the launch file.

#### While playing with ros bag (off-policy)
Before deploying the baseline(robot navigation) node on the robot straight away, it is a very good practice to validate how the node works while playing with a ros bag where all required messges in the correct topics are published. This shows the shortcomings of the code and also acts as a debgging tool.   

In order to run the node with a ros bag, the following arguments must be set (remaining can be set as default):
- `play_bag`: If True, sets the condition to run with a bag file   
- `bag_filepath`: Sets the path to the bag file that will be played   

#### While running live (on-policy)
Once it is validated that the node runs smoothly with the bag file, it is worth trying to see how well it itegrates with te robot in real-time. To do that, the argument `play_bag` must be set to False.   

While running the nodes live, it lanches 4 things:
1. `zed camera node` to run the ZED camera in 15 fps
2. `xsens node` to run the Xsens MTi-G710 GNSS/INS device to get the INS fused GPS coordiantes
3. `deadman switch node` which acts as a software emergency switch which enables the safety driver to take over manual control if needed
4. `robot navigation node` which subscribes to all reuired messages and lets the robot navigte through all goal points


## Examples running the launch file with different settings (arguments)

### Example using fixed trajectories for waypoints proposal and using the euclidean distance-based heuristic with ros bag
`roslaunch robot_orienteering jackal_launch.launch play_bag:=True bag_filepath:=<absolute path to the bag file>`

### Example using fixed trajectories for waypoints proposal and using the global planner model on-policy
`roslaunch robot_orienteering jackal_launch.launch use_global_planner:=True use_nomad:=True`

### Example using NoMaD for waypoints proposal and using the  euclidean distance-based heuristic on-policy
`roslaunch robot_orienteering jackal_drive.launch use_global_planner:=True use_nomad:=True`

### Example using fixed trajectories for waypoints proposal nd using the euclidean distance-based heuristic on-policy recording the live visualization
`roslaunch robot_orienteering jackal_drive.launch record_video:=<absolute path where the video mp4 file should be saved>`


