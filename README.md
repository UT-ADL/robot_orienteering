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

## Running the baseline nodes

### Launching the robot
Once all packages are installed, you are ready to deploy the baseline node on the robot in real-time.   

To launch it on-policy, in one of the terminal windows please run the following command:   
  
```
roslaunch robot_orienteering jackal_launch.launch
```

This launch file launches:
- `zed camera node` to run the ZED camera in 15 fps
- `xsens node` to run the Xsens MTi-G710 GNSS/INS device to get the INS fused GPS coordiantes
- `heading publiher node` to update the GPS heading using the ZED camera's visual odometry
- `deadman switch node` which acts as a software emergency switch which enables the safety driver to take over manual control if needed

This launch file takes one command line argument:
- `initial_heading`: sets the initial heading to the user-input value. If not passed, sets the initial heading to zero (Magentic North)

This launch file sets the robot to publish all necessary messges in corresponding topics for the baseline node to subsribe and act accodingly

### Running the baseline node
In another terminal window, please run the following command:
```
roslaunch robot_orienteering jackal_drive.launch
```

This launch file runs the `robot_drive` node which subscribes to the required topics and retrieves the camera image, INS fused GPS coordiantes, current GPS heading and the state of the deadman switch.

By default, this node proposes 5 fixed trajectories and uses the euclidean distance between the waypoints and the goal (GPS-based distance) and chooses the one closest to the goal. Once chosen, then it navigates towards that waypoint in each iteration.   

However, it can be set as command line argument on how to propose the trajectories and which heuristic to use.   

The list of command line arguments that can be passes on while running this launch files is listed below:   

#### Waypoints proposal

By default, it proposes 5 fixed trajectories to follow, but it can be changed by passing different argeuments.   

- `use_nomad`: If `True`, then it uses nomad to propose trajectories   
`roslaunch robot_orienteering jackal_drive.launch use_nomad:=True`

#### Goal huristic

- `use_global_planner`: If `True`, then it uses the global planner heuristic model and choose the waypoint which has the highest probability based on the global planner predictions. Else, it uses the euclidean distance between the waypoints and the goal, and chooses the waypoint which is closest to the goal.
`roslaunch robot_orienteering jackal_drive.launch use_global_planner:=True`

#### Example using NoMaD for waypoints proposal and using the global planner heuristic model
`roslaunch robot_orienteering jackal_drive.launch use_global_planner:=True use_nomad:=True`

#### Additional utilities
You can also record the live camera feed with additional visulizations using the `record_video` argument.   
Example usage:   
`roslaunch robot_orienteering jackal_drive.launch record_video:=/home/test_video.mp4`


