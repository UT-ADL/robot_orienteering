from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    scripts=['nodes/deadman_switch.py',
             'nodes/heading_publisher.py',
             'nodes/robot_drive.py'],
    packages=['global_planner', 'utils', 'waypoint_planner'],
    package_dir={'': 'src'},
    requires= ['rospy']
)

setup(**setup_args)