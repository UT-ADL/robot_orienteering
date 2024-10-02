import numpy as np

import rospy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry

import tf
import tf.transformations

class HeadingPublisher:

    def __init__(self):

        
        self.initial_heading = rospy.get_param('~initial_heading')
        self.initial_heading = self.limit_angle(self.initial_heading)
        
        self.current_heading_publisher = rospy.Publisher('/current_heading',
                                                         Float32,
                                                         queue_size=1)
        
        rospy.Subscriber('/zed/zed_node/odom', Odometry, self.odom_callback, queue_size=1)


    def limit_angle(self, angle):
        return (angle + 180) % 360 - 180


    def odom_callback(self, msg):
        # Convert quaternion to euler angles
        quaternion = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)

        # Extract yaw (z-axis rotation) and convert to degrees
        yaw = euler[2]  # Yaw is the third element (index 2)
        yaw_degrees = np.degrees(yaw)

        gps_heading = self.initial_heading - yaw_degrees
        gps_heading = self.limit_angle(gps_heading)

        current_heading = Float32()
        current_heading.data = gps_heading
        self.current_heading_publisher.publish(current_heading)

        
if __name__ == "__main__":
    rospy.init_node("gps_heading_publisher", log_level=rospy.INFO)
    node = HeadingPublisher()    
    rospy.spin()