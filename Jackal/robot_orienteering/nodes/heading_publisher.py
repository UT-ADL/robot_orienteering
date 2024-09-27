import numpy as np

import rospy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry

import tf
import tf.transformations

class HeadingPublisher:

    def __init__(self):

        
        self.initial_heading = rospy.get_param('~initial_heading')
        self.current_heading = self.limit_angle(self.initial_heading)

        rospy.loginfo(f"Current gps heading: {self.current_heading}")
        print("----------------------------------------------------")

        self.initial_yaw = None
        self.prev_yaw = None
        
        self.current_heading_publisher = rospy.Publisher('/current_heading',
                                                         Float32,
                                                         queue_size=1)
        
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback, queue_size=1)


    def limit_angle(self, angle):
        return (angle + 180) % 360 - 180

    def odom_callback(self, msg):

        current_quaternion = np.array([msg.pose.pose.orientation.x, 
                                       msg.pose.pose.orientation.y, 
                                       msg.pose.pose.orientation.z, 
                                       msg.pose.pose.orientation.w])  
        
        current_yaw = tf.transformations.euler_from_quaternion(current_quaternion)[2] 
        current_yaw = np.rad2deg(current_yaw)
        current_yaw = self.limit_angle(current_yaw)

        if self.prev_yaw is None:
            self.prev_yaw = current_yaw
            self.prev_yaw = self.limit_angle(self.prev_yaw)
        
        delta_yaw = current_yaw - self.prev_yaw
        
        if abs(delta_yaw) >= 0.1:
            self.current_heading -= delta_yaw

        self.current_heading = self.limit_angle(self.current_heading)

        current_heading = Float32()
        current_heading.data = self.current_heading
        self.current_heading_publisher.publish(current_heading)

        rospy.loginfo(f"Current gps heading: {self.current_heading}")

        self.prev_yaw = current_yaw
        
        
if __name__ == "__main__":
    rospy.init_node("gps_heading_publisher", log_level=rospy.INFO)
    node = HeadingPublisher()    
    rospy.spin()