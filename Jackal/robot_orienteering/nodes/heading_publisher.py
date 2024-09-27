import numpy as np

import rospy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry

import tf
import tf.transformations

class HeadingPublisher:

    def __init__(self):

        # self.initial_heading = rospy.get_param('~initial_heading')
        # self.current_heading = rospy.get_param('~initial_heading')
        self.initial_heading = rospy.get_param('~initial_heading')
        self.initial_heading = self.limit_angle(self.initial_heading)

        self.current_heading = self.initial_heading

        rospy.loginfo(f"initial heading: {self.initial_heading}")

        initial_odom_msg = rospy.wait_for_message(topic='/odometry/filtered',
                                                  topic_type=Odometry)
        
        inital_orientation = initial_odom_msg.pose.pose.orientation
        initial_quat = np.array([inital_orientation.x, inital_orientation.y, inital_orientation.z, inital_orientation.w])

        # angle w.r.t. Odom frame's x-axis
        self.initial_yaw = tf.transformations.euler_from_quaternion(initial_quat)[2]
        self.initial_yaw_degrees = np.rad2deg(self.initial_yaw)

        # gps heading for Odom frame's x-axis
        # Once the GPS heading for the Odom frame's x-axis (or roll axis) is known,
        # we compute the GPS heading w.r.t. Odom frame's GPS heading (it becomes more trivial)
        self.odom_heading = self.initial_heading + self.initial_yaw_degrees
        self.odom_heading = self.limit_angle(self.odom_heading)

        rospy.loginfo(f"Current gps heading: {self.odom_heading}")
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

        # current_gps_heading = self.odom_heading - current_yaw
        # current_gps_heading = self.limit_angle(current_gps_heading)

        # current_heading = Float32()
        # current_heading.data = current_gps_heading
        # self.current_heading_publisher.publish(current_heading)   

        # rospy.loginfo(f"Current gps heading: {current_gps_heading}")

        # if self.initial_yaw is None:
        #     self.initial_yaw = current_yaw

        if self.prev_yaw is None:
            self.prev_yaw = current_yaw
            self.prev_yaw = self.limit_angle(self.prev_yaw)
        
        # delta_yaw = current_yaw - self.initial_yaw
        delta_yaw = current_yaw - self.prev_yaw
        
        if abs(delta_yaw) >= 0.1:
            self.current_heading -= delta_yaw

        # self.current_heading -= delta_yaw

        self.current_heading = self.limit_angle(self.current_heading)

        # if self.current_heading < -180:
        #     self.current_heading += 360
        # elif self.current_heading > 180:
        #     self.current_heading -= 360

        current_heading = Float32()
        current_heading.data = self.current_heading
        self.current_heading_publisher.publish(current_heading)

        rospy.loginfo(f"Current gps heading: {self.current_heading}")

        self.prev_yaw = current_yaw
        
        
if __name__ == "__main__":
    rospy.init_node("gps_heading_publisher", log_level=rospy.INFO)
    node = HeadingPublisher()    
    rospy.spin()