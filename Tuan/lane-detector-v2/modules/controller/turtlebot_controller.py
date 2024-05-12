import rospy
import numpy as np
import matplotlib.pyplot as plt

from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

from utils.lane_line import LaneLine


class TurtlebotController:
    def __init__(self, cfg: dict):
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        self.max_vel = cfg["max_vel"]
        self.Kp = cfg["Kp"]
        self.Kd = cfg["Kd"]

        self.lastError = 0
        self.errors = []

        rospy.on_shutdown(self.stop)

    def follow_lane(self, lane: LaneLine):
        if lane is None:
            self.stop()
            return

        error = lane.dist

        self.errors.append(error)

        angular_z = self.Kp * error + self.Kd * (error - self.lastError)
        self.lastError = error

        twist = Twist()
        # twist.linear.x = 0.1
        twist.linear.x = self.max_vel * ((1 - abs(error) / 500) ** 2.2)
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = max(angular_z, -2.0) if angular_z < 0 else min(angular_z, 2.0)

        self.pub_cmd_vel.publish(twist)

    def stop(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")

        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0

        # plt.plot(np.convolve(self.errors, np.ones(10)/10, mode='valid'))
        # plt.xlabel('time')
        # plt.ylabel('error')
        # plt.axhline(y=0, color='r', linestyle='-')

        # max_y_value = max(abs(min(self.errors)), abs(max(self.errors)))
        # plt.ylim(-max_y_value, max_y_value)

        # plt.show()

        self.pub_cmd_vel.publish(twist)

    def save_errors(self):
        np.save("errors.npy", self.errors)

    def main(self):
        rospy.spin()
