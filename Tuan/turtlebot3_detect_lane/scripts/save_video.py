#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import sys

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class save_video:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.callback)
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter("videos/original.mp4", self.fourcc, 30, (1280, 720))

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = cv2.resize(cv_image, (1280, 720))
            self.out.write(cv_image)

        except CvBridgeError as e:
            print(e)

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(1)


def main(args):
    save_video()
    rospy.init_node("get_image", anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
