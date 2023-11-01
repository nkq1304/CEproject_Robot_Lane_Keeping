#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def main():
    rospy.init_node('image_publisher', anonymous=True)
    image_pub = rospy.Publisher('Image_Node', Image, queue_size=30)
    bridge = CvBridge()

    rate = rospy.Rate(30)  # Tốc độ xuất bản là 30 hình ảnh/giây

    while not rospy.is_shutdown():
        image_msg = get_simulation_image()  
        if image_msg is not None:
            try:
                # Chuyển đổi hình ảnh sang định dạng OpenCV
                cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
                image_pub.publish(image_msg)
            except CvBridgeError as e:
                print(e)

        rate.sleep()

def get_simulation_image():
    image_msg = rospy.wait_for_message('/camera/image', Image)
    return image_msg
    return None 

if __name__ == '__main__':
    main()

