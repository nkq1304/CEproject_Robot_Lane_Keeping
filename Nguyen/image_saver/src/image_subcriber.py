#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

save_dir = "/home/khoinguyen/DA2023/image_saver" 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

bridge = CvBridge()
image_count = 0

existing_images = os.listdir(save_dir)
existing_images = [f for f in existing_images if f.startswith("image_")]
if existing_images:
    image_count = max([int(f.split("_")[1].split(".")[0]) for f in existing_images]) + 1
else:
    image_count = 0
    
def image_callback(msg):
    global image_count
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        # Xử lý hình ảnh ở đây
        image_name = f"image_{image_count}.jpg"
        image_path = os.path.join(save_dir, image_name)
        cv2.imwrite(image_path, cv_image)
        print(f"Saved {image_name} successfully")
        image_count += 1
    except CvBridgeError as e:
        print(e)

def main():
    rospy.init_node('image_subscriber', anonymous=True)
    rospy.Subscriber('Image_Node', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()

