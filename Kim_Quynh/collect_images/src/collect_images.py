#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
import cv2 
from cv_bridge import CvBridge, CvBridgeError
import os
import numpy as np

class ImageSubscriber:
    def __init__(self):
        self.bridge=CvBridge()
        # self.image_subscriber = rospy.Subscriber("/camera/image/compressed", CompressedImage, self.image_callback)
        self.image_subscriber = rospy.Subscriber("/camera/image/compressed", CompressedImage, self.find_yellow_lane_line)
        self.path_to_save_image = "/home/kimquynh/catkin_ws/src/collect_images/images"
        self.image_counter = 0

    def image_callback(self, data):
        cv_image = self.bridge.compressed_imgmsg_to_cv2(data)

        # Hiển thị hình ảnh bằng OpenCV
        cv2.imshow("TurtleBot3 Camera", cv_image)
        cv2.waitKey(1)

        #Luu hinh anh vao folder
        image_filename = os.path.join(self.path_to_save_image, f"image_{self.image_counter:d}.png")
        if cv2.imwrite(image_filename, cv_image):
            print('Luu hinh anh thanh cong')
            self.image_counter += 1
        else:
            print('Lưu hình ảnh that bai') 

    def find_yellow_lane_line(self, data):
        cv_image = self.bridge.compressed_imgmsg_to_cv2(data)
        yellow_lower = np.array([22,93,0], np.uint8)
        yellow_upper = np.array([45,255,255], np.uint8)

        hsv_image=cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

        bitwise_yellow = cv2.bitwise_and(cv_image,cv_image, mask=yellow_mask)

        contours, hierachy = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area >300):
                x,y,w,h = cv2.boundingRect(contour)
                cv_image=cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255, 255, 0), 2)



        cv2.imshow('Yellow Detection', cv_image)
        cv2.waitKey(1)

        image_filename = os.path.join(self.path_to_save_image, f"image_{self.image_counter:d}.png")
        if cv2.imwrite(image_filename, cv_image):
            print('detect mau vang thanh cong')
            self.image_counter += 1
        else:
            print('Lưu hình ảnh thất bại') 

def main():
    rospy.init_node('image_subscriber',anonymous=True)
    image_subscriber =ImageSubscriber()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()