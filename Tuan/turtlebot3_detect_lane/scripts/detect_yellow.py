#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import sys

class detect_yellow:
    def __init__(self):
        self.video = cv2.VideoCapture("./video/test.mp4")
        self.detect(self.video)

    def detect(self, video):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        res_video = cv2.VideoWriter('./video/res_video.mp4', fourcc, 30, (800, 600))

        while True:
            ret, frame = video.read()
            
            if ret == True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([30, 255, 255])
                mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

                # Add rectangle frame on the yellow coin
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w > 20 and h > 20:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                res_video.write(frame)
                cv2.imshow("Image window", frame)
                cv2.waitKey(15)
                # Add frame to the video
            else:
                break

        res_video.release()

def main(args):
    dy = detect_yellow()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
