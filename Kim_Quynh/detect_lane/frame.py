import cv2
import math
import numpy as np
import os

def frame(video_input_path, output_directory):
    cap = cv2.VideoCapture(video_input_path)

    frameRate = cap.get(5)
    x = 1
    while (cap.isOpened()):
        frameId = cap.get(1)
        ret, frame = cap.read()
        if (ret !=True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = f"{output_directory}/Gazebo_{int(x)}.jpg"
            x += 1
            cv2.imwrite(filename, frame)
        
    cap.release()
    print("FRAME Done!")
    
if __name__ == "__main__":  
    video_input_path = "test_videos/gazebo_vid/detect.mp4"
    output_directory = "test_images/gazebo_img"
    frame(video_input_path,output_directory)
    