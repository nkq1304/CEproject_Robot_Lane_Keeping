import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh

# Đọc hình ảnh từ tệp
vidcap = cv2.VideoCapture('japan_driving.mp4')
success, image = vidcap.read()
while success:
    success, image = vidcap.read()
    frame = cv2.resize(image, (640,480))
    	# src = np.float32(
		# [[100, 395],
		# [500, 395],
		# [290, 230],
		# [325, 230]])
    
    # image = cv2.imread('cut_images_from_japan/frame_0049.jpg')
    tr=(320,330)
    bl=(90,478)
    tl =(295,330)
    br=(550,478)

    cv2.circle(frame,tl,5,(0,0,255),-1)
    cv2.circle(frame,bl,5,(0,0,255),-1)
    cv2.circle(frame,tr,5,(0,0,255),-1)
    cv2.circle(frame,br,5,(0,0,255),-1)
    
    pts1 = np.float32([tl,bl,tr,br])
    pts2=np.float32([[0,0],[0,480],[640,0],[640,480]])
    
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    transformed_framed = cv2.warpPerspective(frame, matrix,(640,480))

    # Hiển thị hình ảnh bằng plt
    cv2.imshow("frame",frame) 
    cv2.imshow("transformed_framed",transformed_framed)
    cv2.waitKey(1)
    
    if cv2.waitKey(1) ==27:
        break


