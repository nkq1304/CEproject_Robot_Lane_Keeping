# import required module 
import cv2 
import matplotlib.pyplot as plt 

# read image 
image = cv2.imread('test_images/real_img/frame_1.jpg') 

# call imshow() using plt object 
plt.imshow(image) 

# display that image 
plt.show() 
