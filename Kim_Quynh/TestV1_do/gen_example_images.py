'''
Generate example images to illustrate different pipeline stages' outputs
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
# from line_fit import line_fit, viz2, calc_curve, final_viz
from keypoints import masked_img

# Read camera calibration coefficients
with open('calibrate_camera.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

# Create example pipeline images for all test images
image_files = os.listdir('cut_images_from_japan')
for image_file in image_files:
	out_image_file = image_file.split('.')[0] + '.png'  # write to png format
	img = mpimg.imread('cut_images_from_japan/' + image_file)

	# # Undistort image
	# img = cv2.undistort(img, mtx, dist, None, mtx)
	# plt.imshow(img)
	# plt.savefig('example_images/undistort_' + out_image_file)

	# # Thresholded binary image
	# img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)
	# plt.imshow(img, cmap='gray', vmin=0, vmax=1)
	# plt.savefig('example_images/binary_' + out_image_file)

	# Perspective transform
	img, binary_unwarped, m, m_inv = perspective_transform(img)
	plt.imshow(img, cmap='gray', vmin=0, vmax=1)
	plt.savefig('result/images/warped_' + out_image_file)
 

