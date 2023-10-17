import numpy as np
import cv2
import os
import combined_thresh as ct
import matplotlib.pyplot as plt

def frame_processor(image):
    # get bird's eye view
    result = bird_eye_view(image)

    # apply combined threshold
    combined, abs_bin, mag_bin, dir_bin, hls_bin = ct.combined_thresh(result)

    result = combined

    return result

def bird_eye_view(image):
    """
    Apply perspective transform to get bird's eye view.
    Parameters:
        image: The input test image.
    """
    # get image size
    image_size = (image.shape[1], image.shape[0])
    # source points of the image 640 x 480
    src = np.float32([[80, 300], [400, 300],
                        [640, 480], [0, 480]])
    # destination points of the image 640, 480
    dst = np.float32([[0, 0], [640, 0],
                        [640, 480], [0, 480]])
    # get perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # get bird's eye view
    warped = cv2.warpPerspective(image, M, image_size)

    cv2.imshow('warped', warped)
    return warped

# Get all image in image folder
image_list = os.listdir('image')

for image_name in image_list:
    image = cv2.imread('image/' + image_name)

    # resize image to 640x480
    image = cv2.resize(image, (640, 480))
    result = frame_processor(image)

    # plt.imshow(image)
    # plt.show()

    cv2.imshow('image', image)
    cv2.imshow('result', result)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break



cv2.waitKey(0)