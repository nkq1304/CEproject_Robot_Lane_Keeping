import numpy as np
import cv2
import os
import combined_thresh as ct
import matplotlib.pyplot as plt
import line_fit as lf

def frame_processor(image):
    # get bird's eye view
    result, m, m_inv = bird_eye_view(image)

    # apply combined threshold
    combined, abs_bin, mag_bin, dir_bin, hls_bin = ct.combined_thresh(result)

    result = combined

    left_fit, right_fit, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = lf.line_fit(result)
    
    if left_fit is None or right_fit is None:
        return image
    
    lf.draw_poly_lines(result, left_fit, right_fit, nonzerox, nonzeroy)
    left_curve, right_curve = lf.calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

    bottom_y = image.shape[0] - 1
    bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
    bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
    vehicle_offset = image.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

    result = lf.visualize(image, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)

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
    src = np.float32([[240, 300], [400, 300],
                        [640, 480], [0, 480]])
    # destination points of the image 640, 480
    dst = np.float32([[0, 0], [640, 0],
                        [640, 480], [0, 480]])
    # get perspective transform matrix
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    # get bird's eye view
    warped = cv2.warpPerspective(image, m, image_size)

    return warped, m, m_inv

# Get all image in image folder
image_list = os.listdir('Tuan/image')
image_list = sorted(image_list, key=lambda x: int(x.split('.')[0]))

for image_name in image_list:
    image = cv2.imread('Tuan/image/' + image_name)

    # convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize image to 640x480
    image = cv2.resize(image, (640, 480))
    result = frame_processor(image)

    cv2.imshow('image', image)
    cv2.imshow('result', result)
    cv2.imwrite('Tuan/result/' + image_name, result)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cv2.waitKey(0)