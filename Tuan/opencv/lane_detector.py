import numpy as np
import cv2
import os
import combined_thresh as ct
import matplotlib.pyplot as plt
import line_fit as lf
import show_image

def frame_processor(image):
    # get bird's eye view
    result, m, m_inv = bird_eye_view(image)

    # apply combined threshold
    combined, abs_bin, mag_bin, dir_bin, nor_bin = ct.combined_thresh(result)

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

    src = np.float32([
        (400,300),    
        (530,300), 
        (635,480),  
        (160,480)
    ])

    dst = np.float32([
        (100, 0),
        (854 - 100, 0),
        (854 - 100, 480),
        (100, 480),
    ])
    # get perspective transform matrix
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    # get bird's eye view
    warped = cv2.warpPerspective(image, m, image_size)

    # show the warped with scale is half
    show_image.show_image('warped', warped)

    return warped, m, m_inv

# image_list = os.listdir('Tuan/image')
# image_list = sorted(image_list, key=lambda x: int(x.split('.')[0]))

# for image_name in image_list:
#     image = cv2.imread('Tuan/image/' + image_name)

#     # convert image to RGB
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # resize image to 640x480
#     image = cv2.resize(image, (640, 480))
#     result = frame_processor(image)

#     cv2.imshow('image', image)
#     cv2.imshow('result', result)
#     cv2.imwrite('Tuan/result/' + image_name, result)

#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

cap = cv2.VideoCapture('Tuan/video/video_2.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('Tuan/result/video_2.avi', fourcc, 20.0, (854, 480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        result = frame_processor(frame)
        out.write(result)

        # cv2.imshow('frame',frame)

        show_image.show_image('result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cv2.waitKey(0)