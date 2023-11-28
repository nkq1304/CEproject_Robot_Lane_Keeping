import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import collections
import math
import os
# from moviepy.editor import VideoFileClip

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

frame_capture = collections.deque([], 1)

def draw_lines(img, lines, color = [0, 255, 0], thickness = 5):
    
    # mức độ cho phép đối với slope
    difference = 0.01
    maximumSlope = 0.5
    
    # defaults for left lane
    left_slope = 0
    left_points = []
    left_set = 1
    
    # defaults for right lane
    right_slope = 0
    right_points = []
    right_set = 1
    
    """
    Identify negative and postive slopes and classify them as left and right lanes
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2 - y1)/(x2 - x1)
            #slope is negative for left lane, positive for right lane because (0,0) starts at top-left corner of image
            #but that can be avoided by using absolute value
            if slope < 0:
                # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # print(image_path)
                left_slope = left_slope + (slope - left_slope) / left_set
                if (np.absolute(left_slope - slope) <= difference):
                    left_points.append((x1, y1))
                    left_points.append((x2, y2))
                    # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
                if(len(left_points) == 0):
                    if (np.absolute(left_slope - slope) > difference and np.absolute(left_slope - slope) < 0.2):
                        left_points.append((x1, y1))
                        left_points.append((x2, y2))
                        # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
                left_set = left_set + 1
                # print(np.absolute(left_slope - slope))
            else:
                # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                right_slope = right_slope + (slope - right_slope) / right_set
                if np.absolute(right_slope - slope) <= difference:
                    right_points.append((x1, y1))
                    right_points.append((x2, y2))
                    # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
                right_set = right_set + 1
                
    right_intercept = 0  # Add default values here
    left_intercept = 0   # Add default values here
    
    if len(right_points) > 0 and len(left_points) > 0:
        # for left lane
        [xx, yy, x, y] = cv2.fitLine(np.array(left_points, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
        left_slope = yy / xx
        left_intercept = y - (left_slope * x)

        # for right lane
        [xx, yy, x, y] = cv2.fitLine(np.array(right_points, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
        right_slope = yy / xx
        right_intercept = y - (right_slope * x)
    
        frame_capture.append((right_intercept, right_slope, left_intercept, left_slope))
    
    if len(frame_capture) > 0:
        average = np.sum(frame_capture, -3) / len(frame_capture)
        right_intercept = average[0]
        right_slope = average[1]
        left_intercept = average[2]
        left_slope = average[3]
    
    y_right_bottom_corner = img.shape[0] 
    y_left_top_corner = int(y_right_bottom_corner / 1.65) 
    
    left_lane_x1 = (y_right_bottom_corner - left_intercept) / left_slope
    left_lane_x2 = (y_left_top_corner - left_intercept) / left_slope
    
    right_lane_x1 = (y_right_bottom_corner - right_intercept) / right_slope
    right_lane_x2 = (y_left_top_corner - right_intercept) / right_slope
    
    right_lane_x1 = int(np.squeeze(right_lane_x1))
    left_lane_x1 = int(np.squeeze(left_lane_x1))
    left_lane_x2 = int(np.squeeze(left_lane_x2))
    right_lane_x2 = int(np.squeeze(right_lane_x2))
    
    cv2.line(img, (left_lane_x1, y_right_bottom_corner), (left_lane_x2, y_left_top_corner), (0,0,255), thickness)
    cv2.line(img, (right_lane_x1, y_right_bottom_corner), (right_lane_x2, y_left_top_corner), (0,255,0), thickness)
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α, β, γ):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def main_execution(image):
    # Step 1: Convert colorscaled image to grayscaled image using the grayscale(img) function
    gray_img = grayscale(image)

    # Step 2 : Define the kernel size for filter and apply Gaussian smoothing filter over the image
    kernel_size = 7
    blurred_gray = cv2.GaussianBlur(gray_img,(kernel_size, kernel_size),0)

    # Step 3 : Use Canny edge detection algorithm after defining the thresholds
    low_threshold = 50
    high_threshold = 150
    edges_image = cv2.Canny(blurred_gray, low_threshold, high_threshold)

    # Step 4 : Create a 4-sided polygon for the mask 
    # with real lane
    poly_vertices = np.array([[(110,360),(250, 280), (380, 280), (500,360)]], dtype=np.int32)
    # with gazebo
    # poly_vertices = np.array([[(210,360),(300, 180), (350, 1800), (500,360)]], dtype=np.int32)
    masked_edges = region_of_interest (edges_image, poly_vertices)


    # Step 5 : Define the Hough transform parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 5  # minimum number of votes (intersections in Hough grid cell) 5
    min_line_length = 20 #minimum number of pixels making up a line 60
    max_line_gap = 20   # maximum gap in pixels between connectable line segments

    # # Step 6 : Run Hough transform on edge detected image
    hough_image = hough_lines(masked_edges, rho, theta, threshold,min_line_length, max_line_gap)
    final_image = weighted_img(hough_image, image, α=0.8, β=0.5, γ=0.)
    
    output_path = os.path.join(output_folder, f"detectd_{image_file}")
    cv2.imwrite(output_path, final_image)
    return final_image
    
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = main_execution(image)
    return result

if __name__ == "__main__":  
    input_folder = "test_images/real_img"
    
    # houghline para
    output_folder = "output/real_lane"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lấy danh sách các tệp trong thư mục đầu vào
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    for image_file in image_files:
        # Đọc ảnh từ tệp
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)
        process_image(img)
    print("TEST Done")
    
    
    ############################################# VIDEO ############################## 
    # # with real lane
    # # video_capture = cv2.VideoCapture("test_videos/japan_driving.mp4")
    # # with gazebo
    # video_capture = cv2.VideoCapture("test_videos/Real_vid/japan_driving.mp4")

    # # Xác định các thông số của video đầu ra
    # frame_width = int(video_capture.get(3)) 
    # frame_height = int(video_capture.get(4)) 

    # size = (frame_width, frame_height)

    # # Specify the full path to the output video file
    # output_path = 'output/videos/real_output.mp4'
    # result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, size)

    # # Kiểm tra xem việc mở video có thành công không
    # if not video_capture.isOpened():
    #     print("Không thể mở video.")

    # while True:
    #     ret, frame = video_capture.read()

    #     if not ret:
    #         print("Hết video hoặc có lỗi khi đọc video.")
    #         break
    #     if ret == True:  
    #         # Assuming main_execution is a function that processes each frame
    #         processed_frame = process_image(frame)
    #         result.write(processed_frame) 

    #         cv2.imshow('processed_video', processed_frame) 

    #         # Press S on keyboard  
    #         # to stop the process 
    #         if cv2.waitKey(10) & 0xFF == ord('s'): 
    #             break

    #     # Break the loop 
    #     else: 
    #         break

    # video_capture.release() 
    # result.release() 
    # # Closes all the frames 
    # cv2.destroyAllWindows()  
    # print("The video was successfully saved at:", output_path)