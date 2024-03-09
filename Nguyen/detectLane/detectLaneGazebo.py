import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from scipy.interpolate import splprep, splev
# from scipy.optimize import curve_fit
# Setup directory to save frame
save_in_dir = "D:/DATN/detectLaneGazebo/current/input_frame" 
if not os.path.exists(save_in_dir):
    os.makedirs(save_in_dir)
# save_in_dir2 = "D:/HK231/DAMHKTMT/detectLaneGazebo/detect_input_frame" 
# if not os.path.exists(save_in_dir2):
#     os.makedirs(save_in_dir2)
save_out_dir = "D:/DATN/detectLaneGazebo/current/output_frame" 
if not os.path.exists(save_out_dir):
    os.makedirs(save_out_dir)

# Set global variable
last_left_theta = None # Theta of last left line of previous 5 frames
last_left_line = None # Last left line of previous 5 frames
last_right_theta = None # Theta of last right line of previous 5 frames
last_right_line = None # Last right line of previous 5 frames
frame_num = 5 # Max frame's array number
line_thickness = 6 # Thickness of 
thresh_theta = 10 # Threshold theta in degree
scale_thresh = 2
frame_height = None
frame_width = None
font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 0.65
color = (255, 0, 0) 
thickness = 1
def findInterWithXAxis(line):
    if line is not None:
        x1, y1, x2, y2 = line[0]
        # Find linear equation of left line
        if x1 == x2:
            return True, 0, 0
        else:
            m1 = (y2 - y1) / (x2 - x1)
            b1 = y1 - m1 * x1
            return False, ((frame_height - b1) / m1), frame_height

def calTheta(line):
    x1, y1, x2, y2 = line[0]
    return np.arctan2(y2 - y1, x2 - x1)

def replaceLane(line, theta, approx_theta_arr, last_line, last_theta, ef_num):
    kind_of_error = []
    if(last_theta == None): 
        # If these are the first 5 frames, the default are that the frames with 
        # the largest theta are approximately the same as the lane and approx_theta_arr 
        # is contain correct lanes 
        for i in range(0, len(line)):
            if line[i] is not None:
                theta_temp = calTheta(line[i])
                if theta_temp not in approx_theta_arr:
                    if i == 0:
                        # Because this's first frame, so we need to find 
                        # the first correct lane to replace it
                        pos = 0
                        for j in range (1 , len(line)):
                            if line[j] is not None and calTheta(line[j]) in approx_theta_arr:
                                pos = approx_theta_arr.index(calTheta(line[j]))
                        line[i] = line[pos]
                    else:
                        line[i] = line[i-1]
            else:   
                if i == 0:
                    # Because this's first frame, so we need to find 
                    # the first correct lane to replace it
                    pos = 0
                    for j in range (1 , len(line)):
                        if line[j] is not None and calTheta(line[j]) in approx_theta_arr:
                            pos = approx_theta_arr.index(calTheta(line[j]))
                    line[i] = line[pos]
                else:
                    line[i] = line[i-1]
        last_line = line[len(line) - 1]
        last_theta = calTheta(line[len(line) - 1])
    else:
        if (ef_num > int(frame_num / 2)) or (ef_num <= int(frame_num / 2) and ef_num > 0):
            # If number of error frame larger than half of number of max frame:
            # Accept these error frame are correct lane
            square_angle = 90 * np.pi / 180
            for i in range (frame_num - 1, len(line)):
                if line[i] is not None:
                    # print("Replace error line %s", i)
                    divide_by_zero =  False
                    divide_by_zero, x, _ = findInterWithXAxis(line[i])
                    if not np.isnan(x) or not divide_by_zero:
                        x =  int(x)
                        if (x <= frame_width and x >= 0) or (np.isnan(x)):
                            # print('In, last line', last_line[0])
                            theta_temp = np.abs(calTheta(line[i]))
                            if np.bitwise_xor(np.abs(theta_temp - last_theta) > (thresh_theta * np.pi / (scale_thresh *180)),
                                            np.abs(square_angle - theta_temp) > (thresh_theta * np.pi / (scale_thresh *180))):
                                line[i] = last_line
                                if(np.abs(theta_temp - last_theta) > (thresh_theta * np.pi / (scale_thresh *180))):
                                    kind_of_error.append(1)
                                else:
                                    kind_of_error.append(2)
                            else:
                                kind_of_error.append(0)
                        else:
                            print('Out, last line: ', last_line[0])
                            line[i] = last_line
                            kind_of_error.append(1)
                    else:
                        line[i] = last_line    
                        kind_of_error.append(3)
                else: 
                    # print('Replace not line %s', i)
                    kind_of_error.append(3)
                    line[i] = last_line
        last_line = line[len(line) - 1]
        last_theta = calTheta(line[len(line) - 1])
    return line, last_line, last_theta, kind_of_error

def detectErrorLine(line):  
    # Search for (any) lines that have a theta angle deviating 
    # significantly from the remaining lines; if found, return true.
    # Algo: 1. Get theta value and store them in list
    theta = []
    for index in range(len(line)) :
        if line[index] is not None:
            x1, y1, x2, y2 = line[index][0]
            theta1 = np.arctan2(y2 - y1, x2 - x1)
            theta.append(np.abs(theta1))
    sorted_theta = np.sort(theta) # 2. Sort theta list
    # 3. Get angles that approximate each other in such a way that the quantity of angles is maximal 
    # and these angles have to be largest angles.
    diff_theta = np.diff(sorted_theta)
    index = np.where(diff_theta > (thresh_theta * np.pi / 180))[0]
    split_theta = np.split(sorted_theta, index +1)
    max_theta_arr = np.sort(max(split_theta, key = len))
    # 4. Filter the retreived list again, as this list may still contain 
    # pairs deviating from each other larger than the threshold
    approx_theta_arr = []
    for i in range(len(max_theta_arr)):
        if np.abs(max_theta_arr[i] - max_theta_arr[0]) <= thresh_theta * np.pi / 180:
            approx_theta_arr.append(max_theta_arr[i])
        else:
            break
    # 5. If all theta angle are approximately with each other, return true. Otherwise, return false
    if(len(approx_theta_arr) == len(line)):
        return False, approx_theta_arr, theta
    else:
        return True, approx_theta_arr, theta
    

def detectLane(line, type_of_line, error_frame):
    # Function: Detect left and right lane
    global last_left_theta, last_left_line 
    global last_right_theta, last_right_line
    flag, approx_theta_arr, theta = detectErrorLine(line)
    kind_of_error = []
    # print(approx_theta_arr)
    if(type_of_line == 'Left'):
            # for i in range(len(line)):
            #     if line[i] is not None:
            #         print("Line before replace", i, line[i])
        line, last_left_line, last_left_theta, kind_of_error = replaceLane(line, theta, approx_theta_arr, last_left_line, last_left_theta, error_frame)
        # if flag == True:
        #     for i in range(len(line)):
        #         if line[i] is not None:
        #             print('Line after replace', i, line[i])
    elif (type_of_line == 'Right'):
        #     for i in range(len(line)):
        #         if line[i] is not None:
        #             print("Line before replace", i, line[i])
        line, last_right_line, last_right_theta, kind_of_error = replaceLane(line, theta, approx_theta_arr, last_right_line, last_right_theta, error_frame)
        # if flag == True:
        #     for i in range(len(line)):
        #         if line[i] is not None:
        #             print("Line after replace", i, line[i])
    return line, kind_of_error

def drawInFrame(left_line, right_line, frame_ori, frame_model):
    # if(error_frame > 0):
    #     print("Draw line left - right", left_line[0], right_line[0])
    height, _ = frame_ori.shape[:2]
    if left_line is not None and right_line is not None:
        x1, y1, x2, y2 = left_line[0]
        x3, y3, x4, y4 = right_line[0]
        contours, _ = cv2.findContours(frame_model, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_left_color = (0, 0, 255) # RED
        line_right_color = (0, 255, 0) # GREEN
        for i, contour in enumerate(contours):
            pt1_in_line = cv2.pointPolygonTest(contour, (float(x1), float(y1)), True)
            pt2_in_line = cv2.pointPolygonTest(contour, (float(x2), float(y2)), True)
            pt3_in_line = cv2.pointPolygonTest(contour, (float(x3), float(y3)), True)
            pt4_in_line = cv2.pointPolygonTest(contour, (float(x4), float(y4)), True)
            
            if pt1_in_line >= 0 and pt2_in_line >= 0:
                cv2.drawContours(frame_ori, contours, i, line_left_color, thickness=cv2.FILLED)
            if pt3_in_line >= 0 and pt4_in_line >= 0:
                cv2.drawContours(frame_ori, contours, i, line_right_color, thickness=cv2.FILLED)
        # m1 = b1 = m2 = b2 = None
        # # Find linear equation of left line
        # if x1 != x2:
        #     m1 = (y2 - y1) / (x2 - x1)
        #     b1 = y1 - m1 * x1
        # else:
        #     x1, y1, x2, y2 = last_left_line[0]
        # # Find linear equation of right line 
        # if x3 != x4:
        #     m2 = (y4 - y3) / (x4 - x3)
        #     b2 = y3 - m2 * x3
        # else:
        #     x3, y3, x4, y4 = last_right_line[0]
        # # Draw line
        # if m1 is not None:
        #     cv2.line(frame, (int((height - b1) / m1), height), (x2, y2), line_left_color, line_thickness) 
        # else:
        #     x1, y1, x2, y2 = last_left_line[0]
        #     cv2.line(frame, (x1, y1), (x2, y2), line_left_color, line_thickness) 
        # if m2 is not None:
        #     cv2.line(frame, (x3, y3), (int((height - b2) / m2), height), line_right_color, line_thickness)
        # else:            
        #     cv2.line(frame, (x3, y3), (x4, y4), line_right_color, line_thickness)

def drawErrorInFrame(frame, left, right):
    org = (25, 50)
    org = (25, 50)
    if left == 0 and right != 0:
        if right == 1:
            cv2.putText(frame, 'Error right line: Noisy line!', org, font,  fontScale, color, thickness, cv2.LINE_AA)
            print('Right line is noisy line, replace it by the last correct line')
        elif right == 2:
            cv2.putText(frame, 'Danger: Robot on right line!', org, font,  fontScale, color, thickness, cv2.LINE_AA)
        elif right == 3:
            cv2.putText(frame, 'Error right line: Line not found!', org, font,  fontScale, color, thickness, cv2.LINE_AA)
            print('Right line not found, replace it by the last correct line')
    elif left != 0 and right == 0:
        if left == 1:
            cv2.putText(frame, 'Error left line: Noisy line!', org, font,  fontScale, color, thickness, cv2.LINE_AA)
            print('Left line is noisy line, replace it by the last correct line')
        elif left == 2:
            cv2.putText(frame, 'Danger: Robot on left line!', org, font,  fontScale, color, thickness, cv2.LINE_AA)
        elif left == 3:
            cv2.putText(frame, 'Error left line: Line not found!', org, font,  fontScale, color, thickness, cv2.LINE_AA)
            print('Left line not found, replace it by the last correct line')
    elif left != 0 and right != 0:
        if left == 2:
            cv2.putText(frame, 'Danger: Robot on left line!', org, font,  fontScale, color, thickness, cv2.LINE_AA)
        elif right == 2:
            cv2.putText(frame, 'Danger: Robot on right line!', org, font,  fontScale, color, thickness, cv2.LINE_AA)
        else:
            org2 = (25,75)
            if left == 1:
                cv2.putText(frame, 'Error left line: Noisy line!', org, font,  fontScale, color, thickness, cv2.LINE_AA)
                print('Left line is noisy line, replace it by the last correct line')
                if right == 1:
                    cv2.putText(frame, 'Error right line: Noisy line!', org2, font,  fontScale, color, thickness, cv2.LINE_AA)
                    print('Right line is noisy line, replace it by the last correct line')
                elif right == 3:
                    cv2.putText(frame, 'Error right line: Line not found!', org2, font,  fontScale, color, thickness, cv2.LINE_AA)
                    print('Right line not found, replace it by the last correct line')
            elif left == 3:
                cv2.putText(frame, 'Error left line: Line not found!', org, font,  fontScale, color, thickness, cv2.LINE_AA)
                print('Left line not found, replace it by the last correct line')
                if right == 1:
                    cv2.putText(frame, 'Error right line: Noisy line!', org2, font,  fontScale, color, thickness, cv2.LINE_AA)
                    print('Right line is noisy line, replace it by the last correct line')
                elif right == 3:
                    cv2.putText(frame, 'Error right line: Line not found!', org2, font,  fontScale, color, thickness, cv2.LINE_AA)
                    print('Right line not found, replace it by the last correct line')
    # else:
    #     cv2.putText(frame, 'See two lines', org, font,  fontScale, color, thickness, cv2.LINE_AA)

def detectDrawedErrLine(frame, right_line, left_line, minDistance):
    org = (25, 100)
    isHaveErrFrame = False
    if left_line is not None and right_line is not None:
        x1, y1, x2, y2 = left_line[0]
        x3, y3, x4, y4 = right_line[0]
        m1 = b1 = m2 = b2 = None
        # Find linear equation of left line
        if x1 != x2:
            m1 = (y2 - y1) / (x2 - x1)
            b1 = y1 - m1 * x1
        else:
            x1, y1, x2, y2 = last_left_line[0]
            m1 = (y2 - y1) / (x2 - x1)
            b1 = y1 - m1 * x1
        # Find linear equation of right line 
        if x3 != x4:
            m2 = (y4 - y3) / (x4 - x3)
            b2 = y3 - m2 * x3
        else:
            x3, y3, x4, y4 = last_right_line[0]
            m2 = (y4 - y3) / (x4 - x3)
            b2 = y3 - m2 * x3
        # Find the intersection of the left and right lanes
        inter_x = (b2 - b1) / (m1 - m2)
        inter_y = m1 * inter_x + b1
        if inter_y < y2 and inter_y < y3 and inter_y <= frame_height:
            distance1 = np.sqrt((x4 - x1) *(x4 - x1) + (y4 - y1)*(y4 - y1))
            distance2 = np.sqrt((x3 - x2) *(x3 - x2) + (y3 - y2)*(y3 - y2))
            if (distance1 <= minDistance) and (distance2 <= minDistance):
                cv2.putText(frame, 'Error frame, detect again!', (25,100), font,  fontScale, color, thickness, cv2.LINE_AA)
                isHaveErrFrame = True
            else:
                cv2.putText(frame, 'See two lines', (25, 75), font,  fontScale, color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Error frame, detect again', (25,100), font,  fontScale, color, thickness, cv2.LINE_AA)
            isHaveErrFrame = True
    return isHaveErrFrame
def identifyInvasion(left_line, right_line, frame, minDistance):
    _, width = frame.shape[:2]
    if left_line is not None and right_line is not None:
        x1, y1, x2, y2 = left_line[0]
        x3, y3, x4, y4 = right_line[0]
        m1 = b1 = m2 = b2 = None
        # Find linear equation of left line
        if x1 != x2:
            m1 = (y2 - y1) / (x2 - x1)
            b1 = y1 - m1 * x1
        else:
            x1, y1, x2, y2 = last_left_line[0]
            m1 = (y2 - y1) / (x2 - x1)
            b1 = y1 - m1 * x1
        # Find linear equation of right line 
        if x3 != x4:
            m2 = (y4 - y3) / (x4 - x3)
            b2 = y3 - m2 * x3
        else:
            x3, y3, x4, y4 = last_right_line[0]
            m2 = (y4 - y3) / (x4 - x3)
            b2 = y3 - m2 * x3
        # Find the intersection of the left and right lanes
        inter_x = (b2 - b1) / (m1 - m2)
        avg_x = ((width - b1)/m1 + (width - b2)/m2)/2
        distance = inter_x - avg_x # Variable uses to identify which lane the robot is invade on        
        org = (25,25) 
        if abs(distance) < minDistance:
            cv2.putText(frame, 'No Deviation', org, font,  fontScale, color, thickness, cv2.LINE_AA)
        else:
            if distance < 0:
                cv2.putText(frame, 'Left Deviation: %.2f' % (distance), org, font,  fontScale, color, thickness)
            elif distance > 0:
                cv2.putText(frame, 'Right Deviation: %.2f' % (distance), org, font, fontScale, color, thickness)


def backend(video_binary, video_original):
    global last_left_theta, last_left_line 
    global last_right_theta, last_right_line
    global frame_width, frame_height
    #Variables use to read video and write image, video
    in_img_count = 0
    out_img_count = 0
    is_first_5_frame = False
    output_model = cv2.VideoCapture(video_binary) # Binary output video of model 
    original = cv2.VideoCapture(video_original) # Original video
    frame_width = int(output_model.get(3))
    frame_height = int(output_model.get(4))
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    
    if not output_model.isOpened():
        print("Cannot open output video of model.")
        return
    if not original.isOpened():
        print("Cannot open original video.")
        return 

    # Array of frames, right lines and left lines
    frame_count = 0
    right_lines = []
    left_lines = []
    # error_frame = 0
    while True:
        ret_model, frame_model = output_model.read()
        ret_ori, frame_ori = original.read()
        if not ret_model or not ret_ori: 
            # Can't read frame from binary output video of model and original video
            break
        # If frame can read, push frame of original to frame_array
        # frame_array.append(frame_ori)
        frame_count = frame_count + 1
        image_name = f"image_{in_img_count}.jpg"
        image_path = os.path.join(save_in_dir, image_name)
        cv2.imwrite(image_path, frame_model)
        in_img_count += 1
        
        
        # Reduce noise
        gray_frame = cv2.cvtColor(frame_model, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
        kernel = np.ones((3, 3), np.uint8) 
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2) 
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # frame_model_arr.append(closing)
        lines = cv2.HoughLinesP(closing, 1, np.pi / 180, threshold=50, minLineLength=35, maxLineGap=5)
        
        points = [] 
        # Detect lane 
        if lines is not None:
            # Determine left and right line based on min(if left) /max (if right) theta angle
            # Then push it into corresponding array
            for line in lines:
                if line is not None:
                    x1, y1, x2, y2 = line[0]
                    points.append((x1, y1))
                    points.append((x2, y2))
            filtered_boxes = []
        for cnt in contours:
            filtered_points = []
            if cv2.contourArea(cnt) > 100:
                for point in points:
                    px, py = point
                    inside = cv2.pointPolygonTest(cnt, (int(px), int(py)), False)
                    if inside >= 0:
                        filtered_points.append(point)
                if len(filtered_points) > 5:
                    filtered_boxes.append(filtered_points)
        centroids = 6
        for fpt in filtered_boxes:
            if len(fpt) > 2:
                kmeans = KMeans(n_clusters=centroids, random_state=0)
                kmeans.fit(fpt)
                centers = kmeans.cluster_centers_
                # for center in centers:
                #     cv2.circle(image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
                tck, _ = splprep(centers.T, u=None, s=0.0, per=1)
                x_new, y_new = splev(np.linspace(0, 1, 100), tck)
                # f = interp1d(centers[:, 0], centers[:, 1], kind='linear')
                # # print(centers[:, 0].min(), centers[:, 0].max())
                # popt, _ = curve_fit(f, centers[:, 0], centers[:, 1])
                # x_new = np.linspace(centers[:, 0].min(), centers[:, 0].max(), 100)
                # y_new = f(x_new)
                cv2.polylines(frame_ori, [np.array([x_new, y_new], np.int32).T], False, (0, 255, 0), 4)                
        out.write(frame_ori)
        image_name = f"image_{out_img_count}.jpg"
        image_path = os.path.join(save_out_dir, image_name)
        cv2.imwrite(image_path, frame_ori)
        out_img_count += 1
        cv2.imshow('Processed Frame', frame_ori)
        
        if cv2.waitKey(1) == ord('q'):
            break   
    output_model.release()
    original.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    backend("simulation_detect.mp4", "simulation_original.mp4")

if __name__ == '__main__':
    main()