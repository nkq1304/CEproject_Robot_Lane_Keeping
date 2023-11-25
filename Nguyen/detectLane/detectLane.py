import cv2
import numpy as np
import os

# Setup directory to save frame
save_in_dir = "D:/HK231/DAMHKTMT/task4/input_frame" 
if not os.path.exists(save_in_dir):
    os.makedirs(save_in_dir)
save_out_dir = "D:/HK231/DAMHKTMT/task4/output_frame" 
if not os.path.exists(save_out_dir):
    os.makedirs(save_out_dir)

# Set global variable
last_left_theta = None # Theta of last left line of previous 5 frames
last_left_line = None # Last left line of previous 5 frames
last_right_theta = None # Theta of last right line of previous 5 frames
last_right_line = None # Last right line of previous 5 frames
frame_num = 5 # Max frame's array number
line_thickness = 4
thresh_theta = 10 # Threshold theta in degree

def replaceLane(line, theta, approx_theta_arr, last_line, last_theta, error_frame):
    if(last_theta == None): 
        # If these are the first 5 frames, the default are that the frames with 
        # the largest theta are approximately the same as the lane and approx_theta_arr 
        # is contain correct lanes 
        for i in range(0, len(theta)):
            if theta[i] not in approx_theta_arr:
                # This's mean this is noisy line 
                if i == 0:
                    # Because this's first frame, so we need to find 
                    # the first correct lane to replace it
                    pos = None
                    for j in range (i+1 , len(theta)):
                        pos = np.where(approx_theta_arr == theta[j])[0]
                        if(pos.size > 0):
                            pos = pos[0]
                            break
                    line[i] = line[pos]
                    theta[i] = theta[pos]
                else:
                    line[i] = line[i-1]
                    theta[i] = theta[i-1]
            if i == (len(theta) - 1): # Store the last line information
                if(np.argmax(theta) is not None):
                    last_line = line[np.argmax(theta)] 
                    last_theta = np.max(theta)
    else:
        if error_frame > int(frame_num / 2):
            # If number of error frame larger than half of number of max frame:
            # Accept these error frame are correct lane
            for i in range (len(line) - 1, frame_num - 1, -1):
                if line[i] is not None:                        
                    x1, y1, x2, y2 = line[i][0]
                    last_theta = np.arctan2(y2 - y1, x2 - x1)
                    last_line = line[i]
                    break
        elif error_frame <= int(frame_num / 2) and error_frame > 0: 
            # If number of error frame smaller than or equal to half of number of max frame
            # Replace error lines in these error frames
            for i in range(0, len(theta)):
                if theta[i] not in approx_theta_arr:
                    if i == 0:
                        line[i] = last_line
                        theta[i] = last_theta
                    else:
                        line[i] = line[i-1] 
                        theta[i] = theta[i-1] 
                if i == (len(theta) - 1):
                    if(np.argmax(theta) is not None):
                        last_line = line[np.argmax(theta)] # Store the last line 
                        last_theta = np.max(theta)
        # The last case: number error frame equal to 0 -> all lines in all frames are correct.
        # Not need to do any thing
    return line, last_line, last_theta

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
    aprox_theta_arr = []
    for i in range(len(max_theta_arr)):
        if max_theta_arr[i] - max_theta_arr[0] <= thresh_theta * np.pi / 180:
            aprox_theta_arr.append(max_theta_arr[i])
        else:
            break
    # 5. If all theta angle are approximately with each other, return true. Otherwise, return false
    if(len(aprox_theta_arr) == len(theta)):
        return False, max_theta_arr, theta
    else:
        return True, max_theta_arr, theta
    

def detectLane(line, type_of_line, error_frame):
    # Function: Detect left and right lane
    global last_left_theta, last_left_line 
    global last_right_theta, last_right_line
    # Get theta of each line in "line"
    _, approx_theta_arr, theta = detectErrorLine(line)
    if(type_of_line == 'Left'):
        line, last_left_line, last_left_theta = replaceLane(line, theta, approx_theta_arr, last_left_line, last_left_theta, error_frame)
        # if(last_left_theta == None): 
        #     # If these are the first 5 frames, the default are that the frames with 
        #     # the largest theta are approximately the same as the lane and approx_theta_arr 
        #     # is contain correct lanes 
        #     for i in range(0, len(theta)):
        #         if theta[i] not in approx_theta_arr:
        #             # This's mean this is noisy line 
        #             if i == 0:
        #                 # Because this's first frame, so we need to find 
        #                 # the first correct lane to replace it
        #                 pos = None
        #                 for j in range (i+1 , len(theta)):
        #                     pos = np.where(approx_theta_arr == theta[j])[0]
        #                     if(pos.size > 0):
        #                         pos = pos[0]
        #                         break
        #                 line[i] = line[pos]
        #                 theta[i] = theta[pos]
        #             else:
        #                 line[i] = line[i-1]
        #                 theta[i] = theta[i-1]
        #         if i == (len(theta) - 1):
        #             if(np.argmax(theta) is not None):
        #                 last_left_line = line[np.argmax(theta)] # Store the last line
        #                 last_left_theta = np.max(theta)
        # else:
        #     if error_frame > int(frame_num / 2):
        #         # If number of error frame larger than half of number of max frame:
        #         # Accept these error frame are correct lane
        #         for i in range (len(line) - 1, frame_num - 1, -1):
        #             if line[i] is not None:                        
        #                 x1, y1, x2, y2 = line[i][0]
        #                 last_left_theta = np.arctan2(y2 - y1, x2 - x1)
        #                 last_left_line = line[i]
        #                 break
        #     elif error_frame <= int(frame_num / 2) and error_frame > 0: 
        #         # If number of error frame smaller than or equal to half of number of max frame
        #         # Replace error lines in these error frames
        #         line , last_left_line, last_left_theta = replaceLane(line, theta, approx_theta_arr, 
        #                                                                 last_left_line, last_left_theta)
        #     # The last case: number error frame equal to 0 -> all lines in all frames are correct.
        #     # Not need to do any thing
    elif (type_of_line == 'Right'):
        line, last_right_line, last_right_theta = replaceLane(line, theta, approx_theta_arr, last_right_line, last_right_theta, error_frame)
        # if(last_right_theta == None): 
        #     # If these are the first 5 frames, the default are that the frames with 
        #     # the largest theta are approximately the same as the lane and max_theta_array 
        #     # is contain correct lanes 
        #     for i in range(0, len(theta)):
        #         if theta[i] not in approx_theta_arr:
        #             # This's mean this is noisy line 
        #             if i == 0:
        #                 # Because this's first frame, so we need to find 
        #                 # the first correct lane to replace it
        #                 pos = None
        #                 for j in range (i+1 , len(theta)):
        #                     pos = np.where(approx_theta_arr == theta[j])[0]
        #                     if(pos.size > 0):
        #                         pos = pos[0]
        #                         break
        #                 line[i] = line[pos]
        #                 theta[i] = theta[pos]
        #             else:
        #                 line[i] = line[i-1]
        #                 theta[i] = theta[i-1]
        #         if i == (len(theta) - 1):
        #             if(np.argmax(theta) is not None):
        #                 last_right_line = line[np.argmax(theta)] # Store the last line
        #                 last_right_theta = np.max(theta)
        # else:
        #     if error_frame > int(frame_num / 2):
        #         # If number of error frame larger than half of number of max frame:
        #         # Accept these error frame are correct lane
        #         for i in range (len(line) - 1, frame_num - 1, -1):
        #             if line[i] is not None:                        
        #                 x1, y1, x2, y2 = line[i][0]
        #                 last_right_theta = np.arctan2(y2 - y1, x2 - x1)
        #                 last_right_line = line[i]
        #                 break
        #     elif error_frame <= int(frame_num / 2) and error_frame > 0: 
        #         # If number of error frame smaller than or equal to half of number of max frame
        #         # Replace error lines in these error frames
        #         line , last_right_line, last_right_theta = replaceLane(line, theta, approx_theta_arr, 
        #                                                                 last_right_line, last_right_theta)
        #     # The last case: number error frame equal to 0 -> all lines in all frames are correct.
        #     # Not need to do any thing
        
    return line

def drawInFrame(left_line, right_line, frame):
    height, _ = frame.shape[:2]
    if left_line is not None and right_line is not None:
        x1, y1, x2, y2 = left_line[0]
        x3, y3, x4, y4 = right_line[0]
        # Find linear equation of left line
        m1 = (y2 - y1) / (x2 - x1)
        b1 = y1 - m1 * x1
        # Find linear equation of right line 
        m2 = (y4 - y3) / (x4 - x3)
        b2 = y3 - m2 * x3
        # Draw line
        line_left_color = (0, 0, 255) # RED
        cv2.line(frame, (int((height - b1) / m1), height), (x2, y2), line_left_color, line_thickness) 
        line_right_color = (0, 255, 0) # GREEN
        cv2.line(frame, (x3, y3), (int((height - b2) / m2), height), line_right_color, line_thickness)

def identifyInvasion(left_line, right_line, frame, minDistance):
    _, width = frame.shape[:2]
    if left_line is not None and right_line is not None:
        x1, y1, x2, y2 = left_line[0]
        x3, y3, x4, y4 = right_line[0]
        # Find linear equation of left line
        m1 = (y2 - y1) / (x2 - x1)
        b1 = y1 - m1 * x1
        # Find linear equation of right line 
        m2 = (y4 - y3) / (x4 - x3)
        b2 = y3 - m2 * x3
        # Find the intersection of the left and right lanes
        inter_x = (b2 - b1) / (m1 - m2)
        distance = inter_x - width/2 # Variable uses to identify which lane the robot is invade on
        font = cv2.FONT_HERSHEY_COMPLEX
        org = (50, 50) 
        fontScale = 0.65
        color = (255, 0, 0) 
        thickness = 1
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
    frame_array = [] # This frame will use to write to output video
    frame_count = 0
    right_lines = []
    left_lines = []
    error_frame = 0

    while True:
        ret_model, frame_model = output_model.read()
        ret_ori, frame_ori = original.read()
        if not ret_model or not ret_ori: 
            # Can't read frame from binary output video of model and original video
            break
        # If frame can read, push frame of original to frame_array
        frame_array.append(frame_ori)
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
        lines = cv2.HoughLinesP(closing, 1, np.pi / 180, threshold=75, minLineLength=35, maxLineGap=5)

        # Detect lane 
        if lines is not None:
            # Determine left and right line based on min(if left) /max (if right) theta angle
            # Then push it into corresponding array
            left_line = None
            right_line = None
            max_left_theta = 0
            max_right_theta = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                theta = np.arctan2(y2 - y1, x2 - x1)
                if theta < 0:
                    if abs(theta) > max_left_theta:
                        max_left_theta = abs(theta)
                        left_line = line
                elif theta > 0:
                    if theta > max_right_theta:
                        max_right_theta = theta
                        right_line = line
            right_lines.append(right_line)
            left_lines.append(left_line)
        if is_first_5_frame != False :
            # If number of error frame larger than half the number of max frame
            # or the current frame does not have any errors in lines, 
            # but the error frame count is greater than 0 frame -> Handle it
            if (detectErrorLine(right_lines) or detectErrorLine(left_lines)):
                error_frame = error_frame + 1
                if(error_frame > int(frame_num / 2)):
                    frame_count = frame_count + error_frame
                else:
                    frame_count = frame_count - 1
            elif error_frame <= int(frame_num / 2) and error_frame > 0:
                frame_count = frame_count + error_frame
        
        # Start processing when there are enough frames:
        if frame_count == (frame_num + error_frame):
            # 1. Identify left and right lanes again: Because it is based on the max theta angle, 
            # error (noisy) lines may exist, so we need to remove/replace them with correct lines.
            right_lines = detectLane(right_lines, type_of_line = 'Right', error_frame = error_frame)
            left_lines = detectLane(left_lines, type_of_line = 'Left', error_frame = error_frame)
            if error_frame > 0:
                end = frame_count - 1
            else:
                end = frame_count

            for l_line, r_line, _frame, i in zip(left_lines, right_lines, frame_array, range(0, end)):
                # 2. Write this frame into output video
                # If this frame belong to first 5 frames, write it to output video
                # Else, only draw on frames that have not been processed yet.

                if (is_first_5_frame == False) or (i >= (frame_num - 1)):
                    # 1. Draw detected lines on each original frame in frame_array
                    drawInFrame(l_line, r_line, _frame)
                    # 2. Determine which lane the turtlebot is encroaching on
                    identifyInvasion(l_line, r_line, _frame, minDistance=5)
                    out.write(_frame)
                    image_name = f"image_{out_img_count}.jpg"
                    image_path = os.path.join(save_out_dir, image_name)
                    cv2.imwrite(image_path, _frame)
                    out_img_count += 1
                    cv2.imshow('Processed Frame', _frame)

                if (is_first_5_frame == False) and (i == (frame_num - 1)):    
                    is_first_5_frame = True

            # 3. Remove some element in frame array, right lines array, left line array
            if error_frame > 0:
                right_lines = right_lines[error_frame:]
                left_lines = left_lines[error_frame:]
                frame_array = frame_array[error_frame:]
                frame_count = frame_count - (error_frame + 1)
                error_frame = 0  
            else:
                right_lines = right_lines[1:]
                left_lines = left_lines[1:]
                frame_array = frame_array[1:]
                frame_count = frame_count - 1
                
        if cv2.waitKey(1) == ord('q'):
            break   
    output_model.release()
    original.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    backend("japan_driving.mp4", "japan_driving_original.mp4")

if __name__ == '__main__':
    main()
