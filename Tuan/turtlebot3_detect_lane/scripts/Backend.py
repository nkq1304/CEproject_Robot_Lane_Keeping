import cv2
import numpy as np

class Backend():
    def __init__(self, frame_width = 1280, frame_height = 720):
        self.last_left_theta = None # Theta of last left line of previous 5 frames
        self.last_left_line = None # Last left line of previous 5 frames
        self.last_right_theta = None # Theta of last right line of previous 5 frames
        self.last_right_line = None # Last right line of previous 5 frames
        self.frame_num = 5 # Max frame's array number
        self.line_thickness = 4 # Thickness of 
        self.thresh_theta = 5 # Threshold theta in degree
        self.scale_thresh = 2
        #Variables use to read video and write image, video
        self.is_first_5_frame = False

        # Array of frames, right lines and left lines  
        self.frame_array = [] # This frame will use to write to output video
        self.frame_count = 0
        self.right_lines = []
        self.left_lines = []
        self.error_frame = 0

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.invasion = 0
    
    def findInterWithXAxis(self, line):
        if line is not None:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                return x1, self.frame_height
            # Find linear equation of left line
            m1 = (y2 - y1) / (x2 - x1)
            b1 = y1 - m1 * x1
            return (self.frame_height - b1) / m1, self.frame_height

    def drawInFrame(self, left_line, right_line, frame):
        if left_line is not None and right_line is not None:
            x1, y1, x2, y2 = left_line[0]
            x3, y3, x4, y4 = right_line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), self.line_thickness)
            cv2.line(frame, (x3, y3), (x4, y4), (255, 0, 0), self.line_thickness)

    def calTheta(self, line):
        if line is not None:
            x1, y1, x2, y2 = line[0]
            return np.arctan2(y2 - y1, x2 - x1)

    def replaceLane(self, line, theta, approx_theta_arr, last_line, last_theta, ef_num):
        if (last_theta == None): 
            if len(line) == 0:
                return line, last_line, last_theta
            for i in range(0, len(line)):
                if line[i] is not None:
                    theta_temp = self.calTheta(line[i])
                    if theta_temp not in approx_theta_arr:
                        if i == 0:
                            pos = 0
                            for j in range (1 , len(line)):
                                if line[j] is not None and self.calTheta(line[j]) in approx_theta_arr:
                                    pos = approx_theta_arr.index(self.calTheta(line[j]))
                            line[i] = line[pos]
                        else:
                            line[i] = line[i-1]
                else:   
                    if i == 0:
                        pos = 0
                        for j in range (1 , len(line)):
                            if line[j] is not None and self.calTheta(line[j]) in approx_theta_arr:
                                pos = approx_theta_arr.index(self.calTheta(line[j]))
                        line[i] = line[pos]
                    else:
                        line[i] = line[i - 1]
            last_line = line[len(line) - 1]
            last_theta = self.calTheta(line[len(line) - 1])
        else:
            if len(line) == 0:
                return line, last_line, last_theta
            if (ef_num > int(self.frame_num / 2)) or (ef_num <= int(self.frame_num / 2) and ef_num > 0):
                # If number of error frame larger than half of number of max frame:
                # Accept these error frame are correct lane
                square_angle = 90 * np.pi / 180
                for i in range (self.frame_num - 1, len(line)):
                    if line[i] is not None:
                        # print("Replace error line %s", i)
                        x, _ = self.findInterWithXAxis(line[i])
                        if x <= self.frame_width and x >= 0:
                            # print('In, last line', last_line[0])
                            theta_temp = np.abs(self.calTheta(line[i]))
                            if np.bitwise_xor(np.abs(theta_temp - last_theta) > (self.thresh_theta * np.pi / (self.scale_thresh *180)),
                                            np.abs(square_angle - theta_temp) > (self.thresh_theta * np.pi / (self.scale_thresh *180))):
                                line[i] = last_line
                        else:
                            # print('Out, last line: ', last_line[0])
                            line[i] = last_line
                    else: 
                        # print('Replace not line %s', i)
                        line[i] = last_line
                        
            last_line = line[len(line) - 1]
            last_theta = self.calTheta(line[len(line) - 1])
        return line, last_line, last_theta

    def detectErrorLine(self, line):  
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
        index = np.where(diff_theta > (self.thresh_theta * np.pi / 180))[0]
        split_theta = np.split(sorted_theta, index +1)
        max_theta_arr = np.sort(max(split_theta, key = len))
        # 4. Filter the retreived list again, as this list may still contain 
        # pairs deviating from each other larger than the threshold
        approx_theta_arr = []
        for i in range(len(max_theta_arr)):
            if np.abs(max_theta_arr[i] - max_theta_arr[0]) <= self.thresh_theta * np.pi / 180:
                approx_theta_arr.append(max_theta_arr[i])
            else:
                break
        # 5. If all theta angle are approximately with each other, return true. Otherwise, return false
        if(len(approx_theta_arr) == len(line)):
            return False, approx_theta_arr, theta
        else:
            return True, approx_theta_arr, theta

    def detectLane(self, line, type_of_line, error_frame):
        flag, approx_theta_arr, theta = self.detectErrorLine(line)
        # print(approx_theta_arr)
        if(type_of_line == 'Left'):
            # if flag == True:
            #     print('Replace left')
            #     for i in range(len(line)):
            #         if line[i] is not None:
            #             print("Line before replace", i, line[i])
            line, self.last_left_line, self.last_left_theta = self.replaceLane(line, theta, approx_theta_arr, self.last_left_line, self.last_left_theta, error_frame)
            # if flag == True:
            #     for i in range(len(line)):
            #         if line[i] is not None:
            #             print('Line after replace', i, line[i])
        elif (type_of_line == 'Right'):
            # if flag == True:
            #     print('Replace right')
            #     for i in range(len(line)):
            #         if line[i] is not None:
            #             print("Line before replace", i, line[i])
            line, self.last_right_line, self.last_right_theta = self.replaceLane(line, theta, approx_theta_arr, self.last_right_line, self.last_right_theta, error_frame)
            # if flag == True:
            #     for i in range(len(line)):
            #         if line[i] is not None:
            #             print("Line after replace", i, line[i])
        return line

    def identifyInvasion(self, left_line, right_line, frame, minDistance):
        if left_line is not None and right_line is not None:
            x1, y1, x2, y2 = left_line[0]
            x3, y3, x4, y4 = right_line[0]

            m1 = (y2 - y1) / (x2 - x1)
            b1 = y1 - m1 * x1
            m2 = (y4 - y3) / (x4 - x3)
            b2 = y3 - m2 * x3
            inter_x = (b2 - b1) / (m1 - m2)
            inter_y = m1 * inter_x + b1

            distance = inter_x - self.frame_width / 2 # Variable uses to identify which lane the robot is invade on
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
                return distance
            return 0
    
    def detectErrorLine(self, line):
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
        index = np.where(diff_theta > (self.thresh_theta * np.pi / 180))[0]
        split_theta = np.split(sorted_theta, index +1)
        max_theta_arr = np.sort(max(split_theta, key = len))
        # 4. Filter the retreived list again, as this list may still contain 
        # pairs deviating from each other larger than the threshold
        approx_theta_arr = []
        for i in range(len(max_theta_arr)):
            if np.abs(max_theta_arr[i] - max_theta_arr[0]) <= self.thresh_theta * np.pi / 180:
                approx_theta_arr.append(max_theta_arr[i])
            else:
                break
        # 5. If all theta angle are approximately with each other, return true. Otherwise, return false
        if(len(approx_theta_arr) == len(line)):
            return False, approx_theta_arr, theta
        else:
            return True, approx_theta_arr, theta

    def backend(self, binary_image):
        self.frame_array.append(binary_image)
        self.frame_count = self.frame_count + 1
        # Reduce noise
        gray_frame = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
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
            self.right_lines.append(right_line)
            self.left_lines.append(left_line)
        if self.is_first_5_frame == True:
            # If number of error frame larger than half the number of max frame
            # or the current frame does not have any errors in lines, 
            # but the error frame count is greater than 0 frame -> Handle it
            is_r_lines_have_ef, _, _ = self.detectErrorLine(self.right_lines)
            is_l_lines_have_ef, _, _ = self.detectErrorLine(self.left_lines)
            if (is_l_lines_have_ef or is_r_lines_have_ef):
                self.error_frame = self.error_frame + 1
                if(self.error_frame > int(self.frame_num / 2)):
                    self.frame_count = self.frame_count + self.error_frame
                else:
                    self.frame_count = self.frame_count - 1
            elif self.error_frame <= int(self.frame_num / 2) and self.error_frame > 0:
                self.frame_count = self.frame_count + self.error_frame
        
        # Start processing when there are enough frames:
        if self.frame_count == (self.frame_num + self.error_frame):
            # 1. Identify left and right lanes again: Because it is based on the max theta angle, 
            # error (noisy) lines may exist, so we need to remove/replace them with correct lines.
            if self.error_frame > 0:
                end = self.frame_count - 1
            else:
                end = self.frame_count
            
            self.right_lines = self.detectLane(self.right_lines, type_of_line = 'Right', error_frame = self.error_frame)
            self.left_lines = self.detectLane(self.left_lines, type_of_line = 'Left', error_frame = self.error_frame)

            for l_line, r_line, _frame, i in zip(self.left_lines, self.right_lines, self.frame_array, range(0, end)):
                # 2. Write this frame into output video
                # If this frame belong to first 5 frames, write it to output video
                # Else, only draw on frames that have not been processed yet.
                if (self.is_first_5_frame == False) or (i >= (self.frame_num - 1)):
                    # 1. Draw detected lines on each original frame in frame_array
                    self.drawInFrame(l_line, r_line, _frame)
                    self.invasion = self.identifyInvasion(l_line, r_line, _frame, minDistance = 5)
                    cv2.imshow('frame', _frame)

                if (self.is_first_5_frame == False) and (i == (self.frame_num - 1)):    
                    self.is_first_5_frame = True

            # 3. Remove some element in frame array, right lines array, left line array
            if self.error_frame > 0:
                # print("Frames: %s, %s", in_img_count - error_frame, in_img_count - 1)
                self.right_lines = self.right_lines[self.error_frame:]
                self.left_lines = self.left_lines[self.error_frame:]
                self.frame_array = self.frame_array[self.error_frame:]
                self.frame_count = self.frame_count - (self.error_frame + 1)
                self.error_frame = 0  
            else:
                self.right_lines = self.right_lines[1:]
                self.left_lines = self.left_lines[1:]
                self.frame_array = self.frame_array[1:]
                self.frame_count = self.frame_count - 1

        