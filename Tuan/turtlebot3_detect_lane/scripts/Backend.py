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
        self.frame_model_array = []
        self.frame_count = 0
        self.right_lines = []
        self.left_lines = []
        self.error_frame = 0

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.fontScale = 0.65
        self.color = (255, 0, 0) 
        self.thickness = 1

        self.invasion = 0
    
    def findInterWithXAxis(self, line):
        if line is not None:
            x1, y1, x2, y2 = line[0]
            # Find linear equation of left line
            if x1 == x2:
                return True, 0, 0
            else:
                m1 = (y2 - y1) / (x2 - x1)
                b1 = y1 - m1 * x1
                return False, ((self.frame_height - b1) / m1), self.frame_height

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
        kind_of_error = []
        if (last_theta == None): 
            if len(line) == 0:
                return line, last_line, last_theta, kind_of_error
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
                square_angle = 90 * np.pi / 180
                for i in range (self.frame_num - 1, len(line)):
                    if line[i] is None:
                        line[i] = last_line
                        kind_of_error.append(3)
                        continue

                    divide_by_zero =  False
                    divide_by_zero, x, _ = self.findInterWithXAxis(line[i])

                    if np.isnan(x) or divide_by_zero:
                        line[i] = last_line
                        kind_of_error.append(3)
                        continue

                    x =  int(x)

                    if x >= self.frame_width or x <= 0:
                        line[i] = last_line
                        kind_of_error.append(1)
                        continue

                    theta_temp = np.abs(self.calTheta(line[i]))

                    if np.bitwise_xor(np.abs(theta_temp - last_theta) > (self.thresh_theta * np.pi / (self.scale_thresh *180)),
                                    np.abs(square_angle - theta_temp) > (self.thresh_theta * np.pi / (self.scale_thresh *180))):
                        line[i] = last_line
                        if(np.abs(theta_temp - last_theta) > (self.thresh_theta * np.pi / (self.scale_thresh *180))):
                            kind_of_error.append(1)
                        else:
                            kind_of_error.append(2)

            last_line = line[len(line) - 1]
            last_theta = self.calTheta(line[len(line) - 1])

        return line, last_line, last_theta, kind_of_error

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
        sorted_theta = np.sort(theta)

        diff_theta = np.diff(sorted_theta)
        index = np.where(diff_theta > (self.thresh_theta * np.pi / 180))[0]
        split_theta = np.split(sorted_theta, index +1)
        max_theta_arr = np.sort(max(split_theta, key = len))

        approx_theta_arr = []
        for i in range(len(max_theta_arr)):
            if np.abs(max_theta_arr[i] - max_theta_arr[0]) <= self.thresh_theta * np.pi / 180:
                approx_theta_arr.append(max_theta_arr[i])
            else:
                break
        
        if (len(approx_theta_arr) == len(line)):
            return False, approx_theta_arr, theta
        else:
            return True, approx_theta_arr, theta

    def detectLane(self, line, type_of_line, error_frame):
        flag, approx_theta_arr, theta = self.detectErrorLine(line)
        kind_of_error = []

        if(type_of_line == 'Left'):
            line, self.last_left_line, self.last_left_theta, kind_of_error = self.replaceLane(line, theta, approx_theta_arr, self.last_left_line, self.last_left_theta, error_frame)
        elif (type_of_line == 'Right'):
            line, self.last_right_line, self.last_right_theta, kind_of_error = self.replaceLane(line, theta, approx_theta_arr, self.last_right_line, self.last_right_theta, error_frame)
        return line, kind_of_error
        
    def drawInFrame(self, left_line, right_line, frame_ori, frame_model):
        height, _ = frame_ori.shape[:2]
        if left_line is None or right_line is None:
            return
        
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
    
    def drawErrorInFrame(self, frame, left, right):
        org = (25, 50)
        if left == 0 and right != 0:
            if right == 1:
                self.drawTextInFrame(frame, 'Error right line: Noisy line!')
            elif right == 2:
                self.drawTextInFrame(frame, 'Danger: Robot on right line!')
            elif right == 3:
                self.drawTextInFrame(frame, 'Error right line: Line not found!')
        elif left != 0 and right == 0:
            if left == 1:
                self.drawTextInFrame(frame, 'Error left line: Noisy line!')
            elif left == 2:
                self.drawTextInFrame(frame, 'Danger: Robot on left line!')
            elif left == 3:
                self.drawTextInFrame(frame, 'Error left line: Line not found!')
        elif left != 0 and right != 0:
            if left == 2:
                self.drawTextInFrame(frame, 'Danger: Robot on left line!')
            elif right == 2:
                self.drawTextInFrame(frame, 'Danger: Robot on right line!')
            else:
                org2 = (25,75)
                if left == 1:
                    self.drawTextInFrame(frame, 'Error left line: Noisy line!')
                    if right == 1:
                        self.drawTextInFrame(frame, 'Error right line: Noisy line!', org2)
                    elif right == 3:
                        self.drawTextInFrame(frame, 'Error right line: Line not found!', org2)
                elif left == 3:
                    self.drawTextInFrame(frame, 'Error left line: Line not found!')
                    if right == 1:
                        self.drawTextInFrame(frame, 'Error right line: Noisy line!', org2)
                    elif right == 3:
                        self.drawTextInFrame(frame, 'Error right line: Line not found!', org2)

    def drawTextInFrame(self, frame, text, org = (25, 50)):
        cv2.putText(frame, text, org, self.font,  self.fontScale, self.color, self.thickness, cv2.LINE_AA)

    def detectDrawedErrLine(self, frame, right_line, left_line, minDistance):
        org = (25, 100)
        isHaveErrFrame = False
        if left_line is None or right_line is  None:
            return isHaveErrFrame
        
        x1, y1, x2, y2 = left_line[0]
        x3, y3, x4, y4 = right_line[0]
        m1 = b1 = m2 = b2 = None
        # Find linear equation of left line
        if x1 != x2:
            m1 = (y2 - y1) / (x2 - x1)
            b1 = y1 - m1 * x1
        else:
            x1, y1, x2, y2 = self.last_left_line[0]
            m1 = (y2 - y1) / (x2 - x1)
            b1 = y1 - m1 * x1
        # Find linear equation of right line 
        if x3 != x4:
            m2 = (y4 - y3) / (x4 - x3)
            b2 = y3 - m2 * x3
        else:
            x3, y3, x4, y4 = self.last_right_line[0]
            m2 = (y4 - y3) / (x4 - x3)
            b2 = y3 - m2 * x3
        # Find the intersection of the left and right lanes
        inter_x = (b2 - b1) / (m1 - m2)
        inter_y = m1 * inter_x + b1
        if inter_y < y2 and inter_y < y3 and inter_y <= self.frame_height:
            distance1 = np.sqrt((x4 - x1) *(x4 - x1) + (y4 - y1)*(y4 - y1))
            distance2 = np.sqrt((x3 - x2) *(x3 - x2) + (y3 - y2)*(y3 - y2))
            if (distance1 <= minDistance) and (distance2 <= minDistance):
                cv2.putText(frame, 'Error frame, detect again!', (25,100), self.font,  self.fontScale, self.color, self.thickness, cv2.LINE_AA)
                isHaveErrFrame = True
            else:
                cv2.putText(frame, 'See two lines', (25, 75), self.font,  self.fontScale, self.color, self.thickness, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Error frame, detect again', (25, 100), self.font,  self.fontScale, self.color, self.thickness, cv2.LINE_AA)
            isHaveErrFrame = True

        return isHaveErrFrame
    
    def identifyInvasion(self, left_line, right_line, frame, minDistance):
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
                x1, y1, x2, y2 = self.last_left_line[0]
                m1 = (y2 - y1) / (x2 - x1)
                b1 = y1 - m1 * x1
            # Find linear equation of right line 
            if x3 != x4:
                m2 = (y4 - y3) / (x4 - x3)
                b2 = y3 - m2 * x3
            else:
                x3, y3, x4, y4 = self.last_right_line[0]
                m2 = (y4 - y3) / (x4 - x3)
                b2 = y3 - m2 * x3
            # Find the intersection of the left and right lanes
            inter_x = (b2 - b1) / (m1 - m2)
            avg_x = ((width - b1) / m1 + (width - b2) / m2) / 2

            distance = inter_x - avg_x # Variable uses to identify which lane the robot is invade on        
            org = (25, 25) 
            if abs(distance) < minDistance:
                self.drawTextInFrame(frame, 'No Deviation', org)
                return 0
            else:
                if distance < 0:
                    self.drawTextInFrame(frame, 'Left Deviation: %.2f' % (distance), org)
                elif distance > 0:
                    self.drawTextInFrame(frame, 'Right Deviation: %.2f' % (distance), org)
            return distance

    def backend(self, binary_image):
        self.frame_array.append(binary_image)
        self.frame_count = self.frame_count + 1

        gray_frame = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
        kernel = np.ones((3, 3), np.uint8) 
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)
        lines = cv2.HoughLinesP(closing, 1, np.pi / 180, threshold=75, minLineLength=35, maxLineGap=5)

        self.frame_model_array.append(closing)

        if lines is not None:
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
        
        if self.frame_count == (self.frame_num + self.error_frame):
            self.error_count = 0

            self.right_lines, right_error = self.detectLane(self.right_lines, type_of_line = 'Right', error_frame = self.error_frame)
            self.left_lines, left_error = self.detectLane(self.left_lines, type_of_line = 'Left', error_frame = self.error_frame)

            if self.error_frame > 0:
                end = self.frame_count - 1
            else:
                end = self.frame_count

            for l_line, r_line, f_ori, f_mdl, i in zip(self.left_lines, self.right_lines, self.frame_array, self.frame_model_array, range(0, end)):
                if (self.is_first_5_frame == False) or (i >= (self.frame_num - 1)):
                    # self.drawErrorInFrame(f_ori, left_error[self.error_count], right_error[self.error_count])
                    isHaveErrFrame = self.detectDrawedErrLine(f_ori, r_line, l_line, minDistance = 10)

                    if isHaveErrFrame == False:
                        self.drawInFrame(l_line, r_line, f_ori, f_mdl)
                        self.invasion = self.identifyInvasion(l_line, r_line, f_ori, minDistance = 10)
                    self.error_count = self.error_count + 1
                else:
                    self.drawInFrame(l_line, r_line, f_ori, f_mdl)
                    self.invasion = self.identifyInvasion(l_line, r_line, f_ori, minDistance = 10)
                    self.drawTextInFrame(f_ori, 'See two lines')

                cv2.imshow('frame', f_ori)

                if (self.is_first_5_frame == False) and (i == (self.frame_num - 1)):    
                    self.is_first_5_frame = True

            if self.error_frame > 0:
                self.error_count = 0
                self.right_lines = self.right_lines[self.error_frame:]
                self.left_lines = self.left_lines[self.error_frame:]
                self.frame_array = self.frame_array[self.error_frame:]
                self.frame_model_array = self.frame_model_array[self.error_frame:]
                self.frame_count = self.frame_count - (self.error_frame + 1)
                self.error_frame = 0
            else:
                self.right_lines = self.right_lines[1:]
                self.left_lines = self.left_lines[1:]
                self.frame_array = self.frame_array[1:]
                self.frame_model_array = self.frame_model_array[1:]
                self.frame_count = self.frame_count - 1

        