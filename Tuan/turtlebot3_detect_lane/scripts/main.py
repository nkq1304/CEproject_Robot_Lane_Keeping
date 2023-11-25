import rospy
import numpy as np
import cv2
import torch
from model import TwinLite as net

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

def Run(model, img):
    img = cv2.resize(img, (640, 360))
    img_rs = img.copy()
    img_rs[:,:,:]=0

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img = img.cuda().float() / 255.0
    img = img.cuda()
    with torch.no_grad():
        img_out = model(img)
    x0 = img_out[0]
    x1 = img_out[1]

    _, da_predict = torch.max(x0, 1)
    _, ll_predict = torch.max(x1, 1)

    DA = da_predict.byte().cpu().data.numpy()[0] * 255
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255
    # img_rs[DA>100]=[255,0,0]
    img_rs[LL > 100] = [255, 255, 255]

    return img_rs

def detectLane(line, type_of_line):
    # Function: Detect left and right lane
    thresh_theta = 45 #Theta in degree
    # Get theta of each line in "line"
    theta = []
    for index in range(frame_num) :
        if line[index] is not None:
            x1, y1, x2, y2 = line[index][0]
            theta1 = np.arctan2(y2 - y1, x2 - x1)
            theta.append(np.abs(theta1))
    # Find lines with angles theta approximately the same so that
    # the number of lines is maximum. Result is store in max_theta_group
    # Therefore, max_theta_group may be contain noisy lanes/ correct lanes
    sorted_theta = np.sort(theta)
    diff_theta = np.diff(sorted_theta)
    index = np.where(diff_theta > (thresh_theta * np.pi / 180))[0]
    split_theta_group = np.split(sorted_theta, index +1)
    max_theta_group = np.sort(max(split_theta_group, key = len))
    if(type_of_line == 'Left'):
        if(last_left_theta == None): 
            # If these are the first 5 frames, the default are that the frames with 
            # the largest theta are approximately the same as the lane and max_theta_group 
            # is contain correct lanes 
            for i in range(0, len(theta)):
                if theta[i] not in max_theta_group:
                    # This's mean this is noisy line 
                    if i == 0:
                        # Because this's first frame, so we need to find 
                        # the first correct lane to replace it
                        pos = None
                        for j in range (i+1 , len(theta)):
                            pos = np.where(max_theta_group == theta[j])[0]
                            if(pos.size > 0):
                                pos = pos[0]
                                break
                        line[i] = line[pos]
                    else:
                        line[i] = line[i-1]
                if i == (len(theta) - 1):
                    last_left_line = line[i] # Store the last line
        else:
            approximate_elements = [element for element in max_theta_group if abs(element - last_left_theta) < thresh_theta * np.pi / 180]
            if approximate_elements: 
                # max_theta_group contain correct lanes
                for i in range(0, len(theta)):
                    if theta[i] not in max_theta_group:
                        # line[i] ~ theta[i] is noisy lane 
                        # -> need to find correct lane to replace it
                        if i == 0:
                            line[i] = last_left_line
                        else:
                            line[i] = line[i-1]  
                    if i == (len(theta) - 1):
                        last_left_line = line[i] # Store the last line 
            else: 
                # max_theta_group contains noisy lanes
                for i in range(0, len(theta)):
                    if theta[i] in max_theta_group:
                        # line[i] ~ theta[i] is noisy lane 
                        # -> need to find correct lane to replace it
                        if i == 0:
                            line[i] = last_left_line
                        else:
                            line[i] = line[i-1]
                    if i == (len(theta) - 1):
                        last_left_line = line[i] # Store the last line
    elif (type_of_line == 'Right'):
        if(last_right_theta == None): 
            # If these are the first 5 frames, the default are that the frames with 
            # the largest theta are approximately the same as the lane and max_theta_group 
            # is contain correct lanes 
            for i in range(0, len(theta)):
                if theta[i] not in max_theta_group:
                    # This's mean this is noisy line 
                    if i == 0:
                        # Because this's first frame, so we need to find 
                        # the first correct lane to replace it
                        pos = None
                        for j in range (i+1 , len(theta)):
                            pos = np.where(max_theta_group == theta[j])[0]
                            if(pos.size > 0):
                                pos = pos[0]
                                break
                        line[i] = line[pos]
                    else:
                        line[i] = line[i-1]
                if i == (len(theta) - 1):
                    last_right_line = line[i] # Store the last line
        else:
            approximate_elements = [element for element in max_theta_group if abs(element - last_right_theta) < thresh_theta * np.pi / 180]
            if approximate_elements: 
                # max_theta_group contain correct lanes
                for i in range(0, len(theta)):
                    if theta[i] not in max_theta_group:
                        # line[i] ~ theta[i] is noisy lane 
                        # -> need to find correct lane to replace it
                        if i == 0:
                            line[i] = last_right_line
                        else:
                            line[i] = line[i-1]  
                    if i == (len(theta) - 1):
                        last_right_line = line[i] # Store the last line 
            else: 
                # max_theta_group contains noisy lanes
                for i in range(0, len(theta)):
                    if theta[i] in max_theta_group:
                        # line[i] ~ theta[i] is noisy lane 
                        # -> need to find correct lane to replace it
                        if i == 0:
                            line[i] = last_right_line
                        else:
                            line[i] = line[i-1]
                    if i == (len(theta) - 1):
                        last_right_line = line[i] # Store the last line
        
    return line

def drawInFrame(left_line, right_line, frame):
    height, _ = frame.shape[:2]

    try:
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
    except:
        return frame
    return frame

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
        distance = inter_x - width / 2 # Variable uses to identify which lane the robot is invade on
        if abs(distance) < minDistance:
            return 0
        else:
            return distance
        
def backend(binary_image):
    global frame_count
    global frame_array
    global right_lines
    global left_lines
    global last_left_theta
    global last_left_line
    global last_right_theta
    global last_right_line
    global frame_num
    global line_thickness

    frame_array.append(binary_image)
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

        right_lines.append(right_line)
        left_lines.append(left_line)

        right_lines = detectLane(right_lines, type_of_line = 'Right')
        left_lines = detectLane(left_lines, type_of_line = 'Left')

        for l_line, r_line, _frame in zip(left_lines, right_lines, frame_array):
            result_image = drawInFrame(l_line, r_line, _frame)
            result_distance = identifyInvasion(l_line, r_line, _frame, minDistance = 5)

        if len(right_lines) > frame_num:
            right_lines.pop(0)

        if len(left_lines) > frame_num:
            left_lines.pop(0)

        if len(frame_array) > frame_num:
            frame_array.pop(0)

        return result_image, result_distance

def image_callback(data):
    global last_angular_vel
    
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    cv_image = cv2.resize(cv_image, (1280, 720))
    binary_image = Run(model, cv_image)

    visualized_image, distance = backend(binary_image)

    twist = Twist()
    
    if distance is not None:
        last_angular_vel = -distance * 0.01

    twist.angular.z = last_angular_vel
    twist.linear.x = 0.1

    print("Angular velocity: ", twist.angular.z)

    pub.publish(twist)

    cv2.imshow("Visualized Image", visualized_image)
    # cv2.imshow("Image", cv_image)

    cv2.waitKey(1)

model = net.TwinLiteNet()
# TODO: If the model was trained with only one GPU, then comment the following line
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load("test_/model_0.pth"))
model.eval()

last_left_theta = None # Theta of last left line of previous 5 frames
last_left_line = None # Last left line of previous 5 frames
last_right_theta = None # Theta of last right line of previous 5 frames
last_right_line = None # Last right line of previous 5 frames
line_thickness = 4
# Array of frames, right lines and left lines  
frame_array = [] # This frame will use to write to output video
right_lines = []
left_lines = []

frame_num = 5 # Number of frames to determine the lane

last_angular_vel = 0

bridge = CvBridge()

rospy.init_node("get_image", anonymous=True)
rospy.Subscriber("/camera/image", Image, image_callback)
pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
cv2.destroyAllWindows()


