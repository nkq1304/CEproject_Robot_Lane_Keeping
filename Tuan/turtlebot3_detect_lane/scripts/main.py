import rospy
import numpy as np
import cv2
import torch
from model import TwinLite as net
import math

from Backend import Backend
from ControlLane import ControlLane

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

def process_image(model, img):
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

def image_callback(data):
    global last_angular_vel
    
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    cv_image = cv2.resize(cv_image, (1280, 720))

    binary_image = process_image(model, cv_image)

    backend_node.backend(binary_image)
    control_lane_node.cbFollowLane(backend_node.invasion)

    cv2.waitKey(1)

model = net.TwinLiteNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load("test_/model_0.pth"))
model.eval()

backend_node = Backend(640, 360)

bridge = CvBridge()

rospy.init_node("get_image", anonymous=True)
rospy.Subscriber("/camera/image", Image, image_callback)
control_lane_node = ControlLane()

try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
cv2.destroyAllWindows()


