#!/usr/bin/env python3
import rospy
import numpy as np
import sys

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import torch
import numpy as np
from tqdm.autonotebook import tqdm
import os
import torch
from model import TwinLite as net
import cv2


def Run(model, img):
    img = cv2.resize(img, (640, 360))
    img_rs = img.copy()

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
    img_rs[LL > 100] = [0, 0, 255]

    return img_rs


class detect_lane:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.callback)
        self.counter = 0
        self.model = net.TwinLiteNet()
        # TODO: If the model was trained with only one GPU, then comment the following line
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
        self.model.load_state_dict(torch.load("pretrained/best.pth"))
        self.model.eval()

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = cv2.resize(cv_image, (800, 600))

        except CvBridgeError as e:
            print(e)

        cv_image = Run(self.model, cv_image)

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(1)


def main(args):
    detect_lane()
    rospy.init_node("get_image", anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
