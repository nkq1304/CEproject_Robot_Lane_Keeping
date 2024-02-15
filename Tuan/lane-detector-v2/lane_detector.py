import torch
import numpy as np
from model import TwinLite as net
import cv2

class LaneDetector:
    def __init__(self, config: dict) -> None:
        self.load_model(config['model_path'])
        self.debug = config['debug']

    def load_model(self, model_path: str):
        self.model = net.TwinLiteNet()
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def detect(self, img):
        img = cv2.resize(img, (640, 360))
        img_copy = img.copy()

        binary_img = np.zeros_like(img_copy)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)  # add a batch dimension
        img = img.cuda().float() / 255.0
        img = img.cuda()
        with torch.no_grad():
            img_out = self.model(img)
        x0 = img_out[0]
        x1 = img_out[1]

        _, da_predict = torch.max(x0, 1)
        _, ll_predict = torch.max(x1, 1)

        DA = da_predict.byte().cpu().data.numpy()[0] * 255
        LL = ll_predict.byte().cpu().data.numpy()[0] * 255

        binary_img[LL > 100] = [255, 255, 255]

        if self.debug:
            visualize_img = img_copy.copy()
            visualize_img[LL > 100] = [0, 0, 255]
            cv2.imshow('lane_detector', visualize_img)

        return binary_img