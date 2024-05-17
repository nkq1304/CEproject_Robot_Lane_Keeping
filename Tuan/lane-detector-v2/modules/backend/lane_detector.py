import torch
import numpy as np
from model import TwinLite as net
import cv2


from utils.tracker import Tracker
from modules.backend.image_publisher import ImagePublisher


class LaneDetector:
    def __init__(self, config: dict) -> None:
        self.debug = config["debug"]
        self.video_path = config["video_path"]
        self.save_video = config["save_video"]

        self.cuda = torch.cuda.is_available()
        self.tracker = Tracker("Lane Detector")

        self.create_video_writer()
        self.load_model(config["model_path"])

    def create_video_writer(self) -> None:
        if not self.save_video or self.video_path == "":
            return

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 60.0, (640, 360))

    def load_model(self, model_path: str):
        self.model = net.TwinLiteNet()
        self.model = torch.nn.DataParallel(self.model)

        if self.cuda:
            self.model = self.model.cuda()
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )

        self.model.eval()

    def detect(self, img):
        self.tracker.start()

        img = cv2.resize(img, (640, 360))
        img_copy = img.copy()

        binary_img = np.zeros_like(img_copy)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)  # add a batch dimension

        if self.cuda:
            img = img.cuda().float() / 255.0
            img = img.cuda()
        else:
            img = img / 255.0

        with torch.no_grad():
            img_out = self.model(img)
        x0 = img_out[0]
        x1 = img_out[1]

        _, da_predict = torch.max(x0, 1)
        _, ll_predict = torch.max(x1, 1)

        DA = da_predict.byte().cpu().data.numpy()[0] * 255
        LL = ll_predict.byte().cpu().data.numpy()[0] * 255

        binary_img[LL > 100] = [255, 255, 255]

        self.tracker.end()
        self.visualize(img_copy, LL)
        self.save_lane_video(binary_img)

        return binary_img

    def visualize(self, img, LL):
        if not self.debug:
            return

        visualize_img = img.copy()
        visualize_img[LL > 100] = [0, 0, 255]

        if ImagePublisher.lane_detector is not None:
            ImagePublisher.publish_lane_detector(visualize_img)
        else:
            cv2.imshow("lane_detector", visualize_img)

    def save_lane_video(self, frame):
        if not self.save_video or self.video_path == "":
            return

        self.video_writer.write(frame)
