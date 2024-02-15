import cv2
import numpy as np

from utils.config import Config
from perspective_transform import PerspectiveTransform
from lane_fitting import LaneFitting
from lane_detector import LaneDetector
from frame_debugger import FrameDebugger

if __name__ == '__main__':
    cfg = Config('configs/example.yaml')

    perspective_transform = PerspectiveTransform(cfg.perspective_transform)
    lane_fitting = LaneFitting(cfg.lane_fitting)
    lane_detector = LaneDetector(cfg.lane_detector)

    cap = cv2.VideoCapture(cfg.video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (640, 360))

            FrameDebugger.update(frame)
            
            lane_frame = lane_detector.detect(frame)
            warped_frame = perspective_transform.warp(lane_frame)
            left_line, right_line = lane_fitting.fit(warped_frame)

            FrameDebugger.draw_lane(left_line, right_line)

            FrameDebugger.show()
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

