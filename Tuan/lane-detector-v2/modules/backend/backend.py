import cv2
import time

from utils.config import Config

from modules.backend.image_transform import ImageTransform
from modules.backend.perspective_transform import PerspectiveTransform
from modules.backend.lane_fitting_v2 import LaneFittingV2
from modules.backend.lane_detector import LaneDetector
from modules.backend.lane_tracking import LaneTracking

from modules.backend.frame_debugger import FrameDebugger


class Backend:
    def __init__(self, cfg) -> None:
        self.image_transform = ImageTransform(cfg.image_transform)
        self.perspective_transform = PerspectiveTransform(cfg.perspective_transform)
        self.lane_fitting = LaneFittingV2(cfg.lane_fitting)
        self.lane_detector = LaneDetector(cfg.lane_detector)
        self.lane_tracking = LaneTracking(cfg.lane_tracking)

        self.prev_frame_time = 0
        self.new_frame_time = 0

    def update(self, frame) -> float:
        frame = cv2.resize(frame, (640, 360))
        FrameDebugger.update(frame)

        self.start_fps()
        dist = self.process_frame(frame)
        fps = self.end_fps()

        FrameDebugger.draw_text(f"{fps:.0f}", (610, 20), (255, 255, 255))
        FrameDebugger.show()

        return dist

    def start_fps(self):
        self.new_frame_time = time.time()

    def end_fps(self):
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time

        return fps

    def process_frame(self, frame) -> float:
        # # Image transformation
        frame = self.image_transform.transform(frame)

        # # Detect lanes with TwinLiteNet
        lane_frame = self.lane_detector.detect(frame)

        # # Perspective transform
        warp_frame = self.perspective_transform.get_sky_view(frame, False)
        warp_lane_frame = self.perspective_transform.get_sky_view(lane_frame)

        # # Fit lanes
        lanes = self.lane_fitting.fit(warp_lane_frame)

        # # Track left and right lanes
        dist = self.lane_tracking.track(warp_frame, lanes)

        return dist
