import cv2

from utils.tracker import Tracker
from utils.lane_line import LaneLine

from modules.backend.lane_detector import LaneDetector
from modules.backend.lane_tracking import LaneTracking
from modules.backend.frame_debugger import FrameDebugger
from modules.backend.image_transform import ImageTransform
from modules.backend.lane_fitting_v2 import LaneFittingV2
from modules.backend.perspective_transform import PerspectiveTransform


class Backend:
    def __init__(self, cfg) -> None:
        self.image_transform = ImageTransform(cfg.image_transform)
        self.perspective_transform = PerspectiveTransform(cfg.perspective_transform)
        self.lane_fitting = LaneFittingV2(cfg.lane_fitting)
        self.lane_detector = LaneDetector(cfg.lane_detector)
        self.lane_tracking = LaneTracking(cfg.lane_tracking)

        self.tracker = Tracker("Backend")

    def update(self, frame) -> LaneLine:
        frame = cv2.resize(frame, (640, 360))
        FrameDebugger.update(frame)

        self.tracker.start()
        center_lane = self.process_frame(frame)
        self.tracker.end()

        FrameDebugger.draw_text(f"{self.tracker.fps():.0f}", (610, 20), (255, 255, 255))
        FrameDebugger.show()

        return center_lane

    def process_frame(self, frame) -> LaneLine:
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
        center_lane = self.lane_tracking.track(warp_frame, lanes)

        return center_lane
