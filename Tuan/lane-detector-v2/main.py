import cv2
import time

from utils.config import Config

from perspective_transform import PerspectiveTransform
from lane_fitting_v2 import LaneFittingV2
from lane_detector import LaneDetector
from lane_tracking import LaneTracking

from frame_debugger import FrameDebugger

if __name__ == "__main__":
    cfg = Config("configs/example.yaml")

    perspective_transform = PerspectiveTransform(cfg.perspective_transform)
    lane_fitting = LaneFittingV2(cfg.lane_fitting)
    lane_detector = LaneDetector(cfg.lane_detector)
    lane_tracking = LaneTracking(cfg.lane_tracking)

    cap = cv2.VideoCapture(cfg.video_path)

    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (640, 360))

            FrameDebugger.update(frame)

            new_frame_time = time.time()

            lane_frame = lane_detector.detect(frame)
            warp_frame = perspective_transform.get_sky_view(lane_frame)
            lanes = lane_fitting.fit(warp_frame)
            # lane_tracking.track(frame, lanes)

            fps = str(int(1 / (new_frame_time - prev_frame_time)))
            prev_frame_time = new_frame_time

            FrameDebugger.draw_text(fps, (10, 20), (0, 0, 0))
            FrameDebugger.show()
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
