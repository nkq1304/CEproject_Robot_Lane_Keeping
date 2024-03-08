import cv2
import time

from utils.config import Config
from lane_fitting import LaneFitting
from lane_detector import LaneDetector
from frame_debugger import FrameDebugger

if __name__ == "__main__":
    cfg = Config("configs/example.yaml")

    lane_fitting = LaneFitting(cfg.lane_fitting)
    lane_detector = LaneDetector(cfg.lane_detector)

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
            lane_fitting.fit(lane_frame)

            fps = str(int(1 / (new_frame_time - prev_frame_time)))
            prev_frame_time = new_frame_time

            FrameDebugger.draw_text(fps, (10, 20), (0, 0, 0))
            FrameDebugger.show()
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
