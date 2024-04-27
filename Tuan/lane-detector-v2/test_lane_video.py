import cv2
import time

from modules.backend.frame_debugger import FrameDebugger
from modules.backend.image_transform import ImageTransform
from modules.backend.lane_fitting_v2 import LaneFittingV2
from modules.backend.lane_tracking import LaneTracking
from modules.backend.perspective_transform import PerspectiveTransform

from utils.config import Config

global prev_frame_time
global new_frame_time

prev_frame_time = 0
new_frame_time = 0


def process_frame(frame, lane_frame):
    # Image transformation
    frame = image_transform.transform(frame)

    # # Perspective transform
    warp_frame = perspective_transform.get_sky_view(frame, False)
    warp_lane_frame = perspective_transform.get_sky_view(lane_frame)

    # # Fit lanes
    lanes = lane_fitting.fit(warp_lane_frame)

    # Track left and right lanes
    lane_tracking.track(warp_frame, lanes)


def update(frame, lane_frame):
    global prev_frame_time
    global new_frame_time

    frame = cv2.resize(frame, (640, 360))
    FrameDebugger.update(frame)

    start_fps()
    dist = process_frame(frame, lane_frame)
    fps = end_fps()

    FrameDebugger.draw_text(f"{fps:.0f}", (610, 20), (255, 255, 255))
    FrameDebugger.show()

    return dist


def start_fps():
    global new_frame_time
    new_frame_time = time.time()


def end_fps():
    global prev_frame_time
    global new_frame_time
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    return fps


if __name__ == "__main__":
    cfg = Config("configs/turtlebot_day.yaml")

    image_transform = ImageTransform(cfg.image_transform)
    perspective_transform = PerspectiveTransform(cfg.perspective_transform)
    lane_fitting = LaneFittingV2(cfg.lane_fitting)
    lane_tracking = LaneTracking(cfg.lane_tracking)

    cap_original = cv2.VideoCapture(cfg.video_path)
    cap_lane = cv2.VideoCapture(cfg.lane_detector["video_path"])

    while cap_lane.isOpened() and cap_original.isOpened():
        ret_original, frame = cap_original.read()
        ret_lane, lane_frame = cap_lane.read()

        if ret_lane:
            update(frame, lane_frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
