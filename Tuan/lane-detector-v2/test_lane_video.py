import cv2
import time

from argparse import ArgumentParser

from modules.backend.frame_debugger import FrameDebugger
from modules.backend.image_transform import ImageTransform
from modules.backend.lane_fitting_v2 import LaneFittingV2
from modules.backend.lane_tracking import LaneTracking
from modules.backend.perspective_transform import PerspectiveTransform

from utils.config import Config
from utils.tracker import Tracker


def process_frame(frame, lane_frame) -> None:
    # Image transformation
    frame = image_transform.transform(frame)

    # # Perspective transform
    warp_frame = perspective_transform.get_sky_view(frame, False)
    warp_lane_frame = perspective_transform.get_sky_view(lane_frame)

    # # Fit lanes
    lanes = lane_fitting.fit(warp_lane_frame)

    # Track left and right lanes
    lane_tracking.track(warp_frame, lanes)


def update(frame, lane_frame) -> None:
    frame = cv2.resize(frame, (640, 360))
    FrameDebugger.update(frame)

    backend_tracker.start()
    process_frame(frame, lane_frame)
    backend_tracker.end()

    FrameDebugger.draw_text(f"{backend_tracker.fps():.0f}", (610, 20), (255, 255, 255))
    FrameDebugger.show()


def arg_parse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/turtlebot.yaml",
        help="Config file path",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    cfg = Config(args.config)

    backend_tracker = Tracker("Backend")

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
