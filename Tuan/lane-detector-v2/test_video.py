import cv2

from utils.config import Config
from utils.tracker import TrackerManager

from modules.backend.backend import Backend

from argparse import ArgumentParser


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/turtlebot.yaml",
        help="Config file path",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = Config(args.config)

    backend = Backend(cfg)

    cap = cv2.VideoCapture(cfg.video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            backend.update(frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    TrackerManager.stop()
