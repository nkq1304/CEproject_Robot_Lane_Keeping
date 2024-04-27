import cv2

from utils.config import Config
from modules.backend.backend import Backend

if __name__ == "__main__":
    cfg = Config("configs/turtlebot_day.yaml")

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
