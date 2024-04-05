import cv2

from utils.config import Config
from modules.backend.backend import Backend

if __name__ == "__main__":
    cfg = Config("configs/example.yaml")

    backend = Backend(cfg)

    cap = cv2.VideoCapture(cfg.video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            backend.process_frame(frame)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
