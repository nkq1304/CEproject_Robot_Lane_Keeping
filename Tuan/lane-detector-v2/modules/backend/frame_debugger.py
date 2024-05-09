import cv2 as cv

from exceptions.lane import LaneException
from utils.lane_line import LaneLine

from modules.backend.image_publisher import ImagePublisher


class FrameDebugger:
    frame = None
    font_scale = 0.6
    thickness = 1
    font = cv.FONT_HERSHEY_DUPLEX
    center_lane: LaneLine = None

    @staticmethod
    def update(frame) -> None:
        FrameDebugger.frame = frame

    @staticmethod
    def show() -> None:
        if ImagePublisher.frame_debugger:
            ImagePublisher.publish_frame_debugger(FrameDebugger.frame)
        else:
            cv.imshow("frame_debugger", FrameDebugger.frame)

    @staticmethod
    def draw_error(exception: Exception) -> None:
        if isinstance(exception, LaneException):
            FrameDebugger.draw_lane_error(exception)

    @staticmethod
    def draw_lane_error(exception: LaneException) -> None:
        pos = (10, 340)
        color = (0, 0, 255)
        text = "LaneError: " + exception.message

        FrameDebugger.draw_text(text, pos, color)

    @staticmethod
    def draw_text(text: str, pos: tuple, color: tuple) -> None:
        frame = FrameDebugger.frame
        font = FrameDebugger.font
        thickness = FrameDebugger.thickness
        font_scale = FrameDebugger.font_scale

        cv.putText(frame, text, pos, font, font_scale, color, thickness, cv.LINE_AA)
        return frame

    @staticmethod
    def draw_rectangle(rect: tuple, color: tuple, opacity: float) -> None:
        frame = FrameDebugger.frame
        overlay = frame.copy()
        cv.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
        cv.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        return frame

    @staticmethod
    def draw_tracking_info(angle: float, dist: float) -> None:
        FrameDebugger.draw_rectangle((0, 0, 640, 60), (0, 0, 0), 0.3)
        FrameDebugger.draw_text(
            f"Angle : {angle:.2f} deg",
            (10, 20),
            (255, 255, 255),
        )
        FrameDebugger.draw_text(f"Distance : {dist:.2f}", (10, 45), (255, 255, 255))
