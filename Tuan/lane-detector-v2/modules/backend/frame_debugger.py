import cv2 as cv
import numpy as np

from exceptions.lane import LaneException


class FrameDebugger:
    frame = None
    font_scale = 0.5
    thickness = 1
    font = cv.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def update(frame) -> None:
        FrameDebugger.frame = frame

    @staticmethod
    def show() -> None:
        cv.imshow("frame_debugger", FrameDebugger.frame)

    @staticmethod
    def draw_error(exception: Exception) -> None:
        if isinstance(exception, LaneException):
            FrameDebugger.draw_lane_error(exception)

    @staticmethod
    def draw_lane_error(exception: LaneException) -> None:
        pos = (10, 20)
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
