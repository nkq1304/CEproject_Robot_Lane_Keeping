import cv2 as cv
import numpy as np

from exceptions.lane import LaneException
from utils.lane_line import LaneLine


class FrameDebugger:
    frame = None
    font_scale = 0.8
    thickness = 1
    font = cv.FONT_HERSHEY_DUPLEX
    center_lane: LaneLine = None

    @staticmethod
    def update(frame, center_lane: LaneLine) -> None:
        FrameDebugger.frame = frame
        FrameDebugger.center_lane = center_lane

    @staticmethod
    def show() -> None:
        center_lane = FrameDebugger.center_lane

        if center_lane is not None:
            FrameDebugger.draw_text(
                f"Curvature Radius : {center_lane.get_curvature():.2f}",
                (10, 30),
                (255, 0, 0),
            )
            FrameDebugger.draw_text(
                f"Distance : {center_lane.dist:.2f}", (10, 60), (255, 0, 0)
            )

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
        return frame
