import cv2 as cv
import numpy as np

from exceptions.lane import LaneException
from lane_line import LaneLine
from perspective_transform import PerspectiveTransform

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
        if (isinstance(exception, LaneException)):
            FrameDebugger.draw_lane_error(exception)

    @staticmethod
    def draw_lane_error(exception: LaneException) -> None:
        pos = (10, 20)
        color = (0, 0, 255)
        text = 'LaneError: ' + exception.message

        FrameDebugger.draw_text(text, pos, color)

    @staticmethod 
    def draw_lane(left_line: LaneLine, right_line: LaneLine) -> None:
        frame = FrameDebugger.frame

        start = PerspectiveTransform.get_top()
        end = PerspectiveTransform.get_bottom()

        left_line_points = left_line.get_points(start, end)
        right_line_points = right_line.get_points(start, end)

        layer = np.zeros_like(frame)

        for point in left_line_points:
            cv.circle(layer, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)

        for point in right_line_points:
            cv.circle(layer, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

        cv.addWeighted(layer, 0.9, frame, 1, 0, frame)

    @staticmethod
    def draw_text(text: str, pos: tuple, color: tuple) -> None:
        frame = FrameDebugger.frame
        font = FrameDebugger.font
        thickness = FrameDebugger.thickness
        font_scale = FrameDebugger.font_scale

        cv.putText(frame, text, pos, font, font_scale, color, thickness, cv.LINE_AA)