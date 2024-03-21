import numpy as np
import cv2

from lane_line import LaneLine
from exceptions.lane import LeftLineNotFound, RightLineNotFound, LaneNotFound
from frame_debugger import FrameDebugger

from utils.visualize import draw_lane, draw_intersection


class LaneTracking:
    def __init__(self, config: dict) -> None:
        self.debug = config["debug"]
        self.prev_left_lane = None
        self.prev_right_lane = None

    def track(self, frame, lanes: list[LaneLine]) -> None:
        left_lane, right_lane = self.process_frame(frame, lanes)

        self.kalman_filter(left_lane, right_lane)
        self.visualize(frame, self.prev_left_lane, self.prev_right_lane)
        self.frame_debugger()

        pass

    def process_frame(self, frame, lanes: list[LaneLine]) -> tuple[LaneLine, LaneLine]:
        y = frame.shape[0] * 0.75
        mid_x = frame.shape[1] // 2

        if len(lanes) == 0:
            return None, None

        if len(lanes) == 1:
            if lanes[0].get_x(y) < mid_x:
                return lanes[0], None
            else:
                return None, lanes[0]

        for i, lane in enumerate(lanes):
            if lane.get_x(y) < mid_x:
                continue

            return lanes[i - 1], lane

        return None, None

    def visualize(self, img, left_lane: LaneLine, right_lane: LaneLine) -> None:
        if not self.debug:
            return

        viz_frame = img.copy()

        start = img.shape[0] // 1.5
        end = img.shape[0] - 1

        intersection = left_lane.get_intersection(right_lane, viz_frame)

        draw_lane(viz_frame, left_lane, start, end)
        draw_lane(viz_frame, right_lane, start, end, (255, 0, 0))
        draw_intersection(viz_frame, intersection)

        cv2.imshow("lane_tracking", viz_frame)

    def kalman_filter(self, left_lane: LaneLine, right_lane: LaneLine) -> None:
        if left_lane is not None:
            self.prev_left_lane = left_lane

        if right_lane is not None:
            self.prev_right_lane = right_lane

    def frame_debugger(self) -> None:
        left_lane_found = self.prev_left_lane is not None
        right_lane_found = self.prev_right_lane is not None

        if not left_lane_found and not right_lane_found:
            FrameDebugger.draw_error(LaneNotFound())
        elif not left_lane_found:
            FrameDebugger.draw_error(LeftLineNotFound())
        elif not right_lane_found:
            FrameDebugger.draw_error(RightLineNotFound())
