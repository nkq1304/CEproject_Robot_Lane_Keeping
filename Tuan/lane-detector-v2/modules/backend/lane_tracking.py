import numpy as np
import cv2
import math

from typing import List, Tuple

from utils.lane_line import LaneLine
from exceptions.lane import LeftLineNotFound, RightLineNotFound, LaneNotFound

from modules.backend.frame_debugger import FrameDebugger
from modules.backend.image_publisher import ImagePublisher

from utils.visualize import draw_lane, draw_intersection


class LaneTracking:
    def __init__(self, config: dict) -> None:
        self.center_dist = config["center_dist"]
        self.debug = config["debug"]

        self.center_lane = None
        self.prev_left_lane = None
        self.prev_right_lane = None

        self.left_lane_realiability = 100
        self.right_lane_realiability = 100

    def track(self, frame, lanes: List[LaneLine]) -> Tuple[LaneLine, LaneLine]:
        left_lane, right_lane = self.process_lanes(lanes)

        self.kalman_filter(left_lane, right_lane)
        self.update_lane_reliability()
        self.process_center_lane(self.prev_left_lane, self.prev_right_lane)

        mask_frame = self.visualize(
            frame, self.prev_left_lane, self.prev_right_lane, self.center_lane
        )

        return self.center_lane, mask_frame

    def process_lanes(self, lanes: List[LaneLine]) -> Tuple[LaneLine, LaneLine]:
        if len(lanes) == 0:
            return None, None

        if len(lanes) == 1:
            if lanes[0].dist < 0:
                return lanes[0], None
            else:
                return None, lanes[0]

        for i, lane in enumerate(lanes):
            if lane.dist < 0:
                continue

            return lanes[i - 1], lane

        return None, None

    def visualize(
        self, img, left_lane: LaneLine, right_lane: LaneLine, center_lane: LaneLine
    ) -> None:
        if not self.debug:
            return

        viz_frame = img.copy()
        mask_frame = np.zeros_like(viz_frame)

        if left_lane is not None:
            draw_lane(viz_frame, left_lane)
            draw_lane(mask_frame, left_lane)

        if right_lane is not None:
            draw_lane(viz_frame, right_lane, (255, 0, 0))
            draw_lane(mask_frame, right_lane, (255, 0, 0))

        if center_lane is not None:
            draw_lane(viz_frame, center_lane, (0, 255, 255))
            draw_lane(mask_frame, center_lane, (0, 255, 255))

        if ImagePublisher.lane_tracking is not None:
            ImagePublisher.publish_lane_tracking(viz_frame)
        else:
            cv2.imshow("lane_tracking", viz_frame)

        return mask_frame

    def kalman_filter(self, left_lane: LaneLine, right_lane: LaneLine) -> None:
        if left_lane is not None:
            if self.prev_left_lane is not None:
                if abs(left_lane.dist - self.prev_left_lane.dist) < 30:
                    self.prev_left_lane = left_lane
                else:
                    self.left_lane_realiability -= 5
            elif self.prev_left_lane is None:
                self.prev_left_lane = left_lane

        if right_lane is not None:
            if self.prev_right_lane is not None:
                if abs(right_lane.dist - self.prev_right_lane.dist) < 30:
                    self.prev_right_lane = right_lane
                else:
                    self.right_lane_realiability -= 5
            elif self.prev_right_lane is None:
                self.prev_right_lane = right_lane

    def update_lane_reliability(self) -> None:
        if self.prev_left_lane is not None:
            self.left_lane_realiability += (
                5 if self.prev_left_lane.get_length() > 180 else -5
            )

        if self.prev_right_lane is not None:
            self.right_lane_realiability += (
                5 if self.prev_right_lane.get_length() > 180 else -5
            )

        self.left_lane_realiability = np.clip(self.left_lane_realiability, 0, 100)
        self.right_lane_realiability = np.clip(self.right_lane_realiability, 0, 100)

    def process_center_lane(self, left_lane: LaneLine, right_lane: LaneLine) -> None:
        if left_lane is None and right_lane is None:
            self.center_lane = None
            return

        if left_lane is not None and right_lane is not None:
            if self.left_lane_realiability > 50 and self.right_lane_realiability > 50:
                self.process_center_lane_based_on_both_lanes(left_lane, right_lane)
            elif self.left_lane_realiability > 50:
                self.process_center_lane_based_on_left_lane(left_lane)
            elif self.right_lane_realiability > 50:
                self.process_center_lane_based_on_right_lane(right_lane)
        elif left_lane is not None and self.left_lane_realiability > 50:
            self.process_center_lane_based_on_left_lane(left_lane)
        elif right_lane is not None and self.right_lane_realiability > 50:
            self.process_center_lane_based_on_right_lane(right_lane)

    def process_center_lane_based_on_left_lane(self, left_lane: LaneLine) -> None:
        left_points = left_lane.get_points()
        # Shift the left points x coordinate to the right by 100 pixels
        center_points = [(x + self.center_dist, y) for x, y in left_points]

        self.center_lane = LaneLine.from_points(center_points)
        self.center_lane.start = left_lane.start
        self.center_lane.end = left_lane.end
        self.center_lane.dist = left_lane.dist + self.center_dist

    def process_center_lane_based_on_right_lane(self, right_lane: LaneLine) -> None:
        right_points = right_lane.get_points()
        # Shift the right points x coordinate to the left by 100 pixels
        center_points = [(x - self.center_dist, y) for x, y in right_points]

        self.center_lane = LaneLine.from_points(center_points)
        self.center_lane.start = right_lane.start
        self.center_lane.end = right_lane.end
        self.center_lane.dist = right_lane.dist - self.center_dist

    def process_center_lane_based_on_both_lanes(
        self, left_lane: LaneLine, right_lane: LaneLine
    ) -> None:
        self.center_lane = LaneLine.from_coefficients(
            (left_lane.a + right_lane.a) / 2,
            (left_lane.b + right_lane.b) / 2,
            (left_lane.c + right_lane.c) / 2,
        )

        self.center_lane.start = (left_lane.start + right_lane.start) // 2
        self.center_lane.end = (left_lane.end + right_lane.end) // 2
        self.center_lane.dist = (left_lane.dist + right_lane.dist) / 2

    def frame_debugger(self) -> None:
        left_lane_found = self.prev_left_lane is not None
        right_lane_found = self.prev_right_lane is not None

        if not left_lane_found and not right_lane_found:
            FrameDebugger.draw_error(LaneNotFound())
        elif not left_lane_found:
            FrameDebugger.draw_error(LeftLineNotFound())
        elif not right_lane_found:
            FrameDebugger.draw_error(RightLineNotFound())
