import numpy as np
import cv2

from typing import List, Tuple

from modules.backend.perspective_transform import PerspectiveTransform
from utils.lane_line import LaneLine
from exceptions.lane import LeftLineNotFound, RightLineNotFound, LaneNotFound

from modules.backend.frame_debugger import FrameDebugger
from modules.backend.image_publisher import ImagePublisher

from utils.visualize import draw_lane


class LaneTracking:
    def __init__(self, config: dict) -> None:
        self.center_dist = config["center_dist"]

        self.center_lane = None
        self.prev_left_lane = None
        self.prev_right_lane = None

        self.left_lane_reliability = 100
        self.right_lane_reliability = 100

    def track(self, frame, lanes: List[LaneLine]) -> float:
        left_lane, right_lane = self.process_lanes(lanes)

        self.kalman_filter(left_lane, right_lane)
        self.update_lane_reliability()
        self.process_center_lane(self.prev_left_lane, self.prev_right_lane)

        mask_frame = self.visualize(
            frame, self.prev_left_lane, self.prev_right_lane, self.center_lane
        )

        self.frame_debugger(mask_frame)

        if self.center_lane is not None:
            return self.center_lane.dist

        return 0

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

    def kalman_filter(self, left_lane: LaneLine, right_lane: LaneLine) -> None:

        if (
            self.prev_left_lane is not None
            and left_lane is not None
            and abs(left_lane.dist - self.prev_left_lane.dist) > 30
        ):
            self.prev_left_lane = None
        else:
            self.prev_left_lane = left_lane

        if (
            self.prev_right_lane is not None
            and right_lane is not None
            and abs(right_lane.dist - self.prev_right_lane.dist) > 30
        ):
            self.prev_right_lane = None
        else:
            self.prev_right_lane = right_lane

    def update_lane_reliability(self) -> None:
        if self.prev_left_lane is not None:
            self.left_lane_reliability += (
                5 if self.prev_left_lane.get_length() > 180 else -5
            )

        if self.prev_right_lane is not None:
            self.right_lane_reliability += (
                5 if self.prev_right_lane.get_length() > 180 else -5
            )

        self.left_lane_reliability = np.clip(self.left_lane_reliability, 0, 100)
        self.right_lane_reliability = np.clip(self.right_lane_reliability, 0, 100)

    def process_center_lane(self, left_lane: LaneLine, right_lane: LaneLine) -> None:
        if left_lane is None and right_lane is None:
            self.center_lane = None
            return

        if left_lane is not None and right_lane is not None:
            if self.left_lane_reliability > 50 and self.right_lane_reliability > 50:
                self.process_center_lane_based_on_both_lanes(left_lane, right_lane)
            elif self.left_lane_reliability > 50:
                self.center_lane = self.shift_lane(left_lane, -self.center_dist)
            elif self.right_lane_reliability > 50:
                self.center_lane = self.shift_lane(right_lane, self.center_dist)
        elif left_lane is not None and self.left_lane_reliability > 50:
            self.center_lane = self.shift_lane(left_lane, -self.center_dist)
        elif right_lane is not None and self.right_lane_reliability > 50:
            self.center_lane = self.shift_lane(right_lane, self.center_dist)

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

    def shift_lane(self, lane: LaneLine, shift: int) -> LaneLine:
        if lane is None:
            return None

        points = lane.get_points()
        shifted_points = [(x + shift, y) for x, y in points]

        shifted_lane = LaneLine.from_points(shifted_points)
        shifted_lane.start = lane.start
        shifted_lane.end = lane.end
        shifted_lane.dist = lane.dist - shift

        return shifted_lane

    def visualize(
        self, img, left_lane: LaneLine, right_lane: LaneLine, center_lane: LaneLine
    ) -> None:
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

    def frame_debugger(self, warp_mask_frame) -> None:
        left_lane_found = self.prev_left_lane is not None
        right_lane_found = self.prev_right_lane is not None

        mask_frame = PerspectiveTransform.get_car_view(warp_mask_frame)
        black_pixels_mask = cv2.inRange(mask_frame, (0, 0, 0), (0, 0, 0))
        FrameDebugger.frame[black_pixels_mask != 255] = mask_frame[
            black_pixels_mask != 255
        ]

        dist = 0 if self.center_lane is None else self.center_lane.dist
        curvature = 0 if self.center_lane is None else self.center_lane.get_curvature()

        FrameDebugger.draw_tracking_info(curvature, dist)

        if not left_lane_found and not right_lane_found:
            FrameDebugger.draw_error(LaneNotFound())
        elif not left_lane_found:
            FrameDebugger.draw_error(LeftLineNotFound())
        elif not right_lane_found:
            FrameDebugger.draw_error(RightLineNotFound())
