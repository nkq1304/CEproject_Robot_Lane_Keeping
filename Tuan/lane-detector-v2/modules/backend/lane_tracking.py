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
        self.max_dist_diff = config["max_dist_diff"]

        self.center_lane = None
        self.left_lane = None
        self.right_lane = None

    def track(self, frame, lanes: List[LaneLine]) -> LaneLine:
        self.tracking_lanes_v2(lanes)
        self.process_center_lane(self.left_lane, self.right_lane)

        mask_frame = self.visualize(
            frame, self.left_lane, self.right_lane, self.center_lane
        )

        self.frame_debugger(mask_frame)

        return self.center_lane

    def tracking_lanes_v1(self, lanes: List[LaneLine]) -> None:
        # Get left and right lane
        left_lane, right_lane = self.process_lanes(lanes)

        left_lane_dist_diff = (
            0
            if left_lane is None or self.left_lane is None
            else abs(left_lane.dist - self.left_lane.dist)
        )

        right_lane_dist_diff = (
            0
            if right_lane is None or self.right_lane is None
            else abs(right_lane.dist - self.right_lane.dist)
        )

        # Check if both lanes are too far from previous lanes
        if (
            left_lane_dist_diff > self.max_dist_diff
            and right_lane_dist_diff > self.max_dist_diff
        ):
            if (
                left_lane
                and self.right_lane
                and abs(left_lane.dist - self.right_lane.dist) < self.max_dist_diff
            ):
                # If previous right lane is closer to left lane, current right lane is left lane
                right_lane = left_lane
                self.right_lane = left_lane
                left_lane = None
            elif (
                right_lane
                and self.left_lane
                and abs(right_lane.dist - self.left_lane.dist) < self.max_dist_diff
            ):
                # If previous left lane is closer to right lane, current left lane is right lane
                left_lane = right_lane
                self.left_lane = right_lane
                right_lane = None

        # Shift lanes if one lane is missing
        if (
            self.left_lane is not None
            and left_lane is not None
            and abs(left_lane.dist - self.left_lane.dist) > self.max_dist_diff
        ):
            self.left_lane = self.shift_lane(right_lane, -self.center_dist * 2)
        elif left_lane is None:
            self.left_lane = self.shift_lane(right_lane, -self.center_dist * 2)
        else:
            self.left_lane = left_lane

        if (
            self.right_lane is not None
            and right_lane is not None
            and abs(right_lane.dist - self.right_lane.dist) > self.max_dist_diff
        ):
            self.right_lane = self.shift_lane(left_lane, self.center_dist * 2)
        elif right_lane is None:
            self.right_lane = self.shift_lane(left_lane, self.center_dist * 2)
        else:
            self.right_lane = right_lane

    def tracking_lanes_v2(self, lanes: List[LaneLine]) -> None:
        if self.left_lane is None and self.right_lane is None:
            # If both lanes are missing
            self.left_lane, self.right_lane = self.process_lanes(lanes)
        else:
            # TODO: Fix this. This part is for demo only
            # Find nearest lanes
            left_lane = self.find_nearest_lane(self.left_lane, lanes)
            right_lane = self.find_nearest_lane(self.right_lane, lanes)

            # Shift lanes if one lane is missing
            if left_lane is None and right_lane is not None:
                self.left_lane = self.shift_lane(self.right_lane, -self.center_dist * 2)
            elif right_lane is None and left_lane is not None:
                self.right_lane = self.shift_lane(self.left_lane, self.center_dist * 2)
            elif left_lane is not None and right_lane is not None:
                # If both lanes are found
                self.left_lane = left_lane
                self.right_lane = right_lane

    def process_lanes(self, lanes: List[LaneLine]) -> Tuple[LaneLine, LaneLine]:
        if len(lanes) == 0:
            return None, None

        if len(lanes) == 1:
            angle = lanes[0].get_angle(lanes[0].start)
            if abs(angle) < 20:
                return (lanes[0], None) if lanes[0].dist < 0 else (None, lanes[0])
            else:
                return (
                    (lanes[0], None)
                    if lanes[0].get_angle(lanes[0].start) > 0
                    else (None, lanes[0])
                )

        if len(lanes) == 2:
            return lanes[0], lanes[1]

        for i, lane in enumerate(lanes):
            if lane.dist < 0:
                continue

            return lanes[i - 1], lane

        return None, None

    def process_center_lane(self, left_lane: LaneLine, right_lane: LaneLine) -> None:
        if left_lane is None and right_lane is None:
            self.center_lane = None
            return

        if left_lane is not None and right_lane is not None:
            self.process_center_lane_based_on_both_lanes(left_lane, right_lane)
        elif left_lane is not None:
            self.center_lane = self.shift_lane(left_lane, -self.center_dist)
        elif right_lane is not None:
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
        shifted_lane.dist = lane.dist + shift

        return shifted_lane

    def find_nearest_lane(
        self, target_lane: LaneLine, lanes: List[LaneLine]
    ) -> LaneLine:
        if target_lane is None:
            return target_lane

        nearest_lane = lanes[0]

        for lane in lanes:
            if abs(target_lane.dist - lane.dist) < abs(
                target_lane.dist - nearest_lane.dist
            ):
                nearest_lane = lane

        if abs(nearest_lane.dist - target_lane.dist) > self.max_dist_diff:
            return None

        return nearest_lane

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
        left_lane_found = self.left_lane is not None
        right_lane_found = self.right_lane is not None

        mask_frame = PerspectiveTransform.get_car_view(warp_mask_frame)
        black_pixels_mask = cv2.inRange(mask_frame, (0, 0, 0), (0, 0, 0))
        FrameDebugger.frame[black_pixels_mask != 255] = mask_frame[
            black_pixels_mask != 255
        ]

        dist = 0 if self.center_lane is None else self.center_lane.dist
        angle = 0 if self.center_lane is None else self.center_lane.get_angle()

        FrameDebugger.draw_tracking_info(angle, dist)

        if not left_lane_found and not right_lane_found:
            FrameDebugger.draw_error(LaneNotFound())
        elif not left_lane_found:
            FrameDebugger.draw_error(LeftLineNotFound())
        elif not right_lane_found:
            FrameDebugger.draw_error(RightLineNotFound())
