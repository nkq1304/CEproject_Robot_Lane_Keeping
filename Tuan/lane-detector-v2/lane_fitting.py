import numpy as np
import cv2 as cv

from lane_line import LaneLine
from exceptions.lane import LeftLineNotFound, RightLineNotFound, LaneNotFound
from frame_debugger import FrameDebugger

DEBUG_LANE_COLORS = [(94, 22, 117), (238, 66, 102), (255, 210, 63), (51, 115, 87)]


class LaneFitting:
    def __init__(self, config: dict) -> None:
        self.contours = config["contours"]
        self.debug = config["debug"]
        self.lanes = []

        pass

    def fit(self, frame) -> list[LaneLine]:

        contours = self.get_contour(frame)
        self.lanes = self.get_lane(frame, contours)
        self.visualize_lanes(frame)

        return self.lanes

    def is_overlapping(self, rect1, rect2) -> bool:
        rect1_center = rect1[0]
        rect2_center = rect2[0]

        rect1_half_width = rect1[1][0] / 2
        rect2_half_width = rect2[1][0] / 2

        rect1_half_height = rect1[1][1] / 2
        rect2_half_height = rect2[1][1] / 2

        x_diff = abs(rect1_center[0] - rect2_center[0])
        y_diff = abs(rect1_center[1] - rect2_center[1])

        x_overlap = x_diff < (rect1_half_width + rect2_half_width)
        y_overlap = y_diff < (rect1_half_height + rect2_half_height)

        return x_overlap and y_overlap

    def get_lane(self, frame, contours) -> list[LaneLine]:
        lanes = []
        lane_rects = []

        for contour in contours:
            if (
                cv.contourArea(contour) < self.contours["min_area"]
                or cv.contourArea(contour) > self.contours["max_area"]
            ):
                continue

            rect = cv.minAreaRect(contour)

            if any(self.is_overlapping(rect, other_rect) for other_rect in lane_rects):
                continue

            box = cv.boxPoints(rect).astype(np.int0)

            mask = np.zeros_like(frame)
            mask = cv.fillPoly(mask, [box], (255, 255, 255))
            lane_pixels = cv.bitwise_and(frame, mask)

            lane_mask = np.zeros_like(frame)
            lane_mask = cv.add(lane_mask, lane_pixels)

            nonzero = lane_mask.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            lane = LaneLine(nonzeroy, nonzerox)

            lane_rects.append(rect)
            lanes.append(lane)

        return lanes

    def get_contour(self, frame) -> np.ndarray:
        binary_frame = np.copy(frame)
        binary_frame[:240, :] = 0

        binary_frame = cv.cvtColor(binary_frame, cv.COLOR_BGR2GRAY)
        binary_frame = cv.GaussianBlur(binary_frame, (5, 5), 0)
        binary_frame = cv.threshold(binary_frame, 60, 255, cv.THRESH_BINARY)[1]

        contours, hierarchy = cv.findContours(
            binary_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
        )

        return contours

    def validate_lane(
        self,
        leftx: np.ndarray,
        lefty: np.ndarray,
        rightx: np.ndarray,
        righty: np.ndarray,
    ) -> tuple[LaneLine, LaneLine]:
        left_lane_found = leftx.size != 0 and lefty.size != 0
        right_lane_found = rightx.size != 0 and righty.size != 0

        if left_lane_found and right_lane_found:
            left_line = LaneLine(lefty, leftx)
            right_line = LaneLine(righty, rightx)
            return left_line, right_line

        if not left_lane_found and not right_lane_found:
            FrameDebugger.draw_error(LaneNotFound())
            return None, None
        elif not left_lane_found:
            FrameDebugger.draw_error(LeftLineNotFound())
            return None, LaneLine(righty, rightx)
        elif not right_lane_found:
            FrameDebugger.draw_error(RightLineNotFound())
            return LaneLine(lefty, leftx), None

    def visualize_lanes(self, img) -> None:
        if not self.debug:
            return

        start = img.shape[0] // 1.5
        end = img.shape[0] - 1
        middle = start + (end - start) // 2

        # sort lane lines by their x value at the bottom of the image
        self.lanes.sort(key=lambda lane: lane.get_x(middle))

        for i, lane in enumerate(self.lanes):
            lane_points = lane.get_points(start, end)
            lane_color = DEBUG_LANE_COLORS[i]

            for point in lane_points:
                cv.circle(img, (int(point[0]), int(point[1])), 1, lane_color, 2)

        cv.imshow("lane_fitting", img)
