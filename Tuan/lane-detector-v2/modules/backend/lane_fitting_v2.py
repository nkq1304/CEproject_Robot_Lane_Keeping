import numpy as np
import cv2 as cv

from typing import List

from utils.window import Window
from utils.tracker import Tracker
from utils.lane_line import LaneLine
from utils.visualize import draw_lane, draw_window

from modules.backend.image_publisher import ImagePublisher

DEBUG_LANE_COLORS = [
    (94, 22, 117),
    (238, 66, 102),
    (255, 210, 63),
    (51, 115, 87),
    (0, 127, 115),
    (255, 210, 63),
]


class LaneFittingV2:
    def __init__(self, config: dict) -> None:
        self.debug = config["debug"]

        self.window_height = config["window"].get("height", 40)
        self.window_width = config["window"].get("width", 20)
        self.window_margin = config["window"].get("margin_x", 50)
        self.window_min_pixels = config["window"].get("min_pixels", 50)
        self.window_max_pixels = config["window"].get("max_pixels", 800)

        self.lane_max_width = config["lane"].get("max_width", 100)
        self.max_lanes = config["lane"].get("max_lanes", 2)

        self.tracker = Tracker("Lane Fitting")

    def window_start(self, frame) -> None:
        self.frame = frame
        self.viz_frame = frame.copy()
        self.binary_frame = self.pre_processing(frame)

        self.frame_width = self.binary_frame.shape[1]
        self.frame_height = self.binary_frame.shape[0]
        self.frame_width_center = self.frame_width // 2

        self.num_of_windows = self.frame_height // self.window_height

        self.nonzero_pixel_x = np.array(np.nonzero(self.binary_frame)[1])
        self.nonzero_pixel_y = np.array(np.nonzero(self.binary_frame)[0])

    def fit(self, frame) -> List[LaneLine]:
        self.tracker.start()

        self.window_start(frame)

        pixel_index_on_any_lane = np.array([], dtype=np.int64)

        lanes: list[LaneLine] = []

        first_init = True

        for i in range(0, self.max_lanes * 2):
            start_x = self.get_start_x(pixel_index_on_any_lane, first_init)
            first_init = False

            if start_x is None:
                break

            (
                pixel_index_on_lane,
                pixel_index_on_windows,
                pixel_index_on_any_lane,
                drawn_windows,
            ) = self.sliding_window(start_x, pixel_index_on_any_lane)

            if len(drawn_windows) < self.num_of_windows * 2 // 3:
                continue

            lane = self.lane_extrapolation(pixel_index_on_lane, pixel_index_on_windows)

            if lane is None:
                continue

            lanes.append(lane)
            lane.dist = lane.get_x(0) - self.frame_width_center
            lane.drawn_windows = drawn_windows

            if len(lanes) >= self.max_lanes:
                break

        lanes.sort(key=lambda lane: lane.dist)

        self.tracker.end()
        self.visualize(lanes)

        return lanes

    def get_start_x(self, mask_found_index_on_any_lane, first_init: False) -> int:
        window_height = self.window_height
        window_width = self.frame_width // 2 if first_init else self.frame_width
        window_x = (
            self.frame_width_center + window_width // 2
            if first_init
            else self.frame_width_center
        )

        init_window = Window(
            window_width,
            window_height,
            (window_x, self.frame_height - window_height / 2),
        )

        for i in range(self.frame_height // window_height):
            init_window.set_y(int(self.frame_height - (i + 0.5) * window_height))

            pixel_index_on_window = self.get_pixel_index_on_window(
                init_window, mask_found_index_on_any_lane
            )

            if len(pixel_index_on_window) > 0:
                return np.int32(np.mean(self.nonzero_pixel_x[pixel_index_on_window]))

        return 0

    def sliding_window(self, window_x_mid, mask_found_index_on_any_lane):
        pixel_index_on_lane = np.array([], dtype=np.int64)
        pixel_index_on_windows = []
        drawn_windows = []

        for window_i in range(self.num_of_windows):
            window = Window(
                self.window_width,
                self.window_height,
                (
                    window_x_mid,
                    self.frame_height - (window_i + 0.5) * self.window_height,
                ),
            )

            pixel_index_on_window_i = self.get_pixel_index_on_window(
                window, mask_found_index_on_any_lane
            )
            mask_found_index_on_any_lane = np.append(
                mask_found_index_on_any_lane, pixel_index_on_window_i
            )

            pixel_index_on_lane = np.append(
                pixel_index_on_lane, pixel_index_on_window_i
            )
            pixel_index_on_windows.append(pixel_index_on_window_i)

            if len(self.nonzero_pixel_x[pixel_index_on_window_i]) > 0:
                window.set_x(
                    np.int32(np.mean(self.nonzero_pixel_x[pixel_index_on_window_i]))
                )
                drawn_windows.append(window)

            if len(pixel_index_on_window_i) >= self.window_min_pixels:
                window_x_mid = np.int32(
                    np.mean(self.nonzero_pixel_x[pixel_index_on_window_i])
                )
            elif len(drawn_windows) >= 2:
                window_x_mid += drawn_windows[-1].x - drawn_windows[-2].x

        return (
            pixel_index_on_lane,
            pixel_index_on_windows,
            mask_found_index_on_any_lane,
            drawn_windows,
        )

    def get_pixel_index_on_window(self, window: Window, mask_found_index_on_any_line):
        mask_vertical = (self.nonzero_pixel_y <= window.bottom) & (
            self.nonzero_pixel_y >= window.top
        )
        mask_horizontal = (self.nonzero_pixel_x <= window.right) & (
            self.nonzero_pixel_x >= window.left
        )

        mask_inside = np.zeros_like(mask_vertical, dtype=np.uint8)
        mask_inside[mask_vertical & mask_horizontal] = 1
        mask_inside[mask_found_index_on_any_line] = 0

        x_inside = self.nonzero_pixel_x[np.nonzero(mask_inside)[0][:]]

        if len(x_inside) > 0:
            if (len(np.nonzero(mask_inside)[0][:]) > self.window_max_pixels) | (
                abs(max(x_inside) - min(x_inside)) > self.lane_max_width
            ):
                mask_inside[:]

        return np.nonzero(mask_inside)[0][:]

    def lane_extrapolation(
        self, pixel_index_on_lane, pixel_index_on_windows
    ) -> LaneLine:
        if len(pixel_index_on_lane) == 0:
            return None

        x = self.nonzero_pixel_x[pixel_index_on_lane]
        y = self.nonzero_pixel_y[pixel_index_on_lane]

        if len(x) < 3:
            return None

        lane = LaneLine(y, x)
        lane.start = y[np.argmax(y)]
        lane.end = y[np.argmin(y)]

        return lane

    def pre_processing(self, frame) -> np.ndarray:
        binary_frame = np.copy(frame)
        binary_frame = cv.cvtColor(binary_frame, cv.COLOR_BGR2GRAY)

        return binary_frame

    def visualize(self, lanes):
        if not self.debug:
            return

        for i, lane in enumerate(lanes):
            color = DEBUG_LANE_COLORS[i % len(DEBUG_LANE_COLORS)]

            draw_lane(self.viz_frame, lane, color)

            for window in lane.drawn_windows:
                draw_window(self.viz_frame, window, color)

        if ImagePublisher.lane_fitting is not None:
            ImagePublisher.publish_lane_fitting(self.viz_frame)
        else:
            cv.imshow("lane_fitting_v2", self.viz_frame)
