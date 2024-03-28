import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

from utils.lane_line import LaneLine
from utils.window import Window

from utils.visualize import draw_lane, draw_window

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

        self.histogram_seed = config["histogram"].get("seed", 64)
        self.histogram_height_ratio = config["histogram"].get("height_ratio", 0.6)

        self.lane_max_width = config["lane"].get("max_width", 100)

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

        self.histogram = self.histogram_seeded(self.binary_frame)

        cv.imshow("binary_frame", self.binary_frame)

    def fit(self, frame) -> list[LaneLine]:
        self.window_start(frame)

        histogram = self.histogram

        width_center = self.frame_width_center
        pixel_index_on_any_lane = np.array([], dtype=np.int64)

        # Find curve on the left side
        curve = []

        for i in range(0, 15):
            max_x = self.corrector_window_start(np.argmax(histogram))

            if max_x == 0:
                break

            (
                pixel_index_on_lane,
                pixel_index_on_windows,
                pixel_index_on_any_lane,
                drawn_windows,
            ) = self.sliding_window(max_x, pixel_index_on_any_lane)

            x_found = self.nonzero_pixel_x[pixel_index_on_lane]

            # Erase a bandwidth to do confuse not other sliding windows
            if len(x_found) > 0:
                left_boundary = np.int32(
                    max([min([min(x_found), max_x - self.window_margin]), 0])
                )
                right_boundary = np.int32(
                    min(
                        [
                            max([max(x_found), max_x + self.window_margin]),
                            self.frame_width - 1,
                        ]
                    )
                )
                histogram[left_boundary:right_boundary] = 0
            else:
                left_boundary = np.int32(max([max_x - self.window_width, 0]))
                right_boundary = np.int32(
                    min([max_x + self.window_width, self.frame_width - 1])
                )
                histogram[left_boundary:right_boundary] = 0

            if len(drawn_windows) == 0:
                continue

            # Draw windows
            self.draw_windows(
                drawn_windows, DEBUG_LANE_COLORS[i % len(DEBUG_LANE_COLORS)]
            )

            curve.append(i)

        print("Lane found", len(curve))

        cv.imshow("viz_frame", self.viz_frame)

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
                drawn_windows.append(window)
                window_x_mid = np.int32(
                    np.mean(self.nonzero_pixel_x[pixel_index_on_window_i])
                )

        return (
            pixel_index_on_lane,
            pixel_index_on_windows,
            mask_found_index_on_any_lane,
            drawn_windows,
        )

    def get_next_window(self, window: Window, pixel_index_on_window) -> Window:
        if len(pixel_index_on_window) == 0:
            return window

        x = np.mean(self.nonzero_pixel_x[pixel_index_on_window])
        y = window.y - window.height

        return Window(self.window_width, self.window_height, (x, y))

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

    def pre_processing(self, frame) -> np.ndarray:
        binary_frame = np.copy(frame)

        # Pre-processing
        binary_frame = cv.cvtColor(binary_frame, cv.COLOR_BGR2GRAY)
        binary_frame = cv.GaussianBlur(binary_frame, (5, 5), 0)
        binary_frame = cv.threshold(binary_frame, 60, 255, cv.THRESH_BINARY)[1]

        return binary_frame

    def draw_windows(self, drawn_windows, window_color):
        for window in drawn_windows:
            draw_window(self.viz_frame, window, window_color)

    def histogram_seeded(self, binary_frame) -> np.ndarray:
        histogram_slice_num = self.frame_width // self.histogram_seed
        histogram_height = np.int32(self.frame_height * self.histogram_height_ratio)

        histogram = np.sum(binary_frame[self.frame_height - histogram_height :], axis=0)

        histogram_seeded = np.zeros_like(histogram)

        last_seed = self.frame_width % (self.histogram_seed * histogram_slice_num)

        for i in range(histogram_slice_num):
            left = i * self.histogram_seed
            right = (i + 1) * self.histogram_seed

            histogram_seeded[left + self.histogram_seed // 2] = np.sum(
                histogram[left:right]
            )

        if last_seed > 0:
            histogram_seeded[
                self.histogram_seed * histogram_slice_num + last_seed // 2
            ] = np.sum(
                histogram[
                    self.histogram_seed * histogram_slice_num : self.frame_width - 1
                ]
            )

        return histogram_seeded

    def corrector_window_start(self, max_x):
        max_x_at_start = max_x

        for i in range(self.num_of_windows):
            mask_empty = np.array([], dtype=np.int64)

            window = Window(
                self.window_width,
                self.window_height,
                (max_x_at_start, self.frame_height - (i + 0.5) * self.window_height),
            )

            pixel_index_for_histogram_window = self.get_pixel_index_on_window(
                window, mask_empty
            )

            if len(pixel_index_for_histogram_window) >= self.window_min_pixels:
                max_x_at_start = np.int32(
                    np.mean(self.nonzero_pixel_x[pixel_index_for_histogram_window])
                )
                break

        return max_x_at_start
