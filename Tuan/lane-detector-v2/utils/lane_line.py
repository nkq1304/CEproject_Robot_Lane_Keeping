import numpy as np
from typing import List, Tuple


class LaneLine:
    def __init__(self, lefty, leftx, start=0, end=0, dist=0) -> None:
        poly_fit = np.polyfit(lefty, leftx, 2)

        self.a = poly_fit[0]
        self.b = poly_fit[1]
        self.c = poly_fit[2]

        self.start = start
        self.end = end

        self.dist = dist
        self.abs_dist = abs(dist)

        self.drawn_windows = []

    @classmethod
    def from_coefficients(cls, a, b, c) -> "LaneLine":
        lane = cls.__new__(cls)
        lane.a = a
        lane.b = b
        lane.c = c
        return lane

    @classmethod
    def from_points(cls, points: List[Tuple[int, int]]) -> "LaneLine":
        leftx, lefty = zip(*points)
        return cls(lefty, leftx)

    def get_x(self, y: int) -> int:
        return self.a * y**2 + self.b * y + self.c

    def get_points(self, start, end, step=None) -> List[Tuple[int, int]]:
        self.start = start
        self.end = end

        return self.get_points(step)

    def get_points(self, step=None) -> List[Tuple[int, int]]:
        step = abs(self.end - self.start) + 1 if step is None else int(step)

        y = np.linspace(self.start, self.end, step)
        x = self.get_x(y.astype(int))

        return list(zip(x, y))

    def get_curvature(self, y=None) -> float:
        if y is None:
            y = (self.start - self.end) / 2

        return (1 + (2 * self.a * y + self.b) ** 2) ** 1.5 / np.abs(2 * self.a)

    def get_intersection(self, other, frame) -> Tuple[int, int]:
        a = self.a - other.a
        b = self.b - other.b
        c = self.c - other.c

        delta = b**2 - 4 * a * c

        if delta < 0:
            return None

        y1 = (-b + np.sqrt(delta)) / (2 * a)
        y2 = (-b - np.sqrt(delta)) / (2 * a)

        height = frame.shape[0]

        if 0 < y1 < height:
            y = y1
        elif 0 < y2 < height:
            y = y2
        else:
            return None

        return int(self.get_x(y)), int(y)

    def get_length(self) -> float:
        return self.start - self.end
