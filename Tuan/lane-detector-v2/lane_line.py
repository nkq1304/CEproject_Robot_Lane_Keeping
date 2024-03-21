import numpy as np


class LaneLine:
    def __init__(self, lefty, leftx) -> None:
        poly_fit = np.polyfit(lefty, leftx, 2)

        self.a = poly_fit[0]
        self.b = poly_fit[1]
        self.c = poly_fit[2]

    def get_x(self, y: int) -> int:
        return self.a * y**2 + self.b * y + self.c

    def get_points(self, start, end, step=None) -> list[tuple[int, int]]:
        start = int(start)
        end = int(end)
        step = abs(end - start) + 1 if step is None else int(step)

        y = np.linspace(start, end, step)
        x = self.get_x(y.astype(int))

        return list(zip(x, y))

    def get_curvature(self, y) -> float:
        return ((1 + (2 * self.a * y + self.b) ** 2) ** 1.5) / np.absolute(2 * self.a)

    def get_intersection(self, other, frame) -> tuple[int, int]:
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
