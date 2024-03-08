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
