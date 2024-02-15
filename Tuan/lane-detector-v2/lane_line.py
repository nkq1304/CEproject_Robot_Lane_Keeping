import numpy as np

class LaneLine:
    def __init__(self, poly_fit: np.ndarray) -> None:
        self.a = poly_fit[0]
        self.b = poly_fit[1]
        self.c = poly_fit[2]

    def get_x(self, y: int) -> int:
        return self.a * y ** 2 + self.b * y + self.c

class Lane:
    def __init__(self, left_line: LaneLine, right_line: LaneLine) -> None:
        self.left_line = left_line
        self.right_line = right_line