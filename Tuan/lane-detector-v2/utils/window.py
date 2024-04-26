class Window:
    def __init__(self, width, height, point=(0, 0)) -> None:
        self.width = width
        self.height = height
        self.x = int(point[0])
        self.y = int(point[1])
        self.top = self.y - self.height // 2
        self.bottom = self.y + self.height // 2
        self.left = self.x - self.width // 2
        self.right = self.x + self.width // 2

    def set_x(self, x: int) -> None:
        self.x = int(x)
        self.left = self.x - self.width // 2
        self.right = self.x + self.width // 2

    def set_y(self, y: int) -> None:
        self.y = int(y)
        self.top = self.y - self.height // 2
        self.bottom = self.y + self.height // 2
