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
