class LaneException(Exception):
    def __init__(self, message = ''):
        super().__init__(message)

        self.message = message

class LeftLineNotFound(LaneException):
    def __init__(self, message = ''):
        super().__init__(message)

        self.message = 'Left line not found' if message == '' else message

class RightLineNotFound(LaneException):
    def __init__(self, message = ''):
        super().__init__(message)

        self.message = 'Right line not found' if message == '' else message

class LaneNotFound(LaneException):
    def __init__(self, message = ''):
        super().__init__(message)

        self.message = 'Lane not found' if message == '' else message