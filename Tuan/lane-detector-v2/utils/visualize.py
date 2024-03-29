import cv2

from utils.lane_line import LaneLine


def draw_lane(frame, lane: LaneLine, start, end, color=(0, 255, 0)):
    if lane is None:
        return frame

    for point in lane.get_points(start, end):
        cv2.circle(frame, (int(point[0]), int(point[1])), 1, color, 2)
    return frame


def draw_lane(frame, lane: LaneLine, color=(0, 255, 0)):
    if lane is None:
        return frame

    for point in lane.get_points():
        cv2.circle(frame, (int(point[0]), int(point[1])), 1, color, 2)
    return frame


def draw_intersection(frame, intersection):
    if intersection is None:
        return frame

    cv2.circle(frame, (intersection[0], intersection[1]), 5, (0, 0, 255), -1)
    return frame


def draw_window(frame, window, color=(0, 255, 0)):
    if window is None:
        return frame

    cv2.rectangle(
        frame,
        (window.left, window.top),
        (window.right, window.bottom),
        color,
        2,
    )

    return frame
