import cv2


def draw_lane(frame, lane, start, end, color=(0, 255, 0)):
    if lane is None:
        return frame

    for point in lane.get_points(start, end):
        cv2.circle(frame, (int(point[0]), int(point[1])), 1, color, 2)
    return frame
