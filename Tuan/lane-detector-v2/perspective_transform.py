import cv2 as cv
import numpy as np

from lane_line import LaneLine

class PerspectiveTransform:
    src = None
    dst = None

    def __init__(self, config: dict) -> None:
        PerspectiveTransform.src = np.float32(config['src'])
        PerspectiveTransform.dst = np.float32(config['dst'])
        self.debug = config['debug']
        
        self.M = cv.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv.getPerspectiveTransform(self.dst, self.src)


    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])
        self.visualize(img)
        return cv.warpPerspective(img, self.M, img_size, flags=cv.INTER_LINEAR)

    def unwarp(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv.warpPerspective(img, self.Minv, img_size, flags=cv.INTER_LINEAR)
    
    def unwarp_line(self, line: LaneLine) -> LaneLine:
        start = PerspectiveTransform.get_top()
        end = PerspectiveTransform.get_bottom()

        line_points = line.get_points(start, end)

        unwarp_line_points = cv.perspectiveTransform(np.array([line_points], dtype=np.float32), self.Minv)

        unwarp_line = LaneLine(unwarp_line_points[0][:, 1], unwarp_line_points[0][:, 0])

        return unwarp_line
    
    def visualize(self, img) -> None:
        if (self.debug is False):
            return
        
        src = np.int32(self.src)
        visualize_img = img.copy()
        layer = np.zeros_like(visualize_img)

        cv.polylines(visualize_img, [src], True, (0, 0, 255), 2)
        cv.fillPoly(layer, [src], (0, 0, 255))
        cv.addWeighted(layer, 0.3, visualize_img, 1, 0, visualize_img)

        cv.imshow('perspective_transform', visualize_img)

    def get_top() -> int:
        return PerspectiveTransform.src[0][1]
    
    def get_bottom() -> int:
        return PerspectiveTransform.src[2][1]