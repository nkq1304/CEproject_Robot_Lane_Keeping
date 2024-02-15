import cv2 as cv
import numpy as np

from lane_line import LaneLine

class PerspectiveTransform:
    def __init__(self, config: dict) -> None:
        self.src = np.float32(config['src'])
        self.dst = np.float32(config['dst'])
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
    
    def visualize(self, img) -> None:
        if (self.debug is False):
            return
        
        src = np.int32(self.src)
        visualize_img = img.copy()
        cv.polylines(visualize_img, [src], True, (0, 0, 255), 2)
        cv.imshow('perspective_transform', visualize_img)