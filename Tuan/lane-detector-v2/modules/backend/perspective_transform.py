import cv2 as cv
import numpy as np

from modules.backend.image_publisher import ImagePublisher

class PerspectiveTransform:
    src = None
    dst = None

    Minv = None

    def __init__(self, config: dict) -> None:
        PerspectiveTransform.src = np.float32(config["src"])
        PerspectiveTransform.dst = np.float32(config["dst"])
        self.debug = config["debug"]

        self.M = cv.getPerspectiveTransform(self.src, self.dst)
        PerspectiveTransform.Minv = cv.getPerspectiveTransform(self.dst, self.src)

    def get_sky_view(self, img, debug=True):
        img_size = (img.shape[1], img.shape[0])

        if debug:
            self.visualize(img)
            
        return cv.warpPerspective(img, self.M, img_size, flags=cv.INTER_LINEAR)

    @staticmethod
    def get_car_view(img):
        img_size = (img.shape[1], img.shape[0])
        return cv.warpPerspective(
            img, PerspectiveTransform.Minv, img_size, flags=cv.INTER_LINEAR
        )

    def visualize(self, img) -> None:
        if not self.debug:
            return

        src = np.int32(self.src)
        visualize_img = img.copy()
        layer = np.zeros_like(visualize_img)

        cv.polylines(visualize_img, [src], True, (0, 0, 255), 2)
        cv.fillPoly(layer, [src], (0, 0, 255))
        cv.addWeighted(layer, 0.3, visualize_img, 1, 0, visualize_img)

        if ImagePublisher.perspective_transform is not None:
            ImagePublisher.publish_perspective_transform(visualize_img)
        else:
            cv.imshow("perspective_transform", visualize_img)

    def get_top() -> int:
        return PerspectiveTransform.src[0][1]

    def get_bottom() -> int:
        return PerspectiveTransform.src[2][1]
