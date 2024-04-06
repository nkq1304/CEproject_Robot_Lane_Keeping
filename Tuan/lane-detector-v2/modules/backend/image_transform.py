import cv2 as cv

from modules.backend.image_publisher import ImagePublisher

class ImageTransform:
    def __init__(self, config: dict) -> None:
        self.vertical_flip = config["vertical_flip"]
        self.horizontal_flip = config["horizontal_flip"]
        self.debug = config["debug"]

    def transform(self, img):
        if self.vertical_flip:
            img = cv.flip(img, 0)

        if self.horizontal_flip:
            img = cv.flip(img, 1)
        
        self.visualize(img)

        return img

    def visualize(self, img):
        if not self.debug:
            return
        
        viz_img = img.copy()

        if ImagePublisher.image_transform is not None:
            ImagePublisher.publish_image_transform(img)
            return
        else:
            cv.imshow("image_transform", img)