import cv2

def show_image(name, image, scale = 0.8):
    """
    Show image with scale.
    Parameters:
        name: The name of the window.
        image: The input image.
        scale: The scale of the image.
    """
    show = cv2.resize(image, (0,0), fx=scale, fy=scale)
    cv2.imshow(name, show)

    cv2.waitKey(0)