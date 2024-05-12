import rospy
import cv2

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from utils.config import Config

from modules.backend.backend import Backend
from modules.backend.image_publisher import ImagePublisher
from modules.controller.turtlebot_controller import TurtlebotController


def image_callback(data):
    frame = bridge.compressed_imgmsg_to_cv2(data, "bgr8")

    center_lane = backend.update(frame)
    turtlebot_controller.follow_lane(center_lane)


def on_shutdown():
    print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = Config("configs/turtlebot.yaml")

    backend = Backend(cfg)
    turtlebot_controller = TurtlebotController(cfg.turtlebot_controller)

    bridge = CvBridge()

    rospy.init_node("get_image", anonymous=True)
    rospy.Subscriber("/camera/image/compressed", CompressedImage, image_callback)

    ImagePublisher()

    rospy.on_shutdown(on_shutdown)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
