import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from utils.config import Config

from modules.backend.backend import Backend
from modules.backend.image_publisher import ImagePublisher


def image_callback(data):
    frame = bridge.imgmsg_to_cv2(data, "bgr8")

    backend.process_frame(frame)


def on_shutdown():
    print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = Config("configs/turtlebot.yaml")
    backend = Backend(cfg)

    bridge = CvBridge()

    rospy.init_node("get_image", anonymous=True)
    rospy.Subscriber("/camera/image", Image, image_callback)

    publiser = rospy.Publisher("/lane_frame", Image, queue_size=10)

    ImagePublisher()

    rospy.on_shutdown(on_shutdown)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
