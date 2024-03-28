import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from lane_detector import LaneDetector


def image_callback(data):
    frame = bridge.imgmsg_to_cv2(data, "bgr8")
    frame = cv2.resize(frame, (640, 360))

    lane_frame = lane_detector.detect(frame)

    cv2.imshow("lane_frame", lane_frame)
    cv2.waitKey(1)


def on_shutdown():
    print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = Config("configs/example.yaml")

    lane_detector = LaneDetector(cfg.lane_detector)

    bridge = CvBridge()

    rospy.init_node("get_image", anonymous=True)
    rospy.Subscriber("/camera/image", Image, image_callback)
    rospy.on_shutdown(on_shutdown)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
