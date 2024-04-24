# import rospy
# import cv2

# from rospy import Publisher
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge


class ImagePublisher:
    image_transform = None
    perspective_transform = None
    lane_fitting = None
    lane_detector = None
    lane_tracking = None
    # bridge = CvBridge()

    @staticmethod
    def __init__():
        # ImagePublisher.image_transform = rospy.Publisher(
        #     "/image_transform", Image, queue_size=10
        # )
        # ImagePublisher.perspective_transform = rospy.Publisher(
        #     "/perspective_transform", Image, queue_size=10
        # )
        # ImagePublisher.lane_fitting = rospy.Publisher(
        #     "/lane_fitting", Image, queue_size=10
        # )
        # ImagePublisher.lane_detector = rospy.Publisher(
        #     "/lane_detector", Image, queue_size=10
        # )
        # ImagePublisher.lane_tracking = rospy.Publisher(
        #     "/lane_tracking", Image, queue_size=10
        # )
        pass

    @staticmethod
    def publish_image_transform(image):
        ImagePublisher.publish_image(image, ImagePublisher.image_transform)

    @staticmethod
    def publish_perspective_transform(image):
        ImagePublisher.publish_image(image, ImagePublisher.perspective_transform)

    @staticmethod
    def publish_lane_fitting(image):
        ImagePublisher.publish_image(image, ImagePublisher.lane_fitting)

    @staticmethod
    def publish_lane_detector(image):
        ImagePublisher.publish_image(image, ImagePublisher.lane_detector)

    @staticmethod
    def publish_lane_tracking(image):
        ImagePublisher.publish_image(image, ImagePublisher.lane_tracking)

    @staticmethod
    def publish_image(image, publisher: Publisher):
        publisher.publish(ImagePublisher.bridge.cv2_to_imgmsg(image, "bgr8"))
