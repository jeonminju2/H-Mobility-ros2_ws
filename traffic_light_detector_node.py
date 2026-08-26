import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import DetectionArray
from std_msgs.msg import String

from .lib import camera_perception_func_lib as CPFL

# ---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_IMAGE_TOPIC_NAME = "image_raw"

# Publish할 토픽 이름
# motion_planner_node.py 의 SUB_TRAFFIC_LIGHT_TOPIC_NAME 기본값과 반드시 일치해야 한다.
PUB_TOPIC_NAME = "yolov8_traffic_light_info"

# data.yaml 의 names 와 반드시 일치해야 한다. 틀리면 에러 없이 조용히 검출이 안 잡힌다.
TRAFFIC_LIGHT_CLASS = 'traffic_light'

# HSV 색 범위 - 실측(light_probe.py 등)으로 보정할 것.
# OpenCV HSV 범위: H 0~179, S/V 0~255. 빨강은 색상환 양 끝에 걸쳐 있어 두 구간으로 나눈다.
HSV_RANGES = {
    'red1': (np.array([0, 100, 95]), np.array([10, 255, 255])),
    'red2': (np.array([160, 100, 95]), np.array([179, 255, 255])),
    'yellow': (np.array([20, 100, 95]), np.array([30, 255, 255])),
    'green': (np.array([40, 100, 95]), np.array([90, 255, 255])),
}
# ----------------------------------------------

class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector_node')

        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_image_topic = self.declare_parameter('sub_image_topic', SUB_IMAGE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value

        self.cv_bridge = CvBridge()

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.detection_sub = Subscriber(self, DetectionArray, self.sub_detection_topic, qos_profile=self.qos_profile)
        self.image_sub = Subscriber(self, Image, self.sub_image_topic, qos_profile=self.qos_profile)
        self.ts = ApproximateTimeSynchronizer([self.detection_sub, self.image_sub], queue_size=1, slop=0.5)
        self.ts.registerCallback(self.sync_callback)

        self.publisher = self.create_publisher(String, self.pub_topic, self.qos_profile)

    def sync_callback(self, detection_msg: DetectionArray, image_msg: Image):
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg)

        # 프레임에 신호등 박스가 여러 개 잡힐 수 있으니(오검출 포함), 첫 번째로
        # 발견된 것을 그냥 쓰지 않고 신뢰도(score)가 가장 높은 것 하나만 고른다.
        best = None
        for detection in detection_msg.detections:
            if detection.class_name != TRAFFIC_LIGHT_CLASS:
                continue
            if best is None or detection.score > best.score:
                best = detection

        if best is not None:
            # get_traffic_light_color -> Red, Yellow, Green, Unknown
            traffic_light_color = CPFL.get_traffic_light_color(cv_image, best.bbox, HSV_RANGES)
        else:
            # 검출이 없을 때도 CPFL.get_traffic_light_color 와 동일한 값('Unknown')을 써서,
            # 이 토픽을 구독하는 쪽(motion_planner_node.py)이 'None'/'Unknown' 두 가지를
            # 따로 처리할 필요 없이 Red/Yellow/Green/Unknown 네 값만 다루면 되게 한다.
            traffic_light_color = 'Unknown'

        color_msg = String()
        color_msg.data = traffic_light_color
        self.get_logger().info(f'traffic light: {color_msg.data}')
        self.publisher.publish(color_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
