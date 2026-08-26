import csv
import os
import statistics
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String, Bool
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand
from .lib import decision_making_func_lib as DMFL

#---------------Variable Setting---------------
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_TRAFFIC_LIGHT_TOPIC_NAME = "yolov8_traffic_light_info"
SUB_LIDAR_OBSTACLE_TOPIC_NAME = "lidar_obstacle_info"
PUB_TOPIC_NAME = "topic_control_signal"

# 모션 플랜 발행 주기 (초) - 소수점 필요 (int형은 반영되지 않음)
TIMER = 0.1

# ---- 클래스 이름 ----
# data.yaml 의 names 와 반드시 일치해야 한다. 틀리면 에러 없이 조용히 검출이 안 잡힌다.
BOX_CLASS = 'box'

# ---- 속도 / 조향 ----
MAX_SPEED = 255           # 최대 255
APPROACH_SPEED = 120      # 감속 구간 속도. 관성이 목표 창보다 작으면 MAX_SPEED 로 올려도 된다.
MAX_STEERING = 7          # 최대 7
STEERING_GAIN = 0.35      # 경로 기울기(도) -> 조향값. 사행하면 낮추고, 코너를 못 돌면 높인다.

# 경로 100점 중 뒤에서 몇 번째 점을 '바라볼 지점'으로 삼을지.
# 경로는 y=5(먼 곳) ~ y=179(차 바로 앞)를 100등분하므로 간격은 약 1.76px.
#   10 -> 약 16px 앞  (차 바로 앞만 보고 뒤늦게 되돌리는 동작이 된다. 기존값)
#   40 -> 약 69px 앞  (미리 보고 꺾는다)
# 크게 하면 코너 진입이 빨라지지만 직선에서 과하게 반응할 수 있다.
LOOKAHEAD_POINTS = 40

# ---- 정지선 임계값 : 실측으로 채울 것 ----
# analyze_probe.py 가 1등으로 뽑은 지표를 STOP_METRIC 에 쓴다.
#   'y_max' | 'y_min' | 'height' | 'width' | 'area'
STOP_METRIC = 'y_max'
SLOW_THRESHOLD = 220.0    # 이 값을 넘으면 APPROACH_SPEED 로 감속
STOP_THRESHOLD = 300.0    # 이 값을 넘으면 정지. = 목표 창 앞 경계 - 관성보정
                          #   (stop_calibrate.py 의 '관성 + 지연 보정' 출력값을 뺀다)

CONFIRM_FRAMES = 2        # 임계값을 연속 N프레임 넘어야 인정. 단발 오검출 방지.
MEDIAN_WINDOW = 3         # 지표 중앙값 필터 창. 떨림이 크면 늘리되 지연도 같이 늘어난다.

# 박스가 이 x 범위 안에 있을 때만 인정 (화면 폭 640 기준).
# 트랙 건너편의 박스를 보고 미리 서는 것을 막는다. 640 전체로 두면 사실상 해제.
BOX_X_MIN, BOX_X_MAX = 0.0, 640.0

# ---- 신호등 ----
GREEN_CONFIRM_FRAMES = 0  # Green 을 연속 N번 받아야 출발. 오검출로 튀어나가는 것 방지.

# ---- 주행 로그 ----
# True 면 타이머 주기(TIMER)마다 시각/상태/조향/속도/지표를 CSV 한 줄로 남긴다.
# 파일은 실행할 때마다 새로 만들어진다 (drive_20260813_174500.csv) — 이전 주행을 덮어쓰지 않는다.
LOG_ENABLED = True
LOG_DIR = os.path.expanduser('~/ros2_ws/drive_logs')
#----------------------------------------------


def box_metric(detection, name: str) -> float:
    cx = detection.bbox.center.position.x
    cy = detection.bbox.center.position.y
    bw = detection.bbox.size.x
    bh = detection.bbox.size.y
    return {
        'y_max': cy + bh / 2,
        'y_min': cy - bh / 2,
        'height': bh,
        'width': bw,
        'area': bw * bh,
    }[name]


class MotionPlanningNode(Node):
    """
    주행 상태 기계

        WAIT_GREEN  정지. 신호등 Green 을 연속으로 받으면 출발한다.
                    한 번 출발하면 신호등은 두 번 다시 보지 않는다(래치).
        DRIVE       최고속 차선 추종. 박스 지표가 SLOW_THRESHOLD 를 넘으면 SLOW.
        SLOW        감속 상태로 계속 차선 추종. STOP_THRESHOLD 를 넘으면 STOP.
        STOP        정지 명령. 규정상 여기서 주행 종료.
    """

    def __init__(self):
        super().__init__('motion_planner_node')

        # 토픽 이름 설정
        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter('sub_lane_topic', SUB_PATH_TOPIC_NAME).value
        self.sub_traffic_light_topic = self.declare_parameter('sub_traffic_light_topic', SUB_TRAFFIC_LIGHT_TOPIC_NAME).value
        self.sub_lidar_obstacle_topic = self.declare_parameter('sub_lidar_obstacle_topic', SUB_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value

        self.timer_period = self.declare_parameter('timer', TIMER).value

        # QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 변수 초기화
        self.path_data = None
        self.lidar_data = None

        self.state = 'WAIT_GREEN'
        self.green_streak = 0
        self.traffic_light_seen = False
        self.wait_ticks = 0
        self.slow_streak = 0
        self.stop_streak = 0

        self.metric_buf = deque(maxlen=MEDIAN_WINDOW)
        self.box_value = None      # 중앙값 필터를 거친 박스 지표. 미검출이면 None

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0
        self.slope_value = None    # 조향을 만든 경로 기울기(도). 경로가 없으면 None

        # 서브스크라이버 설정
        self.detection_sub = self.create_subscription(DetectionArray, self.sub_detection_topic, self.detection_callback, self.qos_profile)
        self.path_sub = self.create_subscription(PathPlanningResult, self.sub_path_topic, self.path_callback, self.qos_profile)
        self.traffic_light_sub = self.create_subscription(String, self.sub_traffic_light_topic, self.traffic_light_callback, self.qos_profile)
        self.lidar_sub = self.create_subscription(Bool, self.sub_lidar_obstacle_topic, self.lidar_callback, self.qos_profile)

        # 퍼블리셔 설정
        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)

        # 주행 로그 파일 열기
        self.log_file = None
        self.log_writer = None
        self.log_start = None
        if LOG_ENABLED:
            self.open_log()

        # 타이머 설정
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info(
            f"상태 기계 시작. 지표={STOP_METRIC} 감속={SLOW_THRESHOLD} 정지={STOP_THRESHOLD}")

    # ---------------- 주행 로그 ----------------

    def open_log(self):
        """실행 시각을 파일명에 넣어 새 CSV 를 연다."""
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            path = os.path.join(LOG_DIR, time.strftime('drive_%Y%m%d_%H%M%S.csv'))
            self.log_file = open(path, 'w', newline='')
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow(
                ['t_sec', 'wall_time', 'state', 'steering',
                 'left_speed', 'right_speed', 'slope_deg', STOP_METRIC])
            self.log_start = time.time()
            self.get_logger().info(f"주행 로그: {path}")
        except OSError as e:
            # 로그를 못 남기는 것 때문에 주행이 멈추면 안 된다.
            self.log_writer = None
            self.get_logger().error(f"주행 로그 파일을 열지 못했다: {e}")

    def write_log(self):
        if self.log_writer is None:
            return
        now = time.time()
        self.log_writer.writerow([
            f"{now - self.log_start:.3f}",
            time.strftime('%H:%M:%S', time.localtime(now)) + f".{int(now % 1 * 1000):03d}",
            self.state,
            self.steering_command,
            self.left_speed_command,
            self.right_speed_command,
            '' if self.slope_value is None else f"{self.slope_value:.2f}",
            '' if self.box_value is None else f"{self.box_value:.1f}",
        ])
        # 주행 중 Ctrl+C 로 끊어도 마지막 줄까지 남도록 매 줄 flush 한다.
        self.log_file.flush()

    def close_log(self):
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
            self.log_writer = None

    # ---------------- 콜백 ----------------

    def detection_callback(self, msg: DetectionArray):
        """박스 지표를 뽑아 중앙값 필터에 넣는다."""
        best = None
        for detection in msg.detections:
            if detection.class_name != BOX_CLASS:
                continue
            cx = detection.bbox.center.position.x
            if not (BOX_X_MIN <= cx <= BOX_X_MAX):
                continue
            if best is None or detection.score > best.score:
                best = detection

        if best is None:
            self.metric_buf.clear()
            self.box_value = None
            return

        self.metric_buf.append(box_metric(best, STOP_METRIC))
        self.box_value = statistics.median(self.metric_buf)

    def path_callback(self, msg: PathPlanningResult):
        self.path_data = list(zip(msg.x_points, msg.y_points))

    def traffic_light_callback(self, msg: String):
        # 출발한 뒤로는 신호등을 보지 않는다. 규정상 다시 바뀌지 않으므로
        # 2바퀴째 출발선을 지날 때 오검출로 멈추는 사고를 막는다.
        if self.state != 'WAIT_GREEN':
            return
        self.traffic_light_seen = True
        self.green_streak = self.green_streak + 1 if msg.data == 'Green' else 0

    def lidar_callback(self, msg: Bool):
        self.lidar_data = msg

    # ---------------- 상태 기계 ----------------

    def timer_callback(self):
        if self.state == 'WAIT_GREEN':
            self.run_wait_green()
        elif self.state == 'DRIVE':
            self.run_driving(MAX_SPEED)
        elif self.state == 'SLOW':
            self.run_driving(APPROACH_SPEED)
        else:  # STOP
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0

        self.get_logger().info(
            f"[{self.state}] steering: {self.steering_command}, "
            f"left_speed: {self.left_speed_command}, "
            f"right_speed: {self.right_speed_command}, "
            f"{STOP_METRIC}: {self.box_value if self.box_value is None else round(self.box_value, 1)}")

        motion_command_msg = MotionCommand()
        motion_command_msg.steering = self.steering_command
        motion_command_msg.left_speed = self.left_speed_command
        motion_command_msg.right_speed = self.right_speed_command
        self.publisher.publish(motion_command_msg)

        self.write_log()

    def run_wait_green(self):
        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0

        # 신호등 노드가 안 떠 있으면 영원히 출발하지 않는다. 원인을 알 수 있게 경고를 남긴다.
        self.wait_ticks += 1
        if not self.traffic_light_seen and self.wait_ticks % 50 == 0:
            self.get_logger().warn(
                f"'{self.sub_traffic_light_topic}' 토픽을 한 번도 받지 못했다. "
                f"traffic_light_detector_node 가 떠 있는지 확인할 것.")

        if self.green_streak >= GREEN_CONFIRM_FRAMES:
            self.state = 'DRIVE'
            self.get_logger().info(f"Green {self.green_streak}프레임 확인 -> 출발")

    def run_driving(self, speed: int):
        # 라이다를 쓰는 경우에만 동작한다. 노드를 안 띄우면 lidar_data 는 계속 None.
        if self.lidar_data is not None and self.lidar_data.data is True:
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0
            return

        self.steering_command = self.compute_steering()
        self.left_speed_command = speed
        self.right_speed_command = speed

        self.update_box_state()

    def update_box_state(self):
        """박스 지표로 SLOW / STOP 전이를 판단한다."""
        if self.box_value is None:
            self.slow_streak = 0
            self.stop_streak = 0
            return

        self.stop_streak = self.stop_streak + 1 if self.box_value >= STOP_THRESHOLD else 0
        self.slow_streak = self.slow_streak + 1 if self.box_value >= SLOW_THRESHOLD else 0

        if self.stop_streak >= CONFIRM_FRAMES:
            self.state = 'STOP'
            self.get_logger().info(
                f"정지. {STOP_METRIC}={self.box_value:.1f} >= {STOP_THRESHOLD}")
        elif self.state == 'DRIVE' and self.slow_streak >= CONFIRM_FRAMES:
            self.state = 'SLOW'
            self.get_logger().info(
                f"감속. {STOP_METRIC}={self.box_value:.1f} >= {SLOW_THRESHOLD}")

    def compute_steering(self) -> int:
        """경로 기울기에 비례한 조향. 기존 뱅뱅 제어(±7 아니면 0)는 사행이 심했다."""
        if self.path_data is None or len(self.path_data) < LOOKAHEAD_POINTS:
            self.slope_value = None
            return 0

        target_slope = DMFL.calculate_slope_between_points(
            self.path_data[-LOOKAHEAD_POINTS], self.path_data[-1])

        # 두 점의 y가 같으면 문자열 'inf' 를 반환한다. 비교하면 TypeError 로 콜백이 죽는다.
        if not isinstance(target_slope, (int, float)):
            self.slope_value = None
            return 0

        self.slope_value = target_slope

        steering = int(round(STEERING_GAIN * target_slope))
        return max(-MAX_STEERING, min(MAX_STEERING, steering))


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.close_log()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
