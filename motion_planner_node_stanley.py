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

TIMER = 0.1

BOX_CLASS = 'box'

MAX_SPEED = 255
APPROACH_SPEED = 120
MAX_STEERING = 7

# ---- 이 파일과 motion_planner_node.py(PID/pure-pursuit 버전)의 차이 ----
# 기존 버전은 target_slope(경로가 앞으로 휘어있는 방향)만 보고 조향을 계산했다.
# 이 버전(Stanley 방식)은 거기에 "지금 차량이 차선 중앙에서 얼마나 옆으로 벗어나 있는지"
# (횡방향 오차, CTE)를 추가로 계산해서 같이 반영한다. 곡선 진입은 헤딩(기울기)이 빠르게
# 반응하고, 직선 구간에서 옆으로 살짝 밀렸을 때는 CTE 항이 다시 중앙으로 끌어당긴다.
# CTE는 path_planner_node_stanley.py 가 보내주는 경로(차량 위치를 곡선에 포함시키지 않은 것)를
# 차량이 있는 행(y=179)에서 평가한 x값과, 실제 차량 x(320, CAR_CENTER_POINT) 의 차이로 계산한다.
CAR_CENTER_POINT = (320, 179)

# ---- 조향 PID (헤딩 + 횡방향 오차 결합) ----
STEER_KP = 0.35                # 헤딩(기울기) 비례항. 기존 STEERING_GAIN 과 동일한 의미/기본값.
STEER_KI = 0.0                 # 적분항. 필요할 때만 0.01~0.05부터 아주 조금씩.
STEER_KD = 0.12                # 미분항. 사행(좌우로 훅훅 도는 것)을 누른다. 0에서 시작해서 0.05씩.
STEER_INTEGRAL_LIMIT = 10.0    # 적분 누적 한계 (안티 와인드업)

# 횡방향 오차(픽셀 단위)를 target_slope 와 같은 스케일로 맞추기 위한 변환 계수.
# combined_error = target_slope + CTE_TO_SLOPE_GAIN * cte_px
# 이 값이 너무 크면 CTE가 헤딩보다 과하게 지배해서 직선에서도 미세하게 좌우로 떨릴 수 있고,
# 너무 작으면 "옆으로 밀린 채 똑바로 향하는" 상황에서 교정이 느려진다.
# 0.03~0.08 사이에서 실차 테스트로 잡는 것을 권장. (양수 = 오른쪽이 +x 라는 기존 이미지 좌표계 가정)
CTE_TO_SLOPE_GAIN = 0.05

SLOPE_FILTER_ALPHA = 0.4       # 합쳐진 오차(combined_error)에 적용하는 저역통과 필터 계수 (0~1)
STEERING_RATE_LIMIT = 2        # 한 틱마다 steering_command 가 바뀔 수 있는 최대 폭

# 헤딩(기울기) 계산에 쓸 lookahead. path_planner_node_stanley.py 는 차량 위치를 곡선에
# 포함시키지 않으므로, 여기서는 경로 리스트의 앞/뒤 두 점 사이 기울기를 그대로 쓴다.
LOOKAHEAD_POINTS = 40

STOP_METRIC = 'y_max'
SLOW_THRESHOLD = 220.0
STOP_THRESHOLD = 300.0

CONFIRM_FRAMES = 2
MEDIAN_WINDOW = 3

BOX_X_MIN, BOX_X_MAX = 0.0, 640.0

GREEN_CONFIRM_FRAMES = 0

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


class PID:
    """일반적인 PID 컨트롤러. 안티 와인드업(포화 시 적분 정지) 포함."""

    def __init__(self, kp: float, ki: float, kd: float, output_limit: float, integral_limit: float = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit if integral_limit is not None else output_limit * 4

        self._integral = 0.0
        self._prev_error = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0:
            dt = 1e-3

        p_term = self.kp * error

        d_term = 0.0
        if self._prev_error is not None:
            d_term = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        tentative_integral = self._integral + error * dt
        tentative_integral = max(-self.integral_limit, min(self.integral_limit, tentative_integral))
        i_term = self.ki * tentative_integral

        output = p_term + i_term + d_term
        clamped_output = max(-self.output_limit, min(self.output_limit, output))

        if clamped_output == output:
            self._integral = tentative_integral

        return clamped_output


class MotionPlanningNode(Node):
    """
    주행 상태 기계 (Stanley 방식 조향: 헤딩 오차 + 횡방향 오차)

        WAIT_GREEN  정지. 신호등 Green 을 연속으로 받으면 출발한다.
        DRIVE       최고속 차선 추종. 박스 지표가 SLOW_THRESHOLD 를 넘으면 SLOW.
        SLOW        감속 상태로 계속 차선 추종. STOP_THRESHOLD 를 넘으면 STOP.
        STOP        정지 명령.
    """

    def __init__(self):
        super().__init__('motion_planner_node')

        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter('sub_lane_topic', SUB_PATH_TOPIC_NAME).value
        self.sub_traffic_light_topic = self.declare_parameter('sub_traffic_light_topic', SUB_TRAFFIC_LIGHT_TOPIC_NAME).value
        self.sub_lidar_obstacle_topic = self.declare_parameter('sub_lidar_obstacle_topic', SUB_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value

        self.timer_period = self.declare_parameter('timer', TIMER).value
        self.car_center_point = self.declare_parameter('car_center_point', CAR_CENTER_POINT).value

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.path_data = None
        self.lidar_data = None

        self.state = 'WAIT_GREEN'
        self.green_streak = 0
        self.traffic_light_seen = False
        self.wait_ticks = 0
        self.slow_streak = 0
        self.stop_streak = 0

        self.metric_buf = deque(maxlen=MEDIAN_WINDOW)
        self.box_value = None

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0
        self.slope_value = None        # 원본 헤딩 기울기(도)
        self.cte_value = None          # 원본 횡방향 오차(px)
        self.filtered_error = None     # 필터를 거친 합산 오차 (PID 입력)
        self.prev_steering_command = 0

        self.steering_pid = PID(
            STEER_KP, STEER_KI, STEER_KD,
            output_limit=MAX_STEERING,
            integral_limit=STEER_INTEGRAL_LIMIT,
        )

        self.detection_sub = self.create_subscription(DetectionArray, self.sub_detection_topic, self.detection_callback, self.qos_profile)
        self.path_sub = self.create_subscription(PathPlanningResult, self.sub_path_topic, self.path_callback, self.qos_profile)
        self.traffic_light_sub = self.create_subscription(String, self.sub_traffic_light_topic, self.traffic_light_callback, self.qos_profile)
        self.lidar_sub = self.create_subscription(Bool, self.sub_lidar_obstacle_topic, self.lidar_callback, self.qos_profile)

        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)

        self.log_file = None
        self.log_writer = None
        self.log_start = None
        if LOG_ENABLED:
            self.open_log()

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info(
            f"상태 기계 시작 (Stanley). 지표={STOP_METRIC} 감속={SLOW_THRESHOLD} 정지={STOP_THRESHOLD} "
            f"조향PID=(P={STEER_KP}, I={STEER_KI}, D={STEER_KD}) CTE_GAIN={CTE_TO_SLOPE_GAIN}")

    # ---------------- 주행 로그 ----------------

    def open_log(self):
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            path = os.path.join(LOG_DIR, time.strftime('drive_stanley_%Y%m%d_%H%M%S.csv'))
            self.log_file = open(path, 'w', newline='')
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow(
                ['t_sec', 'wall_time', 'state', 'steering',
                 'left_speed', 'right_speed', 'slope_deg', 'cte_px', 'filtered_error', STOP_METRIC])
            self.log_start = time.time()
            self.get_logger().info(f"주행 로그: {path}")
        except OSError as e:
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
            '' if self.cte_value is None else f"{self.cte_value:.1f}",
            '' if self.filtered_error is None else f"{self.filtered_error:.3f}",
            '' if self.box_value is None else f"{self.box_value:.1f}",
        ])
        self.log_file.flush()

    def close_log(self):
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
            self.log_writer = None

    # ---------------- 콜백 ----------------

    def detection_callback(self, msg: DetectionArray):
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

        self.wait_ticks += 1
        if not self.traffic_light_seen and self.wait_ticks % 50 == 0:
            self.get_logger().warn(
                f"'{self.sub_traffic_light_topic}' 토픽을 한 번도 받지 못했다. "
                f"traffic_light_detector_node 가 떠 있는지 확인할 것.")

        if self.green_streak >= GREEN_CONFIRM_FRAMES:
            self.state = 'DRIVE'
            self.get_logger().info(f"Green {self.green_streak}프레임 확인 -> 출발")

    def run_driving(self, speed: int):
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
        """헤딩(경로 기울기) 오차 + 횡방향(중앙 대비 옆으로 밀린 정도) 오차를 합쳐서
        PID + 저역통과 필터 + 변화율 제한으로 조향값을 만든다 (Stanley 방식)."""
        if self.path_data is None or len(self.path_data) < LOOKAHEAD_POINTS:
            self.slope_value = None
            self.cte_value = None
            self.filtered_error = None
            self.steering_pid.reset()
            return 0

        # ---- 헤딩 오차 ----
        target_slope = DMFL.calculate_slope_between_points(
            self.path_data[-LOOKAHEAD_POINTS], self.path_data[-1])

        if not isinstance(target_slope, (int, float)):
            self.slope_value = None
            return self.prev_steering_command

        self.slope_value = target_slope

        # ---- 횡방향 오차(CTE) ----
        # path_planner_node_stanley.py 가 차량 위치(car_center_point[1] 행)까지 경로를 채워서
        # 보내주므로, 경로의 마지막 점이 그 행(또는 그와 가장 가까운 행)의 예측 중앙 x 값이다.
        path_x_at_car_row, path_y_at_car_row = self.path_data[-1]
        cte_px = path_x_at_car_row - self.car_center_point[0]
        self.cte_value = cte_px

        # ---- 합산 ----
        combined_error = target_slope + CTE_TO_SLOPE_GAIN * cte_px

        # 1) 저역통과 필터
        if self.filtered_error is None:
            self.filtered_error = combined_error
        else:
            self.filtered_error = (
                SLOPE_FILTER_ALPHA * combined_error
                + (1 - SLOPE_FILTER_ALPHA) * self.filtered_error
            )

        # 2) PID
        raw_steering = self.steering_pid.compute(self.filtered_error, self.timer_period)

        # 3) 변화율 제한
        delta = raw_steering - self.prev_steering_command
        delta = max(-STEERING_RATE_LIMIT, min(STEERING_RATE_LIMIT, delta))
        limited_steering = self.prev_steering_command + delta

        steering = int(round(max(-MAX_STEERING, min(MAX_STEERING, limited_steering))))
        self.prev_steering_command = steering
        return steering


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
