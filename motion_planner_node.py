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

# ---- 조향 PID ----
# 기존에는 목표 기울기(target_slope)에 상수(STEERING_GAIN)만 곱하는 순수 비례(P) 제어였다.
# 그래서 차선 인식이 프레임마다 살짝 흔들리면 조향값도 그대로 따라 흔들리고(사행),
# 값이 크게 튈 때 그대로 반영되면서 "훅훅" 꺾이며 가운데선<->바깥선 사이를 오가는
# 현상이 나타났다. 아래 세 가지를 추가해서 이 흔들림을 억제한다.
#   1) 저역통과 필터: target_slope 자체를 부드럽게 만들어 프레임 간 튀는 값을 완화
#   2) PID: 비례(P) 외에 미분(D)으로 사행을 누르고, 필요하면 적분(I)로 한쪽 쏠림을 보정
#   3) 변화율 제한: 조향값이 한 틱에 너무 크게 바뀌지 않도록 제한 (물리적 "훅" 방지)
STEER_KP = 0.35                # 비례항. 기존 STEERING_GAIN 과 동일한 의미/기본값.
STEER_KI = 0.0                 # 적분항. 한쪽 선으로 계속 치우쳐 달리는 정상상태 오차가 보이면
                                # 0.01~0.05부터 아주 조금씩만 올려본다. 너무 올리면 천천히 진동한다.
STEER_KD = 0.12                # 미분항. 사행(좌우로 훅훅 도는 것)을 누르는 역할.
                                # 0에서 시작해서 0.05씩 올려가며 가장 안 흔들리는 지점을 찾는다.
                                # 너무 크게 잡으면 반대로 고주파 떨림이 생긴다.
STEER_INTEGRAL_LIMIT = 10.0    # 적분 누적 한계 (안티 와인드업). 출력이 포화되면 적분을 멈춘다.

SLOPE_FILTER_ALPHA = 0.4       # target_slope 저역통과 필터 계수 (0~1).
                                # 작을수록 부드럽지만 반응이 느려지고, 1이면 필터를 안 쓰는 것과 같다.
STEERING_RATE_LIMIT = 2        # 한 틱(TIMER)마다 steering_command 가 바뀔 수 있는 최대 폭(-7~7 기준).
                                # 값을 줄이면 더 부드럽지만 급커브 대응이 늦어진다.

# ---- 조향 하드웨어 캘리브레이션 (참고용 — 이 노드는 여전히 -7~+7 을 발행한다) ----
# 실측: 중립 455 / 좌 최대 534 / 우 최대 364
# steering_command(-7~+7) 를 실제 서보 PWM 으로 바꾸는 코드는 이 두 파일 밖(모터/서보 제어 노드)에 있다.
# 중립 기준으로 좌측 폭은 79(534-455), 우측 폭은 91(455-364)로 서로 다르다(비대칭).
# 그 변환 노드가 좌우를 대칭으로 매핑하고 있다면, 그게 "한쪽으로 더 세게 꺾이는" 원인 중 하나일 수
# 있으니 이 부분도 같이 확인해볼 것을 권장한다. (이 파일에서는 값을 직접 다루지 않으므로 참고만.)
STEER_CENTER_PWM = 455
STEER_LEFT_PWM = 534
STEER_RIGHT_PWM = 364

# 경로 100점 중 뒤에서 몇 번째 점을 '바라볼 지점'으로 삼을지.
# 경로는 y=5(먼 곳) ~ y=179(차 바로 앞)를 100등분하므로 간격은 약 1.76px.
#   10 -> 약 16px 앞  (차 바로 앞만 보고 뒤늦게 되돌리는 동작이 된다)
#   15 -> 약 26px 앞  (바로 앞 위주. 직선에서 안정적이고, 커브도 너무 늦지 않게 잡는다. 현재값)
#   40 -> 약 69px 앞  (멀리 미리 보고 꺾지만, 먼 쪽 검출이 흔들리면 직선에서도 조향이 반응한다)
# 크게 하면 코너 진입이 빨라지지만 직선에서 과하게 반응할 수 있다. 작게 하면 반대로
# 코너 대응이 늦어지니, 아래 STEER_DEADBAND_DEG 와 같이 맞춰서 조절한다.
LOOKAHEAD_POINTS = 15

# 필터를 거친 기울기(filtered_slope)가 이 값(도) 이내면 노이즈로 보고 완전히 0(직진)으로
# 취급한다. 하드 클리핑이 아니라 "밴드만큼 빼는" 방식이라 경계에서 값이 뚝 끊기지 않는다.
#   0.0 -> 데드밴드 없음 (기존 동작과 동일)
#   1.5 -> 직선에서 차선 검출이 ±1.5도 정도 흔들려도 조향이 전혀 반응하지 않는다.
# LOOKAHEAD_POINTS 를 줄인 만큼 노이즈에 더 민감해지므로, 이 값으로 다시 죽여준다.
# 너무 크게 잡으면 완만한 커브 초입을 직선으로 오인해서 진입이 늦어질 수 있다.
STEER_DEADBAND_DEG = 1.5

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
# 주의: 0으로 두면 green_streak(0에서 시작) >= 0 이 첫 틱부터 참이 되어
# Green 신호를 한 번도 못 받아도 즉시 출발해버린다(대기 로직이 사실상 꺾여있던 버그).
# 반드시 1 이상으로 설정할 것. 오검출 방지 여유를 두려면 3~5 권장.
GREEN_CONFIRM_FRAMES = 3  # Green 을 연속 N번 받아야 출발. 오검출로 튀어나가는 것 방지.

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


class PID:
    """일반적인 PID 컨트롤러.

    - 미분항은 필터를 거친(부드러운) 오차를 기준으로 계산해서 노이즈 증폭을 줄인다.
    - 출력이 상/하한에 걸려 포화된 순간에는 적분을 멈춘다 (안티 와인드업).
      이게 없으면 코너를 오래 도는 동안 적분값이 계속 쌓였다가, 직선 구간에 들어선 뒤에도
      한동안 과도하게 꺾인 채로 남아서 오히려 흔들림을 키운다.
    """

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

        # 적분을 임시로 누적해서, 그걸 포함한 출력이 포화되는지 먼저 확인한다.
        tentative_integral = self._integral + error * dt
        tentative_integral = max(-self.integral_limit, min(self.integral_limit, tentative_integral))
        i_term = self.ki * tentative_integral

        output = p_term + i_term + d_term
        clamped_output = max(-self.output_limit, min(self.output_limit, output))

        # 포화되지 않았을 때만 적분을 실제로 반영 (안티 와인드업)
        if clamped_output == output:
            self._integral = tentative_integral

        return clamped_output


def apply_deadband(value: float, band: float) -> float:
    """band 이내의 값은 0으로, 벗어난 값은 band만큼 뺀 채로 부드럽게 통과시킨다.

    예: band=1.5 일 때 value=1.0 -> 0.0, value=3.0 -> 1.5.
    그냥 0으로 클리핑(if abs(value) < band: return 0)하지 않는 이유는, 그렇게 하면
    경계를 넘나들 때 값이 뚝 끊겨서 PID의 미분(D)항이 순간적으로 튀기 때문이다.
    """
    if band <= 0:
        return value
    if value > band:
        return value - band
    if value < -band:
        return value + band
    return 0.0


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
        self.sub_path_topic = self.declare_parameter('sub_path_topic', SUB_PATH_TOPIC_NAME).value
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
        self.slope_value = None        # 조향을 만든 원본 경로 기울기(도). 경로가 없으면 None
        self.filtered_slope = None     # 저역통과 필터를 거친 기울기 (PID 입력)
        self.prev_steering_command = 0 # 변화율 제한을 위한 직전 조향값

        self.steering_pid = PID(
            STEER_KP, STEER_KI, STEER_KD,
            output_limit=MAX_STEERING,
            integral_limit=STEER_INTEGRAL_LIMIT,
        )

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
            f"상태 기계 시작. 지표={STOP_METRIC} 감속={SLOW_THRESHOLD} 정지={STOP_THRESHOLD} "
            f"조향PID=(P={STEER_KP}, I={STEER_KI}, D={STEER_KD})")

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
                 'left_speed', 'right_speed', 'raw_slope_deg', 'filtered_slope_deg', STOP_METRIC])
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
            '' if self.filtered_slope is None else f"{self.filtered_slope:.2f}",
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
        """경로 기울기를 PID로 제어해 조향값을 만든다.

        기존에는 target_slope * STEERING_GAIN 만 계산하는 순수 비례 제어였고,
        기울기가 튀면 조향도 그대로 튀어서 "훅훅" 꺾이는 사행이 심했다.
        지금은 (1) 기울기를 저역통과 필터로 부드럽게 만들고, (2) PID로 변환하고,
        (3) 조향값 변화율 자체를 제한해서 3중으로 흔들림을 억제한다.
        """
        if self.path_data is None or len(self.path_data) < LOOKAHEAD_POINTS:
            self.slope_value = None
            self.filtered_slope = None
            self.steering_pid.reset()
            # 경로가 잠깐(한 프레임) 비어도 0으로 확 꺾어버리면 그 자체가 "훅" 하는
            # 원인이 된다. 아래 'inf' 케이스와 동일하게 직전 조향값을 유지한다.
            return self.prev_steering_command

        target_slope = DMFL.calculate_slope_between_points(
            self.path_data[-LOOKAHEAD_POINTS], self.path_data[-1])

        # 두 점의 y가 같으면 문자열 'inf' 를 반환한다. 비교하면 TypeError 로 콜백이 죽는다.
        if not isinstance(target_slope, (int, float)):
            self.slope_value = None
            # 값이 이상할 때 0으로 확 꺾어버리면 그 자체로 "훅" 하는 원인이 된다.
            # 대신 직전 조향값을 그대로 유지해서 한 프레임 정도는 부드럽게 넘어간다.
            return self.prev_steering_command

        self.slope_value = target_slope

        # 1) 저역통과 필터 (지수이동평균): 프레임 간 튀는 값을 완화
        if self.filtered_slope is None:
            self.filtered_slope = target_slope
        else:
            self.filtered_slope = (
                SLOPE_FILTER_ALPHA * target_slope
                + (1 - SLOPE_FILTER_ALPHA) * self.filtered_slope
            )

        # 1.5) 데드밴드: 직선인데 검출이 살짝 흔들려서 생기는 미세한 기울기는
        # 아예 0으로 죽여서 조향이 씰룩거리지 않게 한다.
        error_deg = apply_deadband(self.filtered_slope, STEER_DEADBAND_DEG)

        # 2) PID: 목표는 기울기 0(직진)이므로 오차 = error_deg (데드밴드 적용된 filtered_slope)
        raw_steering = self.steering_pid.compute(error_deg, self.timer_period)

        # 3) 변화율 제한: 한 틱에 너무 크게 바뀌지 않도록
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
