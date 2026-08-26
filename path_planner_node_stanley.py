import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from interfaces_pkg.msg import LaneInfo, PathPlanningResult
import numpy as np
from scipy.interpolate import CubicSpline

#---------------Variable Setting---------------
SUB_LANE_TOPIC_NAME = "yolov8_lane_info"  # lane_info_extractor 노드에서 퍼블리시하는 타겟 지점 토픽
PUB_TOPIC_NAME = "path_planning_result"   # 경로 계획 결과 퍼블리시 토픽
CAR_CENTER_POINT = (320, 179) # 이미지 상에서 차량 앞 범퍼의 중심이 위치한 픽셀 좌표 (카메라가 차체에
                               # 고정돼 있으므로 이 좌표 자체는 "측정값"이 아니라 항상 같은 상수다)

#----------------------------------------------
#
# ===== 이 파일과 기존 path_planner_node.py 의 차이 =====
# 기존 버전은 차선 중앙점들에 CAR_CENTER_POINT(차량 위치)를 강제로 끼워 넣고 스플라인을 피팅했다.
# 보간(interpolation)이라서 그 곡선은 항상 차량 위치를 정확히 지나가게 되고, 그 결과
# "지금 차량이 중앙에서 얼마나 벗어나 있는지"(횡방향 오차, CTE)는 이 구조에서 항상 0으로만 계산된다.
# 즉 기존 버전은 "경로가 앞으로 어느 방향으로 휘어 있는가"만 보는 pure-pursuit 방식이다.
#
# 이 파일은 그 대신, 실제로 검출된 차선 중앙점만으로 곡선을 만든다. 그러면 그 곡선을 차량이 있는
# 행(y=179)에서 평가했을 때 나오는 x값과, 실제 차량의 x(320) 사이의 차이가 "진짜" 횡방향 오차가
# 된다. 이걸 motion_planner_node_stanley.py 에서 헤딩(기울기) 오차와 함께 조향에 반영한다.
#
# 검출된 차선점이 차량 바로 앞(y=179)까지 닿지 않는 경우가 많으므로, 피팅에는 실제 점만 쓰되
# 출력 샘플 범위(y_new)만 179까지 넓혀서 그 구간은 스플라인의 외삽(extrapolate)으로 채운다.
# 외삽 거리가 너무 크면(검출이 얼마 안 됐는데 179까지 억지로 늘리면) 부정확해질 수 있으니,
# 초반에 CTE 값이 튀는 것 같으면 로그(motion_planner_node_stanley.py 의 CSV)에서 cte 열을 보고
# 판단할 것.
#----------------------------------------------


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        # 파라미터 선언
        self.sub_lane_topic = self.declare_parameter('sub_lane_topic', SUB_LANE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.car_center_point = self.declare_parameter('car_center_point', CAR_CENTER_POINT).value

        # QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 변수 초기화
        self.target_points = []  # 차선의 타겟 지점들 (차선 중앙)

        # 서브스크라이버 설정 (타겟 지점 구독)
        self.lane_sub = self.create_subscription(LaneInfo, self.sub_lane_topic, self.lane_callback, self.qos_profile)

        # 퍼블리셔 설정 (경로 계획 결과 퍼블리시)
        self.publisher = self.create_publisher(PathPlanningResult, self.pub_topic, self.qos_profile)

    def lane_callback(self, msg: LaneInfo):

        # 타겟 지점 받아오기
        self.target_points = msg.target_points

        # 타겟 지점이 3개 이상 모이면 경로 계획 시작
        if len(self.target_points) >= 3:
            self.plan_path()

    def plan_path(self):
        if not self.target_points:
            self.get_logger().warn("No target points available")
            return

        # TargetPoint 객체에서 x, y 값 추출 (차량 위치는 여기서 섞지 않는다 — 위 설명 참고)
        x_points, y_points = zip(*[(tp.target_x, tp.target_y) for tp in self.target_points])

        # y 값을 기준으로 정렬
        sorted_points = sorted(zip(y_points, x_points), key=lambda point: point[0])

        # 같은 y를 가진 점은 평균 내서 하나로 합친다 (CubicSpline은 x가 엄격히 증가해야 함)
        deduped_y = []
        deduped_x_sum = []
        deduped_count = []
        for y, x in sorted_points:
            if deduped_y and abs(deduped_y[-1] - y) < 1e-6:
                deduped_x_sum[-1] += x
                deduped_count[-1] += 1
            else:
                deduped_y.append(y)
                deduped_x_sum.append(x)
                deduped_count.append(1)

        y_points = deduped_y
        x_points = [x_sum / count for x_sum, count in zip(deduped_x_sum, deduped_count)]

        if len(y_points) < 2:
            self.get_logger().warn("Not enough distinct points to plan a path")
            return

        self.get_logger().info(f"Planning path with {len(y_points)} points (car position excluded from fit)")

        try:
            cs = CubicSpline(y_points, x_points, bc_type='natural')
        except ValueError as e:
            self.get_logger().warn(f"Spline fit failed, skipping this frame: {e}")
            return

        # 출력 범위는 차량이 있는 행(car_center_point[1])까지 항상 포함시킨다.
        # 실제 검출 범위보다 더 멀리 나가는 구간은 스플라인의 외삽으로 채워진다.
        y_low = min(y_points)
        y_high = max(max(y_points), self.car_center_point[1])
        y_new = np.linspace(y_low, y_high, 100)
        x_new = cs(y_new)

        path_msg = PathPlanningResult()
        path_msg.x_points = list(x_new)
        path_msg.y_points = list(y_new)

        self.publisher.publish(path_msg)

        self.target_points.clear()


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
