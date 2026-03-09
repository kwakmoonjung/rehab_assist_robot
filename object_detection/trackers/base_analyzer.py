import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Point
from std_srvs.srv import SetBool, Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from ultralytics import YOLO

# [추가] 만들어둔 운동 모듈 임포트
# (같은 폴더에 있거나 패키지 규칙에 맞게 경로를 설정하세요)
from shoulder_press import ShoulderPressAnalyzer 
from bicep_curl import BicepCurlAnalyzer
from lateral_raise import LateralRaiseAnalyzer

class ExerciseAnalyzer:
    """공통 각도 계산기 (base)"""
    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

class UnifiedPoseAnalyzerNode(Node):
    def __init__(self):
        super().__init__('unified_pose_analyzer_node')
        self.bridge = CvBridge()
        
        # --- 상태 관리 ---
        self.is_exercising = False
        self.publish_trigger = False
        self.current_exercise_type = 'shoulder_press' # 기본값
        
        # --- 퍼블리셔 선언 ---
        self.target_3d_pub = self.create_publisher(Point, '/target_correction_3d', 10)
        self.result_pub = self.create_publisher(String, '/exercise_result', 10) # ⭐️ 결과 퍼블리셔
        self.angle_pub = self.create_publisher(Float32, '/patient_shoulder_angle', 10)

        # --- 구독(Subscription) 선언 ---
        # 듀얼/싱글 카메라 모두 대응하기 위해 전부 구독합니다.
        self.create_subscription(Image, '/robot/camera/color/image_raw', self.front_callback, qos_profile_sensor_data)    
        self.create_subscription(Image, '/fixed/camera/color/image_raw', self.side_callback, qos_profile_sensor_data) 
        self.create_subscription(Image, '/robot/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/robot/camera/color/camera_info', self.camera_info_callback, qos_profile_sensor_data)

        # 시스템 컨트롤러가 "운동 종목"을 바꾸기 위해 쏘는 토픽 수신
        self.create_subscription(String, '/set_exercise_type', self.set_exercise_type_cb, 10)

        # --- 서비스 서버 선언 ---
        self.srv_set_exercise = self.create_service(SetBool, '/set_exercise_state', self.set_exercise_state_cb)
        self.srv_publish_3d = self.create_service(Trigger, '/publish_target_3d', self.publish_3d_cb)

        # --- 모델 및 변수 초기화 ---
        self.model = YOLO('yolo11n-pose.pt')
        self.front_raw = None
        self.side_raw = None
        self.depth_frame = None
        self.intrinsics = None

        # --- [핵심] 운동 분석기 딕셔너리 ---
        # 로깅을 위해 콜백(self.publish_result_cb)을 모두 주입합니다.
        self.analyzers = {
            'shoulder_press': ShoulderPressAnalyzer(publish_callback=self.publish_result_cb),
            'bicep_curl': BicepCurlAnalyzer(publish_callback=self.publish_result_cb),
            'lateral_raise': LateralRaiseAnalyzer(publish_callback=self.publish_result_cb)
        }
        self.current_analyzer = self.analyzers[self.current_exercise_type]

        # 타이머 기반 프레임 처리 (사레레 방식 적용)
        self.timer = self.create_timer(0.033, self.process_frame)
        self.get_logger().info("🚀 [통합 비전 노드] 시작 완료! 명령 대기 중...")

    # ================= 콜백 함수들 =================
    def publish_result_cb(self, data_dict):
        """Tracker에서 생성된 데이터를 String으로 발행"""
        msg = String()
        msg.data = json.dumps(data_dict, ensure_ascii=False)
        self.result_pub.publish(msg)

    def set_exercise_type_cb(self, msg):
        """컨트롤러로부터 운동 종류 변경 명령 수신"""
        req_type = msg.data
        if req_type in self.analyzers:
            self.current_exercise_type = req_type
            self.current_analyzer = self.analyzers[req_type]
            self.get_logger().info(f"🔄 운동 종목 변경됨: {req_type}")
        else:
            self.get_logger().warn(f"⚠️ 알 수 없는 운동: {req_type}")

    def set_exercise_state_cb(self, request, response):
        """운동 로깅 On/Off"""
        self.is_exercising = request.data
        if self.is_exercising:
            self.current_analyzer.tracker.reset() 
            msg = f"✅ [{self.current_exercise_type}] 운동 분석 시작"
        else:
            self.current_analyzer.tracker.emit_data() # 종료 시 최종 기록 발행
            msg = "⏸️ 운동 분석 대기 상태로 전환"
        
        self.get_logger().info(msg)
        response.success = True
        response.message = msg
        return response

    def publish_3d_cb(self, request, response):
        self.publish_trigger = True
        response.success = True
        response.message = "🎯 3D 교정 좌표 발행 요청 수신됨."
        return response

    # ================= 카메라 및 YOLO 처리 =================
    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {"fx": msg.k[0], "fy": msg.k[4], "ppx": msg.k[2], "ppy": msg.k[5]}

    def front_callback(self, msg):
        self.front_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def side_callback(self, msg):
        self.side_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def process_frame(self):
        """타이머에 의해 반복 실행되며 현재 설정된 운동을 분석합니다."""
        if self.front_raw is None: return
        
        image = self.front_raw.copy()
        front_img = cv2.resize(image, (640, 480))
        
        # YOLO 추론
        results = self.model(front_img, verbose=False, device='cpu')[0]
        target_3d_coord = None

        # 1. 사레레 (듀얼 카메라 필요)
        if self.current_exercise_type == 'lateral_raise':
            if self.side_raw is not None:
                side_img = cv2.resize(self.side_raw.copy(), (640, 480))
                res_side = self.model(side_img, verbose=False)[0]
                
                if results.keypoints is not None and res_side.keypoints is not None:
                    if len(results.keypoints.data) > 0 and len(res_side.keypoints.data) > 0:
                        front_kpts = results.keypoints.data[0].cpu().numpy()
                        side_kpts = res_side.keypoints.data[0].cpu().numpy()
                        
                        if self.is_exercising:
                            self.current_analyzer.analyze_dual(front_kpts, side_kpts, front_img, side_img)

                combined_image = np.hstack((front_img, side_img))
                cv2.imshow('Unified Tracker', combined_image)
                cv2.waitKey(1)
            return # 사레레는 여기서 종료

        # 2. 숄더 프레스 / 이두 컬 (싱글 카메라)
        if results.keypoints is not None and len(results.keypoints.xyn) > 0:
            kpts = results.keypoints.xyn[0].cpu().numpy()
            
            # --- [수정된 부분] 신뢰도 값이 확실히 있을 때만 numpy로 변환 ---
            kpt_conf = None
            if hasattr(results.keypoints, 'conf') and results.keypoints.conf is not None:
                if len(results.keypoints.conf) > 0:
                    kpt_conf = results.keypoints.conf[0].cpu().numpy()

            if len(kpts) >= 13:
                if self.is_exercising:
                    if self.current_exercise_type == 'bicep_curl':
                        target_angle, feedback, target_pixel = self.current_analyzer.analyze(kpts, front_img, kpt_conf)
                        # 이두컬 3D 보정 좌표 구하기 (target_pixel 이용)
                    else: # 숄더프레스
                        target_angle, feedback = self.current_analyzer.analyze(kpts, front_img)
                        # 숄더프레스 3D 보정 좌표 구하기 

                else:
                    cv2.putText(front_img, f"IDLE: Ready for {self.current_exercise_type}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Unified Tracker', front_img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedPoseAnalyzerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()