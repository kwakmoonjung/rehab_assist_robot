import sys
import os
import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np

# YOLO 임포트
from ultralytics import YOLO 

# 로그 파일 저장 경로
LOG_FILE = os.path.expanduser("~/exercise_session_log.json")

# ==========================================
# 0. JSON 로깅 모듈 (Lateral Raise 최적화 버전)
# ==========================================
class ExerciseSessionLogger:
    def __init__(self, log_file, exercise_type):
        self.log_file = log_file
        self.exercise_type = exercise_type
        self.reset()

    def reset(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_started_at = now
        self.last_updated_at = now
        self.rep_count = 0

        self.frame_count = 0
        self.analyzed_frame_count = 0 # 실제 관절이 감지되어 계산된 프레임
        self.good_frame_count = 0

        # 사레레 특화 데이터
        self.l_shoulder_angle_sum = 0.0
        self.r_shoulder_angle_sum = 0.0
        self.trunk_angle_sum = 0.0
        self.max_rom_angle = 0.0 # 세션 중 가장 높게 올린 팔 각도

        self.last_feedback = "No data yet"
        self.warning_counts = {
            "lean_back_momentum": 0,
            "chest_down": 0,
            "arms_too_high": 0,
            "arm_balance_issue": 0,
        }
        self.save()

    def update_frame(
        self,
        l_shoulder=None,
        r_shoulder=None,
        trunk_angle=None,
        feedback="No data yet",
        is_correct=False,
        has_valid_measurement=False
    ):
        self.frame_count += 1
        self.last_feedback = feedback
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if has_valid_measurement:
            self.analyzed_frame_count += 1
            self.l_shoulder_angle_sum += float(l_shoulder)
            self.r_shoulder_angle_sum += float(r_shoulder)
            self.trunk_angle_sum += float(trunk_angle)

            # 최대 가동 범위(ROM) 갱신
            current_max = max(l_shoulder, r_shoulder)
            if current_max > self.max_rom_angle:
                self.max_rom_angle = round(float(current_max), 2)

            if is_correct:
                self.good_frame_count += 1
            
            # 경고 카운트 (올바르지 않은 자세일 때 피드백 분석)
            if not is_correct:
                self._count_warning(feedback)

        if self.frame_count % 15 == 0:
            self.save()

    def increment_rep(self, rep_count):
        self.rep_count = int(rep_count)
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def _count_warning(self, feedback):
        if "lean back" in feedback:
            self.warning_counts["lean_back_momentum"] += 1
        elif "chest up" in feedback:
            self.warning_counts["chest_down"] += 1
        elif "too high" in feedback:
            self.warning_counts["arms_too_high"] += 1
        elif "Balance" in feedback:
            self.warning_counts["arm_balance_issue"] += 1

    def _safe_avg(self, value_sum):
        if self.analyzed_frame_count == 0:
            return 0.0
        return round(value_sum / self.analyzed_frame_count, 2)

    def save(self):
        good_ratio = 0.0
        if self.analyzed_frame_count > 0:
            good_ratio = round((self.good_frame_count / self.analyzed_frame_count) * 100.0, 2)

        data = {
            "exercise_type": self.exercise_type,
            "session_started_at": self.session_started_at,
            "last_updated_at": self.last_updated_at,
            "rep_count": self.rep_count,
            "stats": {
                "total_frames": self.frame_count,
                "analyzed_frames": self.analyzed_frame_count,
                "good_posture_ratio": f"{good_ratio}%",
                "max_rom_angle": self.max_rom_angle
            },
            "averages": {
                "avg_l_shoulder_angle": self._safe_avg(self.l_shoulder_angle_sum),
                "avg_r_shoulder_angle": self._safe_avg(self.r_shoulder_angle_sum),
                "avg_trunk_angle": self._safe_avg(self.trunk_angle_sum),
                "avg_asymmetry": round(abs(self._safe_avg(self.l_shoulder_angle_sum) - self._safe_avg(self.r_shoulder_angle_sum)), 2)
            },
            "warning_counts": self.warning_counts,
            "last_feedback": self.last_feedback,
        }

        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"log save error: {e}")

# ==========================================
# 1. 듀얼 분석용 운동 분석기 모듈
# ==========================================
class ExerciseAnalyzer:
    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

class LateralRaiseAnalyzer(ExerciseAnalyzer):
    def __init__(self):
        self.count = 0         
        self.state = "DOWN"    
        self.logger = ExerciseSessionLogger(LOG_FILE, "lateral_raise")

    def analyze_dual(self, front_kpts, side_kpts, front_img, side_img):
        # --- [1. 정면 카메라 데이터 처리 (YOLO Index)] ---
        # 5,7,11: 좌측 어깨-팔꿈치-골반 / 6,8,12: 우측
        l_sh_f, l_el_f, l_hip_f = front_kpts[5][:2], front_kpts[7][:2], front_kpts[11][:2]
        r_sh_f, r_el_f, r_hip_f = front_kpts[6][:2], front_kpts[8][:2], front_kpts[12][:2]
        l_wr_f, r_wr_f = front_kpts[9][:2], front_kpts[10][:2]

        pts_f = {
            'l_sh': (int(l_sh_f[0]), int(l_sh_f[1])), 'l_el': (int(l_el_f[0]), int(l_el_f[1])),
            'l_hip': (int(l_hip_f[0]), int(l_hip_f[1])), 'r_sh': (int(r_sh_f[0]), int(r_sh_f[1])),
            'r_el': (int(r_el_f[0]), int(r_el_f[1])), 'r_hip': (int(r_hip_f[0]), int(r_hip_f[1]))
        }

        # 시각화 뼈대
        cv2.line(front_img, pts_f['l_hip'], pts_f['l_sh'], (0, 255, 0), 3)
        cv2.line(front_img, pts_f['l_sh'], pts_f['l_el'], (0, 255, 0), 3)
        cv2.line(front_img, pts_f['r_hip'], pts_f['r_sh'], (255, 0, 0), 3)
        cv2.line(front_img, pts_f['r_sh'], pts_f['r_el'], (255, 0, 0), 3)

        l_shoulder_angle = self.calculate_angle(l_hip_f, l_sh_f, l_el_f)
        r_shoulder_angle = self.calculate_angle(r_hip_f, r_sh_f, r_el_f)
        avg_shoulder_angle = (l_shoulder_angle + r_shoulder_angle) / 2.0

        # --- [2. 측면 카메라 데이터 처리] ---
        # 5,11,13: 어깨-골반-무릎 (허리 곧음 판별)
        l_sh_s, l_hip_s, l_knee_s = side_kpts[5][:2], side_kpts[11][:2], side_kpts[13][:2]
        trunk_side_angle = self.calculate_angle(l_sh_s, l_hip_s, l_knee_s)

        # --- [3. 상태 판별 및 로깅] ---
        is_correct = True
        feedback = "Good Form!"
        color = (0, 255, 0) 

        if trunk_side_angle > 175: 
            feedback = "Warning: Don't lean back! No momentum."
            is_correct = False; color = (0, 0, 255)
        elif trunk_side_angle < 150: 
            feedback = "Warning: Keep your chest up!"
            is_correct = False; color = (0, 165, 255)
        elif l_shoulder_angle > 100 or r_shoulder_angle > 100: 
            feedback = "Warning: Arms too high! Lower them."
            is_correct = False; color = (0, 0, 255)
        elif abs(l_shoulder_angle - r_shoulder_angle) > 20: 
            feedback = "Warning: Balance your arms!"
            is_correct = False; color = (0, 165, 255)

        # 카운팅 로직
        if is_correct:
            is_down = (l_shoulder_angle < 40) and (r_shoulder_angle < 40)
            is_up = (80 <= l_shoulder_angle <= 95) and (80 <= r_shoulder_angle <= 95)
            if is_down and self.state == "UP": self.state = "DOWN"
            elif is_up and self.state == "DOWN":
                self.state = "UP"; self.count += 1
                self.logger.increment_rep(self.count)

        # ⭐️ JSON 로그 업데이트 (개별 각도 및 측면 허리 각도 전달)
        self.logger.update_frame(
            l_shoulder=l_shoulder_angle,
            r_shoulder=r_shoulder_angle,
            trunk_angle=trunk_side_angle,
            feedback=feedback,
            is_correct=is_correct,
            has_valid_measurement=True
        )

        # UI 텍스트 출력
        cv2.putText(front_img, feedback, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(front_img, f"L/R Angle: {int(l_shoulder_angle)} / {int(r_shoulder_angle)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(front_img, f"Count: {self.count}", (front_img.shape[1] - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        return avg_shoulder_angle, feedback, (int(l_wr_f[0]), int(l_wr_f[1])), (int(r_wr_f[0]), int(r_wr_f[1]))

# ==========================================
# 2. ROS2 메인 노드
# ==========================================
class PoseTrackingNode(Node):
    def __init__(self, exercise_type='lateral_raise'): 
        super().__init__('pose_tracking_node')
        self.bridge = CvBridge()
        
        # 이미지 및 정보 구독 (QoS 적용)
        self.create_subscription(Image, '/fixed/camera/color/image_raw', self.side_callback, qos_profile_sensor_data) 
        self.create_subscription(Image, '/robot/camera/color/image_raw', self.front_callback, qos_profile_sensor_data)    
        self.create_subscription(Image, '/robot/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/robot/camera/color/camera_info', self.camera_info_callback, qos_profile_sensor_data)
        
        # 퍼블리셔
        self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        self.left_wrist_3d_pub = self.create_publisher(Point, '/left_wrist_3d', 10)   
        self.right_wrist_3d_pub = self.create_publisher(Point, '/right_wrist_3d', 10) 
        
        self.front_raw = None
        self.side_raw = None
        self.depth_frame = None
        self.intrinsics = None
        
        self.get_logger().info("YOLOv11-Pose 모델 로드 중...")
        self.pose_model = YOLO('yolo11n-pose.pt') 
        self.current_analyzer = LateralRaiseAnalyzer()

        self.get_logger().info(f"💪 [{exercise_type}] 듀얼 카메라 트레이닝 모드 시작!")
        self.timer = self.create_timer(0.033, self.display_timer_callback)

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {"fx": msg.k[0], "fy": msg.k[4], "ppx": msg.k[2], "ppy": msg.k[5]}

    def front_callback(self, msg):
        try: self.front_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass 

    def side_callback(self, msg):
        try: self.side_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass 

    def _pixel_to_camera_coords(self, x, y, z):
        fx, fy, ppx, ppy = self.intrinsics['fx'], self.intrinsics['fy'], self.intrinsics['ppx'], self.intrinsics['ppy']
        return ((x - ppx) * z / fx, (y - ppy) * z / fy, z)

    def display_timer_callback(self):
        if self.front_raw is not None and self.side_raw is not None:
            front_img = cv2.resize(self.front_raw, (640, 480))
            side_img = cv2.resize(self.side_raw, (640, 480))

            res_front = self.pose_model(front_img, verbose=False)[0]
            res_side = self.pose_model(side_img, verbose=False)[0]

            left_wrist_3d_coord = None
            right_wrist_3d_coord = None

            if res_front.keypoints is not None and len(res_front.keypoints.data) > 0 and \
               res_side.keypoints is not None and len(res_side.keypoints.data) > 0:
                
                front_kpts = res_front.keypoints.data[0].cpu().numpy()
                side_kpts = res_side.keypoints.data[0].cpu().numpy()

                target_angle, feedback, l_wr_pt, r_wr_pt = self.current_analyzer.analyze_dual(
                    front_kpts, side_kpts, front_img, side_img
                )
                
                angle_msg = Float32(); angle_msg.data = float(target_angle)
                self.angle_pub.publish(angle_msg)

                # 3D 좌표 변환 (기존 기능 유지)
                if self.intrinsics is not None and self.depth_frame is not None:
                    depth_resized = cv2.resize(self.depth_frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                    for pt, is_left in [(l_wr_pt, True), (r_wr_pt, False)]:
                        if 0 <= pt[0] < 640 and 0 <= pt[1] < 480:
                            z = float(depth_resized[pt[1], pt[0]])
                            if z > 0:
                                coords = self._pixel_to_camera_coords(pt[0], pt[1], z)
                                if is_left: left_wrist_3d_coord = coords
                                else: right_wrist_3d_coord = coords

            combined_image = np.hstack((front_img, side_img))
            cv2.imshow('Dual View PT Trainer (YOLOv11-Pose)', combined_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32: # Spacebar
                if left_wrist_3d_coord:
                    self.left_wrist_3d_pub.publish(Point(x=float(left_wrist_3d_coord[0]), y=float(left_wrist_3d_coord[1]), z=float(left_wrist_3d_coord[2])))
                if right_wrist_3d_coord:
                    self.right_wrist_3d_pub.publish(Point(x=float(right_wrist_3d_coord[0]), y=float(right_wrist_3d_coord[1]), z=float(right_wrist_3d_coord[2])))
                self.get_logger().info("✅ Wrist 3D points published!")

def main(args=None):
    rclpy.init(args=args)
    node = PoseTrackingNode() 
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()