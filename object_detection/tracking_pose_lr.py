import sys
import os
import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data # [추가] QoS 프로필 임포트
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
# 0. 데이터 로깅 모듈 (기존 유지)
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
        self.good_frame_count = 0
        self.elbow_angle_sum = 0.0
        self.shoulder_angle_sum = 0.0
        self.trunk_angle_sum = 0.0
        self.last_feedback = "No data yet"
        
        self.warning_counts = {
            "lean_back_momentum": 0,
            "chest_down": 0,
            "arms_too_high": 0,
            "arm_balance_issue": 0,
        }
        self.save()

    def update_frame(self, elbow_angle, shoulder_angle, trunk_angle, feedback, is_correct):
        self.frame_count += 1
        if is_correct:
            self.good_frame_count += 1

        self.elbow_angle_sum += float(elbow_angle)
        self.shoulder_angle_sum += float(shoulder_angle)
        self.trunk_angle_sum += float(trunk_angle)
        self.last_feedback = feedback
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._count_warning(feedback)

        if self.frame_count % 15 == 0:
            self.save()

    def increment_rep(self, rep_count):
        self.rep_count = int(rep_count)
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def _count_warning(self, feedback):
        if feedback == "Warning: Don't lean back! No momentum.":
            self.warning_counts["lean_back_momentum"] += 1
        elif feedback == "Warning: Keep your chest up!":
            self.warning_counts["chest_down"] += 1
        elif feedback == "Warning: Arms too high! Lower them.":
            self.warning_counts["arms_too_high"] += 1
        elif feedback == "Warning: Balance your arms!":
            self.warning_counts["arm_balance_issue"] += 1

    def _safe_avg(self, value_sum):
        if self.frame_count == 0:
            return 0.0
        return round(value_sum / self.frame_count, 2)

    def save(self):
        good_ratio = 0.0
        if self.frame_count > 0:
            good_ratio = round((self.good_frame_count / self.frame_count) * 100.0, 2)

        data = {
            "exercise_type": self.exercise_type,
            "session_started_at": self.session_started_at,
            "last_updated_at": self.last_updated_at,
            "rep_count": self.rep_count,
            "frame_count": self.frame_count,
            "good_frame_count": self.good_frame_count,
            "good_posture_ratio": good_ratio,
            "avg_elbow_angle": self._safe_avg(self.elbow_angle_sum),
            "avg_shoulder_angle": self._safe_avg(self.shoulder_angle_sum),
            "avg_trunk_angle": self._safe_avg(self.trunk_angle_sum),
            "warning_counts": self.warning_counts,
            "last_feedback": self.last_feedback,
        }

        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"log save error: {e}")

# ==========================================
# 1. 듀얼 분석용 운동 분석기 모듈 (YOLO 버전)
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
        h, w, _ = front_img.shape

        # --- [1. 정면 카메라 데이터 처리] ---
        l_sh_f = front_kpts[5][:2]
        l_el_f = front_kpts[7][:2]
        l_hip_f = front_kpts[11][:2]
        l_wr_f = front_kpts[9][:2]   

        r_sh_f = front_kpts[6][:2]
        r_el_f = front_kpts[8][:2]
        r_hip_f = front_kpts[12][:2]
        r_wr_f = front_kpts[10][:2]  

        pts_f = {
            'l_sh': (int(l_sh_f[0]), int(l_sh_f[1])), 'l_el': (int(l_el_f[0]), int(l_el_f[1])),
            'l_hip': (int(l_hip_f[0]), int(l_hip_f[1])), 'r_sh': (int(r_sh_f[0]), int(r_sh_f[1])),
            'r_el': (int(r_el_f[0]), int(r_el_f[1])), 'r_hip': (int(r_hip_f[0]), int(r_hip_f[1]))
        }

        cv2.line(front_img, pts_f['l_hip'], pts_f['l_sh'], (0, 255, 0), 3)
        cv2.line(front_img, pts_f['l_sh'], pts_f['l_el'], (0, 255, 0), 3)
        cv2.line(front_img, pts_f['r_hip'], pts_f['r_sh'], (255, 0, 0), 3)
        cv2.line(front_img, pts_f['r_sh'], pts_f['r_el'], (255, 0, 0), 3)
        for pt in pts_f.values(): cv2.circle(front_img, pt, 8, (0, 0, 255), -1)

        l_shoulder_angle = self.calculate_angle(l_hip_f, l_sh_f, l_el_f)
        r_shoulder_angle = self.calculate_angle(r_hip_f, r_sh_f, r_el_f)
        avg_shoulder_angle = (l_shoulder_angle + r_shoulder_angle) / 2.0


        # --- [2. 측면 카메라 데이터 처리] ---
        l_sh_s = side_kpts[5][:2]
        l_hip_s = side_kpts[11][:2]
        l_knee_s = side_kpts[13][:2]

        pts_s = {
            'l_sh': (int(l_sh_s[0]), int(l_sh_s[1])),
            'l_hip': (int(l_hip_s[0]), int(l_hip_s[1])),
            'l_knee': (int(l_knee_s[0]), int(l_knee_s[1]))
        }

        cv2.line(side_img, pts_s['l_hip'], pts_s['l_sh'], (0, 255, 255), 3)
        cv2.line(side_img, pts_s['l_hip'], pts_s['l_knee'], (0, 255, 255), 3)
        for pt in pts_s.values(): cv2.circle(side_img, pt, 8, (0, 0, 255), -1)

        trunk_side_angle = self.calculate_angle(l_sh_s, l_hip_s, l_knee_s)

        # --- [3. 통합 상태 판별 및 카운팅] ---
        is_correct_posture = True
        feedback = "Good Form!"
        color = (0, 255, 0) 

        if trunk_side_angle > 175: 
            feedback = "Warning: Don't lean back! No momentum."
            color = (0, 0, 255)
            is_correct_posture = False
        elif trunk_side_angle < 150: 
            feedback = "Warning: Keep your chest up!"
            color = (0, 165, 255)
            is_correct_posture = False
        elif l_shoulder_angle > 100 or r_shoulder_angle > 100: 
            feedback = "Warning: Arms too high! Lower them."
            color = (0, 0, 255)
            is_correct_posture = False
        elif abs(l_shoulder_angle - r_shoulder_angle) > 20: 
            feedback = "Warning: Balance your arms!"
            color = (0, 165, 255)
            is_correct_posture = False

        if is_correct_posture:
            is_down_pose = (l_shoulder_angle < 40) and (r_shoulder_angle < 40)
            is_up_pose = (80 <= l_shoulder_angle <= 95) and (80 <= r_shoulder_angle <= 95)

            if is_down_pose:
                if self.state == "UP":
                    self.state = "DOWN"
                feedback = "Ready... Raise your arms!"
                color = (0, 255, 255) 
            elif is_up_pose:
                if self.state == "DOWN":
                    self.state = "UP"
                    self.count += 1  
                    self.logger.increment_rep(self.count)
                feedback = "Perfect! Slowly lower arms."
                color = (255, 0, 0) 
            elif self.state == "DOWN" and (40 <= l_shoulder_angle < 80):
                feedback = "Raise a bit higher!"
                color = (0, 255, 0)

        self.logger.update_frame(
            elbow_angle=0.0,
            shoulder_angle=avg_shoulder_angle,
            trunk_angle=trunk_side_angle,
            feedback=feedback,
            is_correct=is_correct_posture,
        )

        cv2.putText(front_img, feedback, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(front_img, f"Front(L/R): {int(l_shoulder_angle)} / {int(r_shoulder_angle)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(front_img, f"Count: {self.count}", (w - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        cv2.putText(side_img, f"[SIDE VIEW]", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(side_img, f"Trunk Angle: {int(trunk_side_angle)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return avg_shoulder_angle, feedback, (int(l_wr_f[0]), int(l_wr_f[1])), (int(r_wr_f[0]), int(r_wr_f[1]))

# ==========================================
# 2. ROS2 듀얼 비전 메인 노드 (YOLO + 3D 퍼블리시)
# ==========================================
class PoseTrackingNode(Node):
    def __init__(self, exercise_type='lateral_raise'): 
        super().__init__('pose_tracking_node')
        self.bridge = CvBridge()
        
        # [수정] 10을 qos_profile_sensor_data로 변경
        # 1. Color 이미지 구독
        self.create_subscription(Image, '/fixed/camera/color/image_raw', self.side_callback, qos_profile_sensor_data) 
        self.create_subscription(Image, '/robot/camera/color/image_raw', self.front_callback, qos_profile_sensor_data)    
        
        # 2. Depth & Camera Info 구독
        self.create_subscription(Image, '/robot/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/robot/camera/color/camera_info', self.camera_info_callback, qos_profile_sensor_data)
        
        # 3. 퍼블리셔 선언
        self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        self.left_wrist_3d_pub = self.create_publisher(Point, '/left_wrist_3d', 10)   
        self.right_wrist_3d_pub = self.create_publisher(Point, '/right_wrist_3d', 10) 
        
        self.front_raw = None
        self.side_raw = None
        self.depth_frame = None
        self.intrinsics = None
        
        self.get_logger().info("YOLOv11-Pose 모델을 로드 중입니다...")
        self.pose_model = YOLO('yolo11n-pose.pt') 

        self.analyzers = {
            'lateral_raise': LateralRaiseAnalyzer()    
        }
        self.current_analyzer = self.analyzers.get(exercise_type, LateralRaiseAnalyzer())

        self.get_logger().info(f" [{exercise_type}] 듀얼 카메라 트레이닝 모드 시작!")
        self.get_logger().info(" 창 클릭 후 [스페이스바]를 누르면 양쪽 손목의 3D 좌표가 발행됩니다!")

        self.timer = self.create_timer(0.033, self.display_timer_callback)

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {"fx": msg.k[0], "fy": msg.k[4], "ppx": msg.k[2], "ppy": msg.k[5]}

    def front_callback(self, msg):
        try:
            self.front_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            pass 

    def side_callback(self, msg):
        try:
            self.side_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            pass 

    def _pixel_to_camera_coords(self, x, y, z):
        fx, fy, ppx, ppy = self.intrinsics['fx'], self.intrinsics['fy'], self.intrinsics['ppx'], self.intrinsics['ppy']
        return ((x - ppx) * z / fx, (y - ppy) * z / fy, z)

    def display_timer_callback(self):
        if self.front_raw is not None and self.side_raw is not None:
            # 해상도 640x480 강제 동기화
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
                
                angle_msg = Float32()
                angle_msg.data = float(target_angle)
                self.angle_pub.publish(angle_msg)

                if self.intrinsics is not None and self.depth_frame is not None:
                    depth_resized = cv2.resize(self.depth_frame, (640, 480), interpolation=cv2.INTER_NEAREST)

                    if 0 <= l_wr_pt[0] < 640 and 0 <= l_wr_pt[1] < 480:
                        l_cz = float(depth_resized[l_wr_pt[1], l_wr_pt[0]])
                        if l_cz > 0:
                            l_cx, l_cy, l_cz = self._pixel_to_camera_coords(l_wr_pt[0], l_wr_pt[1], l_cz)
                            left_wrist_3d_coord = (l_cx, l_cy, l_cz)
                            cv2.putText(front_img, f"L3D Z:{int(l_cz)}", (l_wr_pt[0] - 40, l_wr_pt[1] - 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            cv2.circle(front_img, l_wr_pt, 10, (255, 0, 255), -1)

                    if 0 <= r_wr_pt[0] < 640 and 0 <= r_wr_pt[1] < 480:
                        r_cz = float(depth_resized[r_wr_pt[1], r_wr_pt[0]])
                        if r_cz > 0:
                            r_cx, r_cy, r_cz = self._pixel_to_camera_coords(r_wr_pt[0], r_wr_pt[1], r_cz)
                            right_wrist_3d_coord = (r_cx, r_cy, r_cz)
                            cv2.putText(front_img, f"R3D Z:{int(r_cz)}", (r_wr_pt[0] - 40, r_wr_pt[1] - 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            cv2.circle(front_img, r_wr_pt, 10, (0, 255, 255), -1)

            else:
                cv2.putText(front_img, "Waiting for detection...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            combined_image = np.hstack((front_img, side_img))
            cv2.imshow('Dual View PT Trainer (YOLOv11-Pose) - [Front | Side]', combined_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                if left_wrist_3d_coord is not None:
                    l_msg = Point(x=left_wrist_3d_coord[0], y=left_wrist_3d_coord[1], z=left_wrist_3d_coord[2])
                    self.left_wrist_3d_pub.publish(l_msg)
                    self.get_logger().info(f"[좌측 손목] 3D 발행: X:{int(l_msg.x)}, Y:{int(l_msg.y)}, Z:{int(l_msg.z)}")
                
                if right_wrist_3d_coord is not None:
                    r_msg = Point(x=right_wrist_3d_coord[0], y=right_wrist_3d_coord[1], z=right_wrist_3d_coord[2])
                    self.right_wrist_3d_pub.publish(r_msg)
                    self.get_logger().info(f"[우측 손목] 3D 발행: X:{int(r_msg.x)}, Y:{int(r_msg.y)}, Z:{int(r_msg.z)}")

def main(args=None):
    rclpy.init(args=args)
    node = PoseTrackingNode(exercise_type='lateral_raise') 
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()