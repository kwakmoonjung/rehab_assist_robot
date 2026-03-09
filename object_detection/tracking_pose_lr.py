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

# ==========================================
# [수정됨] 로그 파일 저장 경로 설정 (날짜 및 시간 추가)
# ==========================================
SAVE_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(SAVE_DIR, exist_ok=True) 

current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(SAVE_DIR, f"exercise_session_log_{current_time_str}.json")

# ==========================================
# 0. 데이터 로깅 모듈 (노약자 맞춤형 지표 및 로봇 제어 지표 적용)
# ==========================================
class ExerciseSessionLogger:
    def __init__(self, log_file, exercise_type):
        self.log_file = log_file
        self.exercise_type = exercise_type
        self.reset()

    def reset(self):
        now_dt = datetime.now()
        self.session_start_dt = now_dt
        self.session_started_at = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        self.last_updated_at = self.session_started_at
        
        self.rep_count = 0
        self.frame_count = 0
        self.good_frame_count = 0
        self.trunk_angle_sum = 0.0
        self.last_feedback = "No data yet"
        
        self.max_rom_left = 0.0   
        self.max_rom_right = 0.0  
        self.rep_start_time = None
        self.rep_durations = []   
        self.tremor_count = 0     
        self.last_l_wr_y = None   
        self.last_r_wr_y = None
        
        self.successful_peaks = [] # 완벽한 정자세 횟수의 최고 각도 모음
        self.all_peaks = []        # [신규] 자세 성공 여부 상관없이 모든 횟수의 최고 각도 모음
        
        self.l_z_history = []
        self.r_z_history = []
        self.max_z_drift = 0.0
        
        self.pure_arom = 0.0

        self.warning_counts = {
            "lean_back_momentum": 0,
            "chest_down": 0,
            "arms_too_high": 0,
            "arm_balance_issue": 0,
        }
        self.save()

    def update_frame(self, l_shoulder_angle, r_shoulder_angle, trunk_angle, feedback, is_correct, l_wr_y=None, r_wr_y=None):
        self.frame_count += 1
        if is_correct:
            self.good_frame_count += 1
            current_max_shoulder = max(l_shoulder_angle, r_shoulder_angle)
            if current_max_shoulder > self.pure_arom:
                self.pure_arom = round(float(current_max_shoulder), 2)

        self.trunk_angle_sum += float(trunk_angle)

        if l_shoulder_angle > self.max_rom_left: 
            self.max_rom_left = round(float(l_shoulder_angle), 2)
        if r_shoulder_angle > self.max_rom_right: 
            self.max_rom_right = round(float(r_shoulder_angle), 2)

        if l_wr_y is not None and r_wr_y is not None:
            if self.last_l_wr_y is not None:
                if abs(l_wr_y - self.last_l_wr_y) > 15 or abs(r_wr_y - self.last_r_wr_y) > 15:
                    if l_shoulder_angle > 30 or r_shoulder_angle > 30:
                        self.tremor_count += 1
            self.last_l_wr_y = l_wr_y
            self.last_r_wr_y = r_wr_y

        self.last_feedback = feedback
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._count_warning(feedback)

        if self.frame_count % 15 == 0:
            self.save()

    def update_depth(self, l_z, r_z):
        if l_z is not None and l_z > 0: self.l_z_history.append(l_z)
        if r_z is not None and r_z > 0: self.r_z_history.append(r_z)

    def increment_rep(self, rep_count):
        self.rep_count = int(rep_count)
        now = datetime.now()
        
        if self.rep_start_time is not None:
            duration = (now - self.rep_start_time).total_seconds()
            self.rep_durations.append(round(duration, 2))
        self.rep_start_time = now 
        
        if self.l_z_history and self.r_z_history:
            l_drift = max(self.l_z_history) - min(self.l_z_history)
            r_drift = max(self.r_z_history) - min(self.r_z_history)
            current_drift = max(l_drift, r_drift)
            if current_drift > self.max_z_drift:
                self.max_z_drift = round(current_drift, 2)
        
        self.l_z_history = []
        self.r_z_history = []

        self.last_updated_at = now.strftime("%Y-%m-%d %H:%M:%S")
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

    def save(self):
        good_ratio = 0.0
        if self.frame_count > 0:
            good_ratio = round((self.good_frame_count / self.frame_count) * 100.0, 2)
            avg_trunk = round(self.trunk_angle_sum / self.frame_count, 2)
        else:
            avg_trunk = 0.0

        session_duration = round((datetime.now() - self.session_start_dt).total_seconds(), 2)
        avg_rep_dur = round(sum(self.rep_durations) / len(self.rep_durations), 2) if self.rep_durations else 0.0

        # 성공한 횟수들의 평균 각도
        avg_successful_peak = 0.0
        if self.successful_peaks:
            avg_successful_peak = round(sum(self.successful_peaks) / len(self.successful_peaks), 2)

        # [신규] 전체 횟수들의 평균 각도 계산
        avg_all_peak = 0.0
        if self.all_peaks:
            avg_all_peak = round(sum(self.all_peaks) / len(self.all_peaks), 2)

        assist_trigger_angle = max(0.0, self.pure_arom - 5.0) if self.pure_arom > 0 else 0.0
        target_prom = min(90.0, self.pure_arom + 10.0) if self.pure_arom > 0 else 0.0

        data = {
            "exercise_type": self.exercise_type,
            "session_started_at": self.session_started_at,
            "last_updated_at": self.last_updated_at,
            "session_duration_sec": session_duration, 
            "rep_count": self.rep_count,
            "robot_assist_parameters": {             
                "pure_arom": self.pure_arom,
                "assist_trigger_angle": assist_trigger_angle,
                "target_prom": target_prom
            },
            "elderly_pt_metrics": {
                "avg_successful_peak_angle": avg_successful_peak, # 완벽한 자세 평균 도달 각도
                "avg_all_peak_angle": avg_all_peak,               # [신규] 자세 무관 전체 횟수 평균 도달 각도
                "max_rom_left": self.max_rom_left,       
                "max_rom_right": self.max_rom_right,     
                "avg_rep_duration_sec": avg_rep_dur,     
                "tremor_count": self.tremor_count,       
                "max_z_depth_drift_mm": self.max_z_drift 
            },
            "performance_stats": {
                "total_frames": self.frame_count,
                "good_posture_ratio": good_ratio,
                "avg_trunk_angle": avg_trunk
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
        self.state = "DOWN"    # 상태: DOWN, RAISING, LOWERING
        self.current_rep_peak = 0.0 
        self.eval_feedback = ""     
        self.eval_color = (0, 255, 0)
        self.rep_has_warning = False 
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

        if trunk_side_angle > 185: 
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

        if not is_correct_posture:
            self.rep_has_warning = True

        # 3단계 상태 머신 적용 (DOWN -> RAISING -> LOWERING)
        is_down_pose = (l_shoulder_angle < 40) and (r_shoulder_angle < 40)

        if self.state == "DOWN":
            if avg_shoulder_angle >= 40:
                self.state = "RAISING"
                self.current_rep_peak = avg_shoulder_angle
                self.rep_has_warning = False
                if self.logger.rep_start_time is None:
                    self.logger.rep_start_time = datetime.now()
                feedback = "Keep going up!"
                color = (0, 255, 0)
            else:
                feedback = "Ready... Raise your arms!"
                color = (0, 255, 255)

        elif self.state == "RAISING":
            # 실시간 최고점 갱신
            if avg_shoulder_angle > self.current_rep_peak:
                self.current_rep_peak = avg_shoulder_angle

            # ⭐️ 하강 시작점 포착 (최고점에서 5도 이상 떨어지는 순간) ⭐️
            if avg_shoulder_angle < self.current_rep_peak - 5.0:
                self.state = "LOWERING"
                
                # [기존] 이 찰나의 순간에 즉시 평가 및 DB 저장 진행 (성공한 횟수)
                if not self.rep_has_warning and self.current_rep_peak >= 70.0:
                    self.logger.successful_peaks.append(round(float(self.current_rep_peak), 2))
                    
                # [신규] 자세 붕괴 및 경고 여부와 상관없이 무조건 1회 사이클 최고점 기록 (전체 평균용)
                if self.current_rep_peak >= 40.0:
                    self.logger.all_peaks.append(round(float(self.current_rep_peak), 2))
                
                if self.current_rep_peak >= 80:
                    self.count += 1
                    self.logger.increment_rep(self.count)
                    self.eval_feedback = "Perfect! Great job."
                    self.eval_color = (0, 255, 0)
                else:
                    needed_angle = 80 - int(self.current_rep_peak)
                    self.eval_feedback = f"Good (Peak: {int(self.current_rep_peak)}). Try {needed_angle} deg more!"
                    self.eval_color = (0, 255, 255)
                
                feedback = "Now slowly lower arms."
                color = (255, 165, 0)
            else:
                if self.current_rep_peak >= 80:
                    feedback = "Perfect height! Now slowly down."
                    color = (255, 0, 0)
                elif is_correct_posture:
                    feedback = "Keep going up!"
                    color = (0, 255, 0)

        elif self.state == "LOWERING":
            if is_down_pose:
                self.state = "DOWN"
                self.current_rep_peak = 0.0 # 다음 횟수를 위해 리셋
                self.eval_feedback = ""     # 피드백 초기화
                
            # 내려가는 동안에는 평가 피드백을 계속 유지하여 보여줌
            if self.eval_feedback != "" and is_correct_posture:
                feedback = self.eval_feedback
                color = self.eval_color
            elif is_correct_posture:
                feedback = "Slowly down..."
                color = (255, 165, 0)

        self.logger.update_frame(
            l_shoulder_angle=l_shoulder_angle,
            r_shoulder_angle=r_shoulder_angle,
            trunk_angle=trunk_side_angle,
            feedback=feedback,
            is_correct=is_correct_posture,
            l_wr_y=l_wr_f[1],
            r_wr_y=r_wr_f[1]
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
        
        self.create_subscription(Image, '/fixed/camera/color/image_raw', self.side_callback, qos_profile_sensor_data) 
        self.create_subscription(Image, '/robot/camera/color/image_raw', self.front_callback, qos_profile_sensor_data)    
        
        self.create_subscription(Image, '/robot/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/robot/camera/color/camera_info', self.camera_info_callback, qos_profile_sensor_data)
        
        self.angle_pub = self.create_publisher(Float32, '/patient_shoulder_angle', 10)
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

        self.get_logger().info(f" [{exercise_type}] 듀얼 카메라 트레이닝 모드 시작! (데이터 저장 경로: {SAVE_DIR})")
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
            front_img = cv2.resize(self.front_raw, (640, 480))
            side_img = cv2.resize(self.side_raw, (640, 480))

            res_front = self.pose_model(front_img, verbose=False)[0]
            res_side = self.pose_model(side_img, verbose=False)[0]

            left_wrist_3d_coord = None
            right_wrist_3d_coord = None
            
            l_cz_val = None
            r_cz_val = None

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
                            l_cz_val = l_cz 
                            l_cx, l_cy, l_cz = self._pixel_to_camera_coords(l_wr_pt[0], l_wr_pt[1], l_cz)
                            left_wrist_3d_coord = (l_cx, l_cy, l_cz)
                            cv2.putText(front_img, f"L3D Z:{int(l_cz)}", (l_wr_pt[0] - 40, l_wr_pt[1] - 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            cv2.circle(front_img, l_wr_pt, 10, (255, 0, 255), -1)

                    if 0 <= r_wr_pt[0] < 640 and 0 <= r_wr_pt[1] < 480:
                        r_cz = float(depth_resized[r_wr_pt[1], r_wr_pt[0]])
                        if r_cz > 0:
                            r_cz_val = r_cz 
                            r_cx, r_cy, r_cz = self._pixel_to_camera_coords(r_wr_pt[0], r_wr_pt[1], r_cz)
                            right_wrist_3d_coord = (r_cx, r_cy, r_cz)
                            cv2.putText(front_img, f"R3D Z:{int(r_cz)}", (r_wr_pt[0] - 40, r_wr_pt[1] - 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            cv2.circle(front_img, r_wr_pt, 10, (0, 255, 255), -1)
                
                self.current_analyzer.logger.update_depth(l_cz_val, r_cz_val)

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