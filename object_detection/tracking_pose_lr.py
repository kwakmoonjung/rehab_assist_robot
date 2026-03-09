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

# ==========================================
# 로그 파일 저장 경로 설정 (날짜 및 시간 추가)
# ==========================================
SAVE_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(SAVE_DIR, exist_ok=True) 

current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(SAVE_DIR, f"exercise_session_log_{current_time_str}.json")

# ==========================================
# 0. 데이터 로깅 모듈
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
        
        self.successful_peaks = [] 
        self.all_peaks = []        
        
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

    def update_depth(self, r_z):
        if r_z is not None and r_z > 0: self.r_z_history.append(r_z)

    def increment_rep(self, rep_count):
        self.rep_count = int(rep_count)
        now = datetime.now()
        
        if self.rep_start_time is not None:
            duration = (now - self.rep_start_time).total_seconds()
            self.rep_durations.append(round(duration, 2))
        self.rep_start_time = now 
        
        if self.r_z_history:
            r_drift = max(self.r_z_history) - min(self.r_z_history)
            if r_drift > self.max_z_drift:
                self.max_z_drift = round(r_drift, 2)
        
        self.r_z_history = []
        self.last_updated_at = now.strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def _count_warning(self, feedback):
        if feedback == "Warning: Keep your body straight!":
            self.warning_counts["lean_back_momentum"] += 1
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

        avg_successful_peak = 0.0
        if self.successful_peaks:
            avg_successful_peak = round(sum(self.successful_peaks) / len(self.successful_peaks), 2)

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
                "avg_successful_peak_angle": avg_successful_peak,
                "avg_all_peak_angle": avg_all_peak,
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
        self.state = "DOWN"
        self.current_rep_peak = 0.0 
        self.eval_feedback = ""     
        self.eval_color = (0, 255, 0)
        self.rep_has_warning = False 
        self.logger = ExerciseSessionLogger(LOG_FILE, "lateral_raise")

    def analyze_dual(self, fixed_kpts, robot_kpts, fixed_img, robot_img):
        h, w, _ = fixed_img.shape

        # --- [1. 정면 전체 자세 (Fixed Camera)] ---
        nose = fixed_kpts[0][:2]
        
        l_sh_f = fixed_kpts[5][:2]
        l_el_f = fixed_kpts[7][:2]
        l_hip_f = fixed_kpts[11][:2]
        l_wr_f = fixed_kpts[9][:2]   

        r_sh_f = fixed_kpts[6][:2]
        r_el_f = fixed_kpts[8][:2]
        r_hip_f = fixed_kpts[12][:2]
        r_wr_f = fixed_kpts[10][:2]  

        pts_f = {
            'l_sh': (int(l_sh_f[0]), int(l_sh_f[1])), 'l_el': (int(l_el_f[0]), int(l_el_f[1])),
            'l_hip': (int(l_hip_f[0]), int(l_hip_f[1])), 'r_sh': (int(r_sh_f[0]), int(r_sh_f[1])),
            'r_el': (int(r_el_f[0]), int(r_el_f[1])), 'r_hip': (int(r_hip_f[0]), int(r_hip_f[1])),
            'nose': (int(nose[0]), int(nose[1]))
        }

        # 정면 뼈대 시각화
        cv2.line(fixed_img, pts_f['l_hip'], pts_f['l_sh'], (0, 255, 0), 3)
        cv2.line(fixed_img, pts_f['l_sh'], pts_f['l_el'], (0, 255, 0), 3)
        cv2.line(fixed_img, pts_f['r_hip'], pts_f['r_sh'], (255, 0, 0), 3)
        cv2.line(fixed_img, pts_f['r_sh'], pts_f['r_el'], (255, 0, 0), 3)
        for pt in pts_f.values(): cv2.circle(fixed_img, pt, 8, (0, 0, 255), -1)

        # 정면 기준 각도 계산
        l_shoulder_angle = self.calculate_angle(l_hip_f, l_sh_f, l_el_f)
        r_shoulder_angle = self.calculate_angle(r_hip_f, r_sh_f, r_el_f)
        avg_shoulder_angle = (l_shoulder_angle + r_shoulder_angle) / 2.0

        # 몸통 기울기 계산 (Nose와 양 골반 중앙의 수직선 비교)
        mid_hip = [(l_hip_f[0] + r_hip_f[0]) / 2, (l_hip_f[1] + r_hip_f[1]) / 2]
        vertical_ref = [mid_hip[0], mid_hip[1] - 0.1]
        trunk_front_angle = self.calculate_angle(vertical_ref, mid_hip, nose)

        # --- [2. 로봇 카메라 우측 팔 인식 (Robot Camera)] ---
        # 8번 인덱스 = 오른쪽 팔꿈치(Right Elbow)
        r_el_robot = robot_kpts[8][:2]
        r_el_pt = (int(r_el_robot[0]), int(r_el_robot[1]))
        
        # 로봇 카메라에는 대상점(오른쪽 팔꿈치)만 시각화
        cv2.circle(robot_img, r_el_pt, 12, (0, 255, 255), -1)
        cv2.putText(robot_img, "[Robot View] Tracking Right Elbow", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # --- [3. 통합 상태 판별 및 카운팅] ---
        is_correct_posture = True
        feedback = "Good Form!"
        color = (0, 255, 0) 

        # 정면 카메라를 기준으로 몸통의 기울어짐과 양팔 밸런스 체크
        if trunk_front_angle > 15: 
            feedback = "Warning: Keep your body straight!"
            color = (0, 0, 255)
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

        # 상태 머신
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
            if avg_shoulder_angle > self.current_rep_peak:
                self.current_rep_peak = avg_shoulder_angle

            if avg_shoulder_angle < self.current_rep_peak - 5.0:
                self.state = "LOWERING"
                
                if not self.rep_has_warning and self.current_rep_peak >= 70.0:
                    self.logger.successful_peaks.append(round(float(self.current_rep_peak), 2))
                    
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
                self.current_rep_peak = 0.0 
                self.eval_feedback = ""     
                
            if self.eval_feedback != "" and is_correct_posture:
                feedback = self.eval_feedback
                color = self.eval_color
            elif is_correct_posture:
                feedback = "Slowly down..."
                color = (255, 165, 0)

        self.logger.update_frame(
            l_shoulder_angle=l_shoulder_angle,
            r_shoulder_angle=r_shoulder_angle,
            trunk_angle=trunk_front_angle,
            feedback=feedback,
            is_correct=is_correct_posture,
            l_wr_y=l_wr_f[1],
            r_wr_y=r_wr_f[1]
        )

        cv2.putText(fixed_img, "[Full Body View]", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(fixed_img, feedback, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(fixed_img, f"Angles(L/R): {int(l_shoulder_angle)} / {int(r_shoulder_angle)}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(fixed_img, f"Count: {self.count}", (w - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        return avg_shoulder_angle, feedback, r_el_pt

# ==========================================
# 2. ROS2 듀얼 비전 메인 노드 (YOLO + 3D 퍼블리시)
# ==========================================
class PoseTrackingNode(Node):
    def __init__(self, exercise_type='lateral_raise'): 
        super().__init__('pose_tracking_node')
        self.bridge = CvBridge()
        
        # Fixed Camera: 정면 (Full Body)
        self.create_subscription(Image, '/fixed/camera/color/image_raw', self.fixed_callback, qos_profile_sensor_data) 
        # Robot Camera: 우측 팔 
        self.create_subscription(Image, '/robot/camera/color/image_raw', self.robot_callback, qos_profile_sensor_data)    
        
        # Robot Camera 뎁스
        self.create_subscription(Image, '/robot/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/robot/camera/color/camera_info', self.camera_info_callback, qos_profile_sensor_data)
        
        self.angle_pub = self.create_publisher(Float32, '/patient_shoulder_angle', 10)
        
        # 우측 팔꿈치 좌표 발행
        self.right_elbow_3d_pub = self.create_publisher(Point, '/right_elbow_3d', 10)
        
        self.fixed_raw = None
        self.robot_raw = None
        self.depth_frame = None
        self.intrinsics = None
        
        self.get_logger().info("YOLOv11-Pose 모델을 로드 중입니다...")
        self.pose_model = YOLO('yolo11n-pose.pt') 

        self.analyzers = {
            'lateral_raise': LateralRaiseAnalyzer()    
        }
        self.current_analyzer = self.analyzers.get(exercise_type, LateralRaiseAnalyzer())

        self.get_logger().info(f" [{exercise_type}] 정면/우측팔 카메라 트레이닝 모드 시작! (데이터 저장 경로: {SAVE_DIR})")
        self.get_logger().info(" 창 클릭 후 [스페이스바]를 누르면 우측 팔꿈치 3D 좌표가 발행됩니다!")

        self.timer = self.create_timer(0.033, self.display_timer_callback)

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {"fx": msg.k[0], "fy": msg.k[4], "ppx": msg.k[2], "ppy": msg.k[5]}

    def fixed_callback(self, msg):
        try:
            self.fixed_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            pass 

    def robot_callback(self, msg):
        try:
            self.robot_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            pass 

    def _pixel_to_camera_coords(self, x, y, z):
        fx, fy, ppx, ppy = self.intrinsics['fx'], self.intrinsics['fy'], self.intrinsics['ppx'], self.intrinsics['ppy']
        return ((x - ppx) * z / fx, (y - ppy) * z / fy, z)

    # def display_timer_callback(self):
    #     if self.fixed_raw is not None and self.robot_raw is not None:
    #         fixed_img = cv2.resize(self.fixed_raw, (640, 480))
    #         robot_img = cv2.resize(self.robot_raw, (640, 480))

    #         res_fixed = self.pose_model(fixed_img, verbose=False)[0]
    #         res_robot = self.pose_model(robot_img, verbose=False)[0]

    #         right_elbow_3d_coord = None
    #         r_cz_val = None

    #         if res_fixed.keypoints is not None and len(res_fixed.keypoints.data) > 0 and \
    #            res_robot.keypoints is not None and len(res_robot.keypoints.data) > 0:
                
    #             fixed_kpts = res_fixed.keypoints.data[0].cpu().numpy()
    #             robot_kpts = res_robot.keypoints.data[0].cpu().numpy()

    #             target_angle, feedback, r_el_pt = self.current_analyzer.analyze_dual(
    #                 fixed_kpts, robot_kpts, fixed_img, robot_img
    #             )
                
    #             angle_msg = Float32()
    #             angle_msg.data = float(target_angle)
    #             self.angle_pub.publish(angle_msg)

    #             # 로봇 카메라에서 얻어온 우측 팔꿈치 2D좌표에 Depth 적용
    #             if self.intrinsics is not None and self.depth_frame is not None:
    #                 depth_resized = cv2.resize(self.depth_frame, (640, 480), interpolation=cv2.INTER_NEAREST)

    #                 if 0 <= r_el_pt[0] < 640 and 0 <= r_el_pt[1] < 480:
    #                     r_cz = float(depth_resized[r_el_pt[1], r_el_pt[0]])
    #                     if r_cz > 0:
    #                         r_cz_val = r_cz 
    #                         r_cx, r_cy, r_cz = self._pixel_to_camera_coords(r_el_pt[0], r_el_pt[1], r_cz)
    #                         right_elbow_3d_coord = (r_cx, r_cy, r_cz)
    #                         cv2.putText(robot_img, f"R-Elbow 3D Z:{int(r_cz)}", (r_el_pt[0] - 40, r_el_pt[1] + 30), 
    #                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
    #             self.current_analyzer.logger.update_depth(r_cz_val)

    #         else:
    #             cv2.putText(fixed_img, "Waiting for detection...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    #         combined_image = np.hstack((fixed_img, robot_img))
    #         cv2.imshow('Dual View PT Trainer - [Full Body | Robot Right Arm]', combined_image)
            
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == 32: # 스페이스바 입력
    #             if right_elbow_3d_coord is not None:
    #                 r_msg = Point(x=right_elbow_3d_coord[0], y=right_elbow_3d_coord[1], z=right_elbow_3d_coord[2])
    #                 self.right_elbow_3d_pub.publish(r_msg) # 퍼블리셔 이름 변경
    #                 self.get_logger().info(f"✅ [우측 팔꿈치] 목표 3D 발행: X:{int(r_msg.x)}, Y:{int(r_msg.y)}, Z:{int(r_msg.z)}")

    def display_timer_callback(self):
            if self.fixed_raw is not None and self.robot_raw is not None:
                # 해상도 640x480 강제 동기화 (YOLO 연산용 화면)
                fixed_img = cv2.resize(self.fixed_raw, (640, 480))
                robot_img = cv2.resize(self.robot_raw, (640, 480))

                res_fixed = self.pose_model(fixed_img, verbose=False)[0]
                res_robot = self.pose_model(robot_img, verbose=False)[0]

                right_elbow_3d_coord = None
                r_cz_val = None

                if res_fixed.keypoints is not None and len(res_fixed.keypoints.data) > 0 and \
                res_robot.keypoints is not None and len(res_robot.keypoints.data) > 0:
                    
                    fixed_kpts = res_fixed.keypoints.data[0].cpu().numpy()
                    robot_kpts = res_robot.keypoints.data[0].cpu().numpy()

                    target_angle, feedback, r_el_pt = self.current_analyzer.analyze_dual(
                        fixed_kpts, robot_kpts, fixed_img, robot_img
                    )
                    
                    angle_msg = Float32()
                    angle_msg.data = float(target_angle)
                    self.angle_pub.publish(angle_msg)

                    # ----------------------------------------------------
                    # [수정됨] YOLO 640x480 좌표를 원본 해상도(예: 1280x720)로 
                    # 복원하여 오차 없는 진짜 3D 좌표(mm) 추출
                    # ----------------------------------------------------
                    if self.intrinsics is not None and self.depth_frame is not None:
                        # 1. 로봇 카메라 원본 이미지의 가로, 세로 크기 추출
                        orig_h, orig_w = self.robot_raw.shape[:2]
                        
                        # 2. 640x480 비율만큼 줄였던 것을 다시 확대하는 스케일 계산
                        scale_x = orig_w / 640.0
                        scale_y = orig_h / 480.0
                        
                        # 3. YOLO가 찾은 좌표(r_el_pt)를 원본 1280x720 좌표계로 변환
                        orig_x = int(r_el_pt[0] * scale_x)
                        orig_y = int(r_el_pt[1] * scale_y)

                        # 4. 리사이즈 하지 않은 "원본 Depth 프레임"에서 깊이값(Z) 추출
                        if 0 <= orig_x < orig_w and 0 <= orig_y < orig_h:
                            r_cz = float(self.depth_frame[orig_y, orig_x])
                            if r_cz > 0:
                                r_cz_val = r_cz 
                                
                                # 5. 카메라 내부 파라미터 공식에 "원본 픽셀 위치(orig_x, orig_y)"를 넣어서 실제 3D 계산
                                r_cx, r_cy, r_cz = self._pixel_to_camera_coords(orig_x, orig_y, r_cz)
                                right_elbow_3d_coord = (r_cx, r_cy, r_cz)
                                
                                # 화면에 글씨 띄우는 건 640x480 이미지에 하는 거라 원래 좌표(r_el_pt) 사용
                                cv2.putText(robot_img, f"R-Elbow 3D Z:{int(r_cz)}", (r_el_pt[0] - 40, r_el_pt[1] + 30), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    self.current_analyzer.logger.update_depth(r_cz_val)

                else:
                    cv2.putText(fixed_img, "Waiting for detection...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                combined_image = np.hstack((fixed_img, robot_img))
                cv2.imshow('Dual View PT Trainer - [Full Body | Robot Right Arm]', combined_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 32: # 스페이스바 입력
                    if right_elbow_3d_coord is not None:
                        r_msg = Point(x=right_elbow_3d_coord[0], y=right_elbow_3d_coord[1], z=right_elbow_3d_coord[2])
                        self.right_elbow_3d_pub.publish(r_msg) # 퍼블리셔 이름 변경
                        self.get_logger().info(f"✅ [우측 팔꿈치] 목표 3D 발행: X:{int(r_msg.x)}, Y:{int(r_msg.y)}, Z:{int(r_msg.z)}")


def main(args=None):
    rclpy.init(args=args)
    node = PoseTrackingNode(exercise_type='lateral_raise') 
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()