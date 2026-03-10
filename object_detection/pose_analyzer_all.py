import os
import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from std_srvs.srv import SetBool, Trigger

# ==========================================
# 1. 운동 세션 트래커 모듈 (JSON 로컬 저장 제거, Topic 전용)
# ==========================================
class BaseTracker:
    def __init__(self, exercise_type, publish_callback):
        self.exercise_type = exercise_type
        self.publish_callback = publish_callback # 메인 노드의 퍼블리셔 콜백
        self.reset()

    def reset(self):
        now = datetime.now()
        self.session_started_at = now.strftime("%Y-%m-%d %H:%M:%S")
        self.last_updated_at = self.session_started_at
        self.rep_count = 0
        self.frame_count = 0
        self.good_frame_count = 0
        self.last_feedback = "No data yet"
        self._reset_specific()

    def _reset_specific(self):
        pass

    def emit_data(self, data):
        """데이터를 파이썬 딕셔너리로 받아 메인 노드로 전달"""
        if self.publish_callback:
            self.publish_callback(data)


class SPTracker(BaseTracker):
    """숄더 프레스 전용 트래커"""
    def _reset_specific(self):
        self.elbow_angle_sum = 0.0
        self.shoulder_angle_sum = 0.0
        self.trunk_angle_sum = 0.0
        self.warning_counts = {
            "body_not_straight": 0, "arm_balance_issue": 0,
            "too_low": 0, "bend_elbows_at_bottom": 0,
        }
        self.build_and_emit()

    def update_frame(self, elbow_angle, shoulder_angle, trunk_angle, feedback, is_correct):
        self.frame_count += 1
        if is_correct: self.good_frame_count += 1

        self.elbow_angle_sum += float(elbow_angle)
        self.shoulder_angle_sum += float(shoulder_angle)
        self.trunk_angle_sum += float(trunk_angle)
        self.last_feedback = feedback
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if feedback == "Warning: Keep your body straight!": self.warning_counts["body_not_straight"] += 1
        elif feedback == "Warning: Balance your arms!": self.warning_counts["arm_balance_issue"] += 1
        elif feedback == "Warning: Don't go too low!": self.warning_counts["too_low"] += 1
        elif feedback == "Warning: Bend elbows at bottom!": self.warning_counts["bend_elbows_at_bottom"] += 1

        # 15프레임마다 DB 연동용 토픽 발생
        if self.frame_count % 15 == 0:
            self.build_and_emit()

    def increment_rep(self, rep_count):
        self.rep_count = int(rep_count)
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.build_and_emit()

    def build_and_emit(self):
        good_ratio = round((self.good_frame_count / self.frame_count) * 100.0, 2) if self.frame_count > 0 else 0.0
        avg = lambda x: round(x / self.frame_count, 2) if self.frame_count > 0 else 0.0
        data = {
            "exercise_type": self.exercise_type,
            "session_started_at": self.session_started_at,
            "last_updated_at": self.last_updated_at,
            "rep_count": self.rep_count,
            "frame_count": self.frame_count,
            "good_frame_count": self.good_frame_count,
            "good_posture_ratio": good_ratio,
            "avg_elbow_angle": avg(self.elbow_angle_sum),
            "avg_shoulder_angle": avg(self.shoulder_angle_sum),
            "avg_trunk_angle": avg(self.trunk_angle_sum),
            "warning_counts": self.warning_counts,
            "last_feedback": self.last_feedback,
        }
        self.emit_data(data)


class BCTracker(BaseTracker):
    """이두 컬 전용 트래커"""
    def _reset_specific(self):
        self.analyzed_frame_count = 0
        self.ignored_frame_count = 0
        self.elbow_angle_sum = 0.0
        self.upper_arm_angle_sum = 0.0
        self.trunk_angle_sum = 0.0
        self.warning_counts = {
            "body_not_straight": 0, "arm_balance_issue": 0,
            "elbows_not_close_to_body": 0, "arms_not_visible": 0,
        }
        self.build_and_emit()

    def update_frame(self, elbow_angle=None, upper_arm_angle=None, trunk_angle=None, feedback="No data yet", is_correct=False, has_valid_measurement=False, count_warning=False):
        self.frame_count += 1
        self.last_feedback = feedback
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if has_valid_measurement:
            self.analyzed_frame_count += 1
            self.elbow_angle_sum += float(elbow_angle)
            self.upper_arm_angle_sum += float(upper_arm_angle)
            self.trunk_angle_sum += float(trunk_angle)
            if is_correct: self.good_frame_count += 1
        else:
            self.ignored_frame_count += 1

        if count_warning:
            if feedback == "Warning: Keep your body straight!": self.warning_counts["body_not_straight"] += 1
            elif feedback == "Warning: Move both arms evenly!": self.warning_counts["arm_balance_issue"] += 1
            elif feedback == "Warning: Keep elbows close to body!": self.warning_counts["elbows_not_close_to_body"] += 1
            elif feedback == "Warning: Keep both arms visible!": self.warning_counts["arms_not_visible"] += 1

        if self.frame_count % 15 == 0:
            self.build_and_emit()

    def increment_rep(self, rep_count):
        self.rep_count = int(rep_count)
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.build_and_emit()

    def build_and_emit(self):
        good_ratio = round((self.good_frame_count / self.analyzed_frame_count) * 100.0, 2) if self.analyzed_frame_count > 0 else 0.0
        avg = lambda x: round(x / self.analyzed_frame_count, 2) if self.analyzed_frame_count > 0 else 0.0
        data = {
            "exercise_type": self.exercise_type,
            "session_started_at": self.session_started_at,
            "last_updated_at": self.last_updated_at,
            "rep_count": self.rep_count,
            "frame_count": self.frame_count,
            "analyzed_frame_count": self.analyzed_frame_count,
            "ignored_frame_count": self.ignored_frame_count,
            "good_posture_ratio": good_ratio,
            "avg_elbow_angle": avg(self.elbow_angle_sum),
            "avg_upper_arm_angle": avg(self.upper_arm_angle_sum),
            "avg_trunk_angle": avg(self.trunk_angle_sum),
            "warning_counts": self.warning_counts,
            "last_feedback": self.last_feedback,
        }
        self.emit_data(data)


class LRTracker(BaseTracker):
    """사이드 레터럴 레이즈 전용 트래커"""
    def _reset_specific(self):
        self.session_start_dt = datetime.now()
        self.trunk_angle_sum = 0.0
        self.max_rom_left = 0.0   
        self.max_rom_right = 0.0  
        self.rep_start_time = None
        self.rep_durations = []   
        self.tremor_count = 0     
        self.last_l_wr_y = None   
        self.last_r_wr_y = None
        self.successful_peaks = [] 
        self.all_peaks = []        
        self.r_z_history = []
        self.max_z_drift = 0.0
        self.pure_arom = 0.0
        self.warning_counts = {
            "lean_back_momentum": 0, "chest_down": 0,
            "arms_too_high": 0, "arm_balance_issue": 0,
        }
        self.build_and_emit()

    def update_frame(self, l_shoulder_angle, r_shoulder_angle, trunk_angle, feedback, is_correct, l_wr_y=None, r_wr_y=None):
        self.frame_count += 1
        if is_correct:
            self.good_frame_count += 1
            current_max_shoulder = max(l_shoulder_angle, r_shoulder_angle)
            if current_max_shoulder > self.pure_arom:
                self.pure_arom = round(float(current_max_shoulder), 2)

        self.trunk_angle_sum += float(trunk_angle)

        if l_shoulder_angle > self.max_rom_left: self.max_rom_left = round(float(l_shoulder_angle), 2)
        if r_shoulder_angle > self.max_rom_right: self.max_rom_right = round(float(r_shoulder_angle), 2)

        if l_wr_y is not None and r_wr_y is not None:
            if self.last_l_wr_y is not None:
                if abs(l_wr_y - self.last_l_wr_y) > 15 or abs(r_wr_y - self.last_r_wr_y) > 15:
                    if l_shoulder_angle > 30 or r_shoulder_angle > 30:
                        self.tremor_count += 1
            self.last_l_wr_y = l_wr_y
            self.last_r_wr_y = r_wr_y

        self.last_feedback = feedback
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if feedback == "Warning: Keep your body straight!": self.warning_counts["lean_back_momentum"] += 1
        elif feedback == "Warning: Arms too high! Lower them.": self.warning_counts["arms_too_high"] += 1
        elif feedback == "Warning: Balance your arms!": self.warning_counts["arm_balance_issue"] += 1

        if self.frame_count % 15 == 0:
            self.build_and_emit()

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
            if r_drift > self.max_z_drift: self.max_z_drift = round(r_drift, 2)
        
        self.r_z_history = []
        self.last_updated_at = now.strftime("%Y-%m-%d %H:%M:%S")
        self.build_and_emit()

    def build_and_emit(self):
        good_ratio = round((self.good_frame_count / self.frame_count) * 100.0, 2) if self.frame_count > 0 else 0.0
        avg_trunk = round(self.trunk_angle_sum / self.frame_count, 2) if self.frame_count > 0 else 0.0
        session_duration = round((datetime.now() - self.session_start_dt).total_seconds(), 2)
        avg_rep_dur = round(sum(self.rep_durations) / len(self.rep_durations), 2) if self.rep_durations else 0.0

        avg_successful_peak = round(sum(self.successful_peaks) / len(self.successful_peaks), 2) if self.successful_peaks else 0.0
        avg_all_peak = round(sum(self.all_peaks) / len(self.all_peaks), 2) if self.all_peaks else 0.0
        assist_trigger_angle = max(0.0, self.pure_arom - 5.0) if self.pure_arom > 0 else 0.0
        target_prom = min(90.0, self.pure_arom + 10.0) if self.pure_arom > 0 else 0.0

        data = {
            "exercise_type": self.exercise_type,
            "session_started_at": self.session_started_at,
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
                "good_posture_ratio": good_ratio,
                "avg_trunk_angle": avg_trunk
            },
            "warning_counts": self.warning_counts,
            "last_feedback": self.last_feedback,
        }
        self.emit_data(data)


# ==========================================
# 2. 운동 전략 패턴 분석기 모듈 (Analyzer)
# ==========================================
class ExerciseAnalyzer:
    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0: angle = 360 - angle
        return angle

    def analyze(self, node):
        raise NotImplementedError

class ShoulderPressAnalyzer(ExerciseAnalyzer):
    def __init__(self, publish_callback):
        self.count = 0
        self.state = "UP"
        self.tracker = SPTracker("shoulder_press", publish_callback)

    def analyze(self, node):
        if node.main_raw is None: return None
        image = node.main_raw.copy()
        h, w, _ = image.shape
        results = node.model(image, verbose=False, device='cpu')[0]

        if results.keypoints is None or len(results.keypoints.xyn) == 0: return None
        kpts = results.keypoints.xyn[0].cpu().numpy()
        if len(kpts) < 13: return None

        nose = [kpts[0][0], kpts[0][1]]
        l_sh, l_el, l_wr, l_hip = [kpts[5][0], kpts[5][1]], [kpts[7][0], kpts[7][1]], [kpts[9][0], kpts[9][1]], [kpts[11][0], kpts[11][1]]
        r_sh, r_el, r_wr, r_hip = [kpts[6][0], kpts[6][1]], [kpts[8][0], kpts[8][1]], [kpts[10][0], kpts[10][1]], [kpts[12][0], kpts[12][1]]

        l_pts = [(int(l_sh[0]*w), int(l_sh[1]*h)), (int(l_el[0]*w), int(l_el[1]*h)), (int(l_wr[0]*w), int(l_wr[1]*h)), (int(l_hip[0]*w), int(l_hip[1]*h))]
        r_pts = [(int(r_sh[0]*w), int(r_sh[1]*h)), (int(r_el[0]*w), int(r_el[1]*h)), (int(r_wr[0]*w), int(r_wr[1]*h)), (int(r_hip[0]*w), int(r_hip[1]*h))]
        nose_pt = (int(nose[0]*w), int(nose[1]*h))

        # 시각화 뼈대
        cv2.line(image, l_pts[3], l_pts[0], (0, 255, 0), 3)
        cv2.line(image, l_pts[0], l_pts[1], (0, 255, 0), 3)
        cv2.line(image, l_pts[1], l_pts[2], (0, 255, 0), 3)
        cv2.line(image, r_pts[3], r_pts[0], (255, 0, 0), 3)
        cv2.line(image, r_pts[0], r_pts[1], (255, 0, 0), 3)
        cv2.line(image, r_pts[1], r_pts[2], (255, 0, 0), 3)
        for pt in l_pts + r_pts + [nose_pt]: cv2.circle(image, pt, 8, (0, 0, 255), -1)

        l_elbow_angle = self.calculate_angle(l_sh, l_el, l_wr)
        r_elbow_angle = self.calculate_angle(r_sh, r_el, r_wr)
        l_shoulder_angle = self.calculate_angle(l_hip, l_sh, l_el)
        r_shoulder_angle = self.calculate_angle(r_hip, r_sh, r_el)

        mid_hip = [(l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2]
        vertical_ref = [mid_hip[0], mid_hip[1]-0.1]
        trunk_angle = self.calculate_angle(vertical_ref, mid_hip, nose)
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0
        avg_shoulder_angle = (l_shoulder_angle + r_shoulder_angle) / 2.0

        is_correct_posture, feedback, color = True, "Good Form!", (0, 255, 0)
        if trunk_angle > 15: feedback, color, is_correct_posture = "Warning: Keep your body straight!", (0, 165, 255), False
        elif abs(l_elbow_angle - r_elbow_angle) > 30: feedback, color, is_correct_posture = "Warning: Balance your arms!", (0, 0, 255), False
        elif l_shoulder_angle < 70 or r_shoulder_angle < 70: feedback, color, is_correct_posture = "Warning: Don't go too low!", (0, 0, 255), False
        elif (l_shoulder_angle < 120 and l_elbow_angle > 140) or (r_shoulder_angle < 120 and r_elbow_angle > 140):
            feedback, color, is_correct_posture = "Warning: Bend elbows at bottom!", (0, 0, 255), False

        if is_correct_posture:
            is_down_pose = (70 <= l_shoulder_angle <= 120) and (70 <= l_elbow_angle <= 120) and (70 <= r_shoulder_angle <= 120) and (70 <= r_elbow_angle <= 120)
            is_up_pose = (l_shoulder_angle >= 150 and l_elbow_angle >= 150 and r_shoulder_angle >= 150 and r_elbow_angle >= 150)
            if is_down_pose:
                if self.state == "UP": self.state = "DOWN"
                feedback, color = "Ready... Push UP!", (0, 255, 255)
            elif is_up_pose:
                if self.state == "DOWN":
                    self.state = "UP"
                    self.count += 1
                    self.tracker.increment_rep(self.count)
                feedback, color = "Perfect! Keep going!", (255, 0, 0)

        self.tracker.update_frame(avg_elbow_angle, avg_shoulder_angle, trunk_angle, feedback, is_correct_posture)

        cv2.putText(image, feedback, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f"L Elbow: {int(l_elbow_angle)} | L Shld: {int(l_shoulder_angle)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"R Elbow: {int(r_elbow_angle)} | R Shld: {int(r_shoulder_angle)}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, f"Count: {self.count}", (w - 280, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        target_pixel = (int((l_wr[0]*w + r_wr[0]*w)/2), int((l_wr[1]*h + r_wr[1]*h)/2))
        cv2.circle(image, target_pixel, 10, (0, 255, 255), -1)

        return {
            "angle_val": avg_elbow_angle, "angle_topic": "/patient_elbow_angle",
            "target_pixel": target_pixel, "camera": "main", "point_topic": "/target_correction_3d",
            "display_img": image
        }


class BicepCurlAnalyzer(ExerciseAnalyzer):
    def __init__(self, publish_callback):
        self.count = 0
        self.tracker = BCTracker("bicep_curl", publish_callback)
        self.TRUNK_THRESHOLD = 15
        self.BALANCE_THRESHOLD = 35
        self.UPPER_ARM_THRESHOLD = 85
        self.DOWN_ELBOW_ANGLE = 160
        self.UP_ELBOW_ANGLE = 60
        self.DOWN_CONFIRM_FRAMES = 3
        self.UP_CONFIRM_FRAMES = 2
        self.confirmed_pose_state = "DOWN"
        self.down_pose_streak = 0
        self.up_pose_streak = 0

    def analyze(self, node):
        if node.main_raw is None: return None
        image = node.main_raw.copy()
        h, w, _ = image.shape
        results = node.model(image, verbose=False, device='cpu')[0]
        
        if results.keypoints is None or len(results.keypoints.xyn) == 0: return None
        kpts = results.keypoints.xyn[0].cpu().numpy()
        kpt_conf = results.keypoints.conf[0].cpu().numpy() if hasattr(results.keypoints, "conf") and results.keypoints.conf is not None else None
        
        visible_now = all((float(kpt_conf[i]) >= 0.35 if kpt_conf is not None and i < len(kpt_conf) else True) for i in [0,5,6,7,8,9,10,11,12])

        if not visible_now or len(kpts) < 13:
            feedback, color = "Warning: Keep both arms visible!", (0, 0, 255)
            self.tracker.update_frame(feedback=feedback, is_correct=False, has_valid_measurement=False, count_warning=True)
            cv2.putText(image, feedback, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            return {"angle_val": 0.0, "angle_topic": "/patient_elbow_angle", "target_pixel": None, "camera": "main", "point_topic": "/target_correction_3d", "display_img": image}

        nose = [kpts[0][0], kpts[0][1]]
        l_sh, l_el, l_wr, l_hip = [kpts[5][0], kpts[5][1]], [kpts[7][0], kpts[7][1]], [kpts[9][0], kpts[9][1]], [kpts[11][0], kpts[11][1]]
        r_sh, r_el, r_wr, r_hip = [kpts[6][0], kpts[6][1]], [kpts[8][0], kpts[8][1]], [kpts[10][0], kpts[10][1]], [kpts[12][0], kpts[12][1]]

        l_pts = [(int(l_sh[0]*w), int(l_sh[1]*h)), (int(l_el[0]*w), int(l_el[1]*h)), (int(l_wr[0]*w), int(l_wr[1]*h)), (int(l_hip[0]*w), int(l_hip[1]*h))]
        r_pts = [(int(r_sh[0]*w), int(r_sh[1]*h)), (int(r_el[0]*w), int(r_el[1]*h)), (int(r_wr[0]*w), int(r_wr[1]*h)), (int(r_hip[0]*w), int(r_hip[1]*h))]
        
        cv2.line(image, l_pts[3], l_pts[0], (0, 255, 0), 3)
        cv2.line(image, l_pts[0], l_pts[1], (0, 255, 0), 3)
        cv2.line(image, l_pts[1], l_pts[2], (0, 255, 0), 3)
        cv2.line(image, r_pts[3], r_pts[0], (255, 0, 0), 3)
        cv2.line(image, r_pts[0], r_pts[1], (255, 0, 0), 3)
        cv2.line(image, r_pts[1], r_pts[2], (255, 0, 0), 3)

        target_pixel = (int((l_pts[2][0] + r_pts[2][0])/2), int((l_pts[2][1] + r_pts[2][1])/2))
        cv2.circle(image, target_pixel, 10, (0, 255, 255), -1)

        l_elbow_angle = self.calculate_angle(l_sh, l_el, l_wr)
        r_elbow_angle = self.calculate_angle(r_sh, r_el, r_wr)
        l_upper_arm_angle = self.calculate_angle(l_hip, l_sh, l_el)
        r_upper_arm_angle = self.calculate_angle(r_hip, r_sh, r_el)

        mid_hip = [(l_hip[0]+r_hip[0])/2.0, (l_hip[1]+r_hip[1])/2.0]
        vertical_ref = [mid_hip[0], mid_hip[1]-0.1]
        trunk_angle = self.calculate_angle(vertical_ref, mid_hip, nose)
        avg_elbow_angle, avg_upper_arm_angle = (l_elbow_angle + r_elbow_angle) / 2.0, (l_upper_arm_angle + r_upper_arm_angle) / 2.0

        is_correct_posture, feedback, color = True, "Good Form!", (0, 255, 0)
        if trunk_angle > self.TRUNK_THRESHOLD: feedback, color, is_correct_posture = "Warning: Keep your body straight!", (0, 165, 255), False
        elif abs(l_elbow_angle - r_elbow_angle) > self.BALANCE_THRESHOLD: feedback, color, is_correct_posture = "Warning: Move both arms evenly!", (0, 0, 255), False
        elif l_upper_arm_angle > self.UPPER_ARM_THRESHOLD or r_upper_arm_angle > self.UPPER_ARM_THRESHOLD: feedback, color, is_correct_posture = "Warning: Keep elbows close to body!", (0, 0, 255), False

        if is_correct_posture:
            if l_elbow_angle >= self.DOWN_ELBOW_ANGLE and r_elbow_angle >= self.DOWN_ELBOW_ANGLE and l_upper_arm_angle <= self.UPPER_ARM_THRESHOLD and r_upper_arm_angle <= self.UPPER_ARM_THRESHOLD:
                self.down_pose_streak += 1; self.up_pose_streak = 0
                if self.down_pose_streak >= self.DOWN_CONFIRM_FRAMES:
                    if self.confirmed_pose_state == "UP":
                        self.count += 1
                        self.tracker.increment_rep(self.count)
                    self.confirmed_pose_state = "DOWN"
                    feedback, color = "Ready... Curl UP!", (0, 255, 255)
            elif l_elbow_angle <= self.UP_ELBOW_ANGLE and r_elbow_angle <= self.UP_ELBOW_ANGLE and l_upper_arm_angle <= self.UPPER_ARM_THRESHOLD + 20 and r_upper_arm_angle <= self.UPPER_ARM_THRESHOLD + 20:
                self.up_pose_streak += 1; self.down_pose_streak = 0
                if self.up_pose_streak >= self.UP_CONFIRM_FRAMES:
                    self.confirmed_pose_state = "UP"
                    feedback, color = "Great Curl! Slowly lower your arms!", (255, 0, 0)
            else:
                self.down_pose_streak = 0; self.up_pose_streak = 0
        else:
            self.down_pose_streak = 0; self.up_pose_streak = 0

        self.tracker.update_frame(elbow_angle=avg_elbow_angle, upper_arm_angle=avg_upper_arm_angle, trunk_angle=trunk_angle, feedback=feedback, is_correct=is_correct_posture, has_valid_measurement=True, count_warning=feedback.startswith("Warning:"))
        
        cv2.putText(image, feedback, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f"Count: {self.count}", (w - 280, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        return {
            "angle_val": avg_elbow_angle, "angle_topic": "/patient_elbow_angle",
            "target_pixel": target_pixel, "camera": "main", "point_topic": "/target_correction_3d",
            "display_img": image
        }


class LateralRaiseAnalyzer(ExerciseAnalyzer):
    def __init__(self, publish_callback):
        self.count = 0
        self.state = "DOWN"
        self.current_rep_peak = 0.0
        self.eval_feedback = ""
        self.eval_color = (0, 255, 0)
        self.rep_has_warning = False
        self.tracker = LRTracker("lateral_raise", publish_callback)

    def analyze(self, node):
        if node.fixed_raw is None or node.robot_raw is None: return None
        
        # 듀얼 카메라 처리
        fixed_img = cv2.resize(node.fixed_raw, (640, 480))
        robot_img = cv2.resize(node.robot_raw, (640, 480))

        res_fixed = node.model(fixed_img, verbose=False, device='cpu')[0]
        res_robot = node.model(robot_img, verbose=False, device='cpu')[0]

        if res_fixed.keypoints is None or len(res_fixed.keypoints.xyn) == 0: return None
        if res_robot.keypoints is None or len(res_robot.keypoints.xyn) == 0: return None

        fixed_kpts = res_fixed.keypoints.xyn[0].cpu().numpy()
        robot_kpts = res_robot.keypoints.xyn[0].cpu().numpy()
        if len(fixed_kpts) < 13 or len(robot_kpts) < 9: return None

        w, h = 640, 480
        nose = fixed_kpts[0][:2]
        l_sh_f, l_el_f, l_wr_f, l_hip_f = fixed_kpts[5][:2], fixed_kpts[7][:2], fixed_kpts[9][:2], fixed_kpts[11][:2]
        r_sh_f, r_el_f, r_wr_f, r_hip_f = fixed_kpts[6][:2], fixed_kpts[8][:2], fixed_kpts[10][:2], fixed_kpts[12][:2]

        pts_f = {
            'l_sh': (int(l_sh_f[0]*w), int(l_sh_f[1]*h)), 'l_el': (int(l_el_f[0]*w), int(l_el_f[1]*h)),
            'l_hip': (int(l_hip_f[0]*w), int(l_hip_f[1]*h)), 'r_sh': (int(r_sh_f[0]*w), int(r_sh_f[1]*h)),
            'r_el': (int(r_el_f[0]*w), int(r_el_f[1]*h)), 'r_hip': (int(r_hip_f[0]*w), int(r_hip_f[1]*h))
        }
        for pt in pts_f.values(): cv2.circle(fixed_img, pt, 8, (0, 0, 255), -1)

        l_shoulder_angle = self.calculate_angle(l_hip_f, l_sh_f, l_el_f)
        r_shoulder_angle = self.calculate_angle(r_hip_f, r_sh_f, r_el_f)
        avg_shoulder_angle = (l_shoulder_angle + r_shoulder_angle) / 2.0

        mid_hip = [(l_hip_f[0]+r_hip_f[0])/2, (l_hip_f[1]+r_hip_f[1])/2]
        vertical_ref = [mid_hip[0], mid_hip[1] - 0.1]
        trunk_front_angle = self.calculate_angle(vertical_ref, mid_hip, nose)

        is_correct_posture, feedback, color = True, "Good Form!", (0, 255, 0)
        if trunk_front_angle > 15: feedback, color, is_correct_posture = "Warning: Keep your body straight!", (0, 0, 255), False
        elif l_shoulder_angle > 100 or r_shoulder_angle > 100: feedback, color, is_correct_posture = "Warning: Arms too high! Lower them.", (0, 0, 255), False
        elif abs(l_shoulder_angle - r_shoulder_angle) > 20: feedback, color, is_correct_posture = "Warning: Balance your arms!", (0, 165, 255), False
        if not is_correct_posture: self.rep_has_warning = True

        if self.state == "DOWN":
            if avg_shoulder_angle >= 40:
                self.state, self.current_rep_peak, self.rep_has_warning = "RAISING", avg_shoulder_angle, False
                feedback, color = "Keep going up!", (0, 255, 0)
            else: feedback, color = "Ready... Raise your arms!", (0, 255, 255)
        elif self.state == "RAISING":
            if avg_shoulder_angle > self.current_rep_peak: self.current_rep_peak = avg_shoulder_angle
            if avg_shoulder_angle < self.current_rep_peak - 5.0:
                self.state = "LOWERING"
                if not self.rep_has_warning and self.current_rep_peak >= 70.0: self.tracker.successful_peaks.append(round(float(self.current_rep_peak), 2))
                if self.current_rep_peak >= 40.0: self.tracker.all_peaks.append(round(float(self.current_rep_peak), 2))
                if self.current_rep_peak >= 80:
                    self.count += 1
                    self.tracker.increment_rep(self.count)
                    self.eval_feedback, self.eval_color = "Perfect! Great job.", (0, 255, 0)
                else:
                    self.eval_feedback, self.eval_color = f"Good. Try {80-int(self.current_rep_peak)} deg more!", (0, 255, 255)
            elif self.current_rep_peak >= 80: feedback, color = "Perfect height! Now slowly down.", (255, 0, 0)
        elif self.state == "LOWERING":
            if (l_shoulder_angle < 40) and (r_shoulder_angle < 40): self.state, self.current_rep_peak, self.eval_feedback = "DOWN", 0.0, ""
            if self.eval_feedback != "" and is_correct_posture: feedback, color = self.eval_feedback, self.eval_color

        self.tracker.update_frame(l_shoulder_angle, r_shoulder_angle, trunk_front_angle, feedback, is_correct_posture, l_wr_f[1]*h, r_wr_f[1]*h)

        cv2.putText(fixed_img, feedback, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(fixed_img, f"Count: {self.count}", (w - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        r_el_robot = robot_kpts[8][:2]
        r_el_pt = (int(r_el_robot[0]*w), int(r_el_robot[1]*h))
        cv2.circle(robot_img, r_el_pt, 12, (0, 255, 255), -1)

        depth_x = int(r_el_robot[0] * node.robot_depth.shape[1]) if node.robot_depth is not None else -1
        depth_y = int(r_el_robot[1] * node.robot_depth.shape[0]) if node.robot_depth is not None else -1

        combined_image = np.hstack((fixed_img, robot_img))
        
        return {
            "angle_val": avg_shoulder_angle, "angle_topic": "/patient_shoulder_angle",
            "target_pixel": (depth_x, depth_y), "camera": "robot", "point_topic": "/right_elbow_3d",
            "display_img": combined_image
        }


# ==========================================
# 3. 통합 ROS2 메인 노드 (Main Manager)
# ==========================================
class PoseAnalyzerAllNode(Node):
    def __init__(self):
        super().__init__('pose_analyzer_all_node')
        self.bridge = CvBridge()
        self.is_exercising = False
        self.publish_trigger = False
        
        self.get_logger().info("⏳ YOLO 모델 로드 중...")
        self.model = YOLO('yolo11n-pose.pt') 

        # Publishers
        self.result_pub = self.create_publisher(String, '/exercise_result', 10)
        self.shoulder_angle_pub = self.create_publisher(Float32, '/patient_shoulder_angle', 10)
        self.elbow_angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        self.target_3d_pub = self.create_publisher(Point, '/target_correction_3d', 10)
        self.right_elbow_3d_pub = self.create_publisher(Point, '/right_elbow_3d', 10)

        # 전략 패턴(Strategy)
        self.analyzers = {
            'lateral_raise': LateralRaiseAnalyzer(self.publish_result_cb),
            'bicep_curl': BicepCurlAnalyzer(self.publish_result_cb),
            'shoulder_press': ShoulderPressAnalyzer(self.publish_result_cb)
        }
        self.current_exercise = 'shoulder_press' # 기본값
        self.current_analyzer = self.analyzers[self.current_exercise]

        # Subscribers
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.main_cam_cb, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.main_depth_cb, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.main_info_cb, 10)
        
        self.create_subscription(Image, '/fixed/camera/color/image_raw', self.fixed_cam_cb, 10)
        self.create_subscription(Image, '/robot/camera/color/image_raw', self.robot_cam_cb, 10)
        self.create_subscription(Image, '/robot/camera/aligned_depth_to_color/image_raw', self.robot_depth_cb, 10)
        self.create_subscription(CameraInfo, '/robot/camera/color/camera_info', self.robot_info_cb, 10)
        
        self.main_raw, self.main_depth, self.main_intrinsics = None, None, None
        self.fixed_raw, self.robot_raw, self.robot_depth, self.robot_intrinsics = None, None, None, None

        self.srv_set_exercise = self.create_service(SetBool, '/set_exercise_state', self.set_exercise_cb)
        self.srv_publish_3d = self.create_service(Trigger, '/publish_target_3d', self.publish_3d_cb)
        self.sub_mode = self.create_subscription(String, '/set_exercise_mode', self.set_mode_cb, 10)

        self.timer = self.create_timer(0.033, self.display_timer_callback)
        self.get_logger().info(f"🚀 [통합 버전] 노드 시작! (현재 모드: {self.current_exercise})")

    def publish_result_cb(self, data_dict):
        """딕셔너리를 받아서 JSON String으로 변환한 뒤 DB 연동용 토픽으로 날림"""
        msg = String()
        msg.data = json.dumps(data_dict, ensure_ascii=False)
        self.result_pub.publish(msg)

    def main_cam_cb(self, msg): self.main_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    def main_depth_cb(self, msg): self.main_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    def main_info_cb(self, msg): self.main_intrinsics = {"fx": msg.k[0], "fy": msg.k[4], "ppx": msg.k[2], "ppy": msg.k[5]}
    
    def fixed_cam_cb(self, msg): self.fixed_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    def robot_cam_cb(self, msg): self.robot_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    def robot_depth_cb(self, msg): self.robot_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    def robot_info_cb(self, msg): self.robot_intrinsics = {"fx": msg.k[0], "fy": msg.k[4], "ppx": msg.k[2], "ppy": msg.k[5]}

    def set_mode_cb(self, msg):
        new_mode = msg.data.lower()
        if new_mode in self.analyzers:
            self.current_exercise = new_mode
            self.current_analyzer = self.analyzers[self.current_exercise]
            self.current_analyzer.tracker.reset()
            self.get_logger().info(f"🔄 운동 모드가 [{new_mode}]로 변경되었습니다.")
        else:
            self.get_logger().warn(f"⚠️ 지원하지 않는 운동 모드입니다: {new_mode}")

    def set_exercise_cb(self, request, response):
        self.is_exercising = request.data
        if self.is_exercising:
            self.current_analyzer.tracker.reset() 
            msg = f"✅ [{self.current_exercise}] 운동 분석을 시작합니다."
        else:
            self.current_analyzer.tracker.build_and_emit() # 종료 시 최종 기록 한 번 더 발행
            msg = f"⏸️ 운동 분석 대기(IDLE) 전환."
        self.get_logger().info(msg)
        response.success = True
        response.message = msg
        return response

    def publish_3d_cb(self, request, response):
        self.publish_trigger = True
        response.success, response.message = True, "🎯 3D 교정 좌표 발행 요청 수신됨."
        return response

    def _publish_3d_point(self, px, py, depth_frame, intrinsics, pub):
        if px <= 0 or py <= 0 or depth_frame is None or intrinsics is None: return None
        h, w = depth_frame.shape
        if not (0 <= px < w and 0 <= py < h): return None
        cz = float(depth_frame[py, px])
        if cz > 0:
            cx, cy = (px - intrinsics['ppx']) * cz / intrinsics['fx'], (py - intrinsics['ppy']) * cz / intrinsics['fy']
            pub.publish(Point(x=float(cx), y=float(cy), z=float(cz)))
            return (cx, cy, cz)
        return None

    def display_timer_callback(self):
        if not self.is_exercising: return # IDLE 상태면 리소스 소모 방지

        res = self.current_analyzer.analyze(self)
        if not res: return
        
        cv2.imshow('Integrated PT Analyzer', res['display_img'])
        key = cv2.waitKey(1) & 0xFF
        if key == 32: self.publish_trigger = True 

        # 1. 각도 퍼블리시
        angle_msg = Float32(data=float(res['angle_val']))
        if res['angle_topic'] == '/patient_shoulder_angle': self.shoulder_angle_pub.publish(angle_msg)
        else: self.elbow_angle_pub.publish(angle_msg)

        # 2. 3D 좌표 퍼블리시
        if self.publish_trigger:
            if res['target_pixel'] is not None:
                px, py = res['target_pixel']
                d_frame = self.robot_depth if res['camera'] == 'robot' else self.main_depth
                intr_info = self.robot_intrinsics if res['camera'] == 'robot' else self.main_intrinsics
                pub_obj = self.right_elbow_3d_pub if res['point_topic'] == '/right_elbow_3d' else self.target_3d_pub
                
                res_3d = self._publish_3d_point(px, py, d_frame, intr_info, pub_obj)
                
                if res_3d and self.current_exercise == 'lateral_raise':
                    self.current_analyzer.tracker.update_depth(res_3d[2])

                if res_3d: self.get_logger().info(f"✅ 3D 좌표 발행 ({res['point_topic']}): X:{int(res_3d[0])} Y:{int(res_3d[1])} Z:{int(res_3d[2])}")
                else: self.get_logger().warn("⚠️ 유효한 Depth 3D 좌표를 추출할 수 없습니다.")
            self.publish_trigger = False


def main(args=None):
    rclpy.init(args=args)
    node = PoseAnalyzerAllNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()