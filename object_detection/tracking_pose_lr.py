import sys
import os
import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# [추가] 로그 파일 저장 경로
LOG_FILE = os.path.expanduser("~/exercise_session_log.json")

# ==========================================
# 0. 데이터 로깅 모듈 (Test 코드에서 이식됨)
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
        
        # ⭐️ 사레레(Lateral Raise) 전용 에러 항목으로 변경하여 기록
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
        # ⭐️ 사레레 코드의 피드백 문구와 정확히 매칭
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
# 1. 운동 분석기 모듈
# ==========================================
class ExerciseAnalyzer:
    """모든 운동 분석기가 상속받을 기본 뼈대 클래스"""
    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def analyze(self, landmarks, image):
        raise NotImplementedError

class LateralRaiseAnalyzer(ExerciseAnalyzer):
    """사이드 레터럴 레이즈(사레레) 분석기"""
    def __init__(self):
        self.count = 0         # 운동 횟수
        self.state = "DOWN"    # 현재 상태 (UP 또는 DOWN)
        # [추가] 로깅 객체 초기화
        self.logger = ExerciseSessionLogger(LOG_FILE, "lateral_raise")

    def analyze(self, landmarks, image):
        h, w, _ = image.shape

        # [정면 관절 추출] 11/12(어깨), 13/14(팔꿈치), 23/24(골반)
        l_sh = [landmarks[11].x, landmarks[11].y]
        l_el = [landmarks[13].x, landmarks[13].y]
        l_hip = [landmarks[23].x, landmarks[23].y]

        r_sh = [landmarks[12].x, landmarks[12].y]
        r_el = [landmarks[14].x, landmarks[14].y]
        r_hip = [landmarks[24].x, landmarks[24].y]

        # [측면 관절 추출] 25(왼쪽 무릎) - Realsense 사이드뷰 대응
        l_knee = [landmarks[25].x, landmarks[25].y]

        # 픽셀 좌표 변환
        pts = {
            'l_sh': (int(l_sh[0]*w), int(l_sh[1]*h)),
            'l_el': (int(l_el[0]*w), int(l_el[1]*h)),
            'l_hip': (int(l_hip[0]*w), int(l_hip[1]*h)),
            'r_sh': (int(r_sh[0]*w), int(r_sh[1]*h)),
            'r_el': (int(r_el[0]*w), int(r_el[1]*h)),
            'r_hip': (int(r_hip[0]*w), int(r_hip[1]*h)),
            'l_knee': (int(l_knee[0]*w), int(l_knee[1]*h))
        }

        # 시각화 (뼈대)
        cv2.line(image, pts['l_hip'], pts['l_sh'], (0, 255, 0), 3)
        cv2.line(image, pts['l_sh'], pts['l_el'], (0, 255, 0), 3)
        cv2.line(image, pts['r_hip'], pts['r_sh'], (255, 0, 0), 3)
        cv2.line(image, pts['r_sh'], pts['r_el'], (255, 0, 0), 3)
        # 측면 뷰 기준 라인 (골반-무릎)
        cv2.line(image, pts['l_hip'], pts['l_knee'], (0, 255, 255), 3)

        for pt in pts.values():
            cv2.circle(image, pt, 8, (0, 0, 255), -1)

        # 1. 정면 뷰 각도: 골반-어깨-팔꿈치 (사레레 가동범위)
        l_shoulder_angle = self.calculate_angle(l_hip, l_sh, l_el)
        r_shoulder_angle = self.calculate_angle(r_hip, r_sh, r_el)
        
        # [추가] 로깅용 평균 어깨 각도 계산
        avg_shoulder_angle = (l_shoulder_angle + r_shoulder_angle) / 2.0

        # 2. 측면 뷰 각도: 어깨-골반-무릎 (상체 기울기 및 반동 감지)
        trunk_side_angle = self.calculate_angle(l_sh, l_hip, l_knee)

        # ---------------- 상태 판별 및 카운팅 로직 ----------------
        is_correct_posture = True
        feedback = "Good Form!"
        color = (0, 255, 0) # 기본 초록색

        # [1] 에러 체크 (잘못된 자세 감지)
        if trunk_side_angle > 175: # 상체를 뒤로 젖히며 반동을 주는 경우 (치팅)
            feedback = "Warning: Don't lean back! No momentum."
            color = (0, 0, 255)
            is_correct_posture = False
        elif trunk_side_angle < 150: # 너무 많이 숙여서 후면 삼각근으로 빠지는 경우
            feedback = "Warning: Keep your chest up!"
            color = (0, 165, 255)
            is_correct_posture = False
        elif l_shoulder_angle > 100 or r_shoulder_angle > 100: # 100도 이상 거상 (어깨 충돌 위험)
            feedback = "Warning: Arms too high! Lower them."
            color = (0, 0, 255)
            is_correct_posture = False
        elif abs(l_shoulder_angle - r_shoulder_angle) > 20: # 좌우 밸런스 붕괴
            feedback = "Warning: Balance your arms!"
            color = (0, 165, 255)
            is_correct_posture = False

        # [2] 상태 전환 및 카운팅 (올바른 자세를 유지 중일 때만)
        if is_correct_posture:
            # DOWN 자세: 팔이 몸통 옆에 있는 상태 (40도 미만)
            is_down_pose = (l_shoulder_angle < 40) and (r_shoulder_angle < 40)
            
            # UP 자세: 세계적 기준인 80도~95도 사이 도달
            is_up_pose = (80 <= l_shoulder_angle <= 95) and (80 <= r_shoulder_angle <= 95)

            if is_down_pose:
                if self.state == "UP":
                    self.state = "DOWN"
                feedback = "Ready... Raise your arms!"
                color = (0, 255, 255) # 준비 (노란색)
            
            elif is_up_pose:
                if self.state == "DOWN":
                    self.state = "UP"
                    self.count += 1  # ⭐️ 제대로 된 타겟 지점 도달 시 1회 카운트
                    # [추가] 카운트가 올라갈 때 로거 업데이트
                    self.logger.increment_rep(self.count)
                feedback = "Perfect! Slowly lower arms."
                color = (255, 0, 0) # 타겟 도달 (파란색)
            
            elif self.state == "DOWN" and (40 <= l_shoulder_angle < 80):
                # 팔을 올리는 중이거나 가동범위가 부족한 경우
                feedback = "Raise a bit higher!"
                color = (0, 255, 0)

        # [추가] 매 프레임마다 데이터 로깅 (사레레는 팔꿈치 각도를 구하지 않으므로 0.0 전달)
        self.logger.update_frame(
            elbow_angle=0.0,
            shoulder_angle=avg_shoulder_angle,
            trunk_angle=trunk_side_angle,
            feedback=feedback,
            is_correct=is_correct_posture,
        )

        # 화면 텍스트 출력
        cv2.putText(image, feedback, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f"Front(L/R): {int(l_shoulder_angle)} / {int(r_shoulder_angle)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Side Trunk Angle: {int(trunk_side_angle)}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(image, f"Count: {self.count}", (w - 280, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        return avg_shoulder_angle, feedback

# ==========================================
# 2. ROS2 메인 노드
# ==========================================
class PoseTrackingNode(Node):
    # 인자 기본값을 lateral_raise로 변경
    def __init__(self, exercise_type='lateral_raise'): 
        super().__init__('pose_tracking_node')
        self.bridge = CvBridge()
        
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        
        # Tasks API 설정
        model_path = '/home/rokey/cobot_ws/src/cobot2_ws/rehab_assist_robot/object_detection/pose_landmarker_lite.task'
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # 딕셔너리에 사레레(lateral_raise) 분석기 추가
        self.analyzers = {
            'lateral_raise': LateralRaiseAnalyzer()    
        }
        self.current_analyzer = self.analyzers.get(exercise_type, LateralRaiseAnalyzer())

        self.get_logger().info(f"💪 [{exercise_type}] 트레이닝 모드로 비전 노드가 시작되었습니다.")
        # [추가] 로그 경로 출력
        self.get_logger().info(f"exercise log path: {LOG_FILE}")

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = self.detector.detect(mp_image)

            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                
                target_angle, feedback = self.current_analyzer.analyze(landmarks, image)
                
                angle_msg = Float32()
                angle_msg.data = float(target_angle)
                self.angle_pub.publish(angle_msg)

            cv2.imshow('MediaPipe PT Trainer', image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"에러 발생: {e}")

def main(args=None):
    rclpy.init(args=args)
    # 인자를 lateral_raise로 주어 사레레 모드로 실행되게 합니다.
    node = PoseTrackingNode(exercise_type='lateral_raise') 
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()