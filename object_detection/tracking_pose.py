import sys
import os
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

class ShoulderFlexionAnalyzer(ExerciseAnalyzer):
    """밴드 숄더플렉션 분석기 (측면 기준)"""
    def analyze(self, landmarks, image):
        # 기존 숄더플렉션 코드 동일하게 유지
        h, w, _ = image.shape
        shoulder = [landmarks[12].x, landmarks[12].y]
        elbow = [landmarks[14].x, landmarks[14].y]
        wrist = [landmarks[16].x, landmarks[16].y]
        hip = [landmarks[24].x, landmarks[24].y]

        pts = {
            'shoulder': (int(shoulder[0]*w), int(shoulder[1]*h)),
            'elbow': (int(elbow[0]*w), int(elbow[1]*h)),
            'wrist': (int(wrist[0]*w), int(wrist[1]*h)),
            'hip': (int(hip[0]*w), int(hip[1]*h))
        }

        cv2.line(image, pts['hip'], pts['shoulder'], (255, 0, 0), 3)
        cv2.line(image, pts['shoulder'], pts['elbow'], (0, 255, 0), 3)
        cv2.line(image, pts['elbow'], pts['wrist'], (0, 255, 0), 3)
        for pt in pts.values():
            cv2.circle(image, pt, 8, (0, 0, 255), -1)

        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        shoulder_angle = self.calculate_angle(hip, shoulder, elbow)
        vertical_ref = [hip[0], hip[1] - 0.1]
        trunk_angle = self.calculate_angle(vertical_ref, hip, shoulder)

        feedback = "Good Posture!"
        color = (0, 255, 0)
        
        if elbow_angle < 150:
            feedback = "Warning: Straighten your elbow!"
            color = (0, 0, 255)
        elif trunk_angle > 15:
            feedback = "Warning: Do not lean back!"
            color = (0, 165, 255)

        cv2.putText(image, feedback, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f"Shoulder Angle: {int(shoulder_angle)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return shoulder_angle, feedback

class ShoulderPressAnalyzer(ExerciseAnalyzer):
    """숄더 프레스 분석기 (정면 기준, 양팔 동시 측정 및 카운팅)"""
    def __init__(self):
        self.count = 0         # 운동 횟수
        self.state = "UP"      # 현재 상태 (UP 또는 DOWN)

    def analyze(self, landmarks, image):
        h, w, _ = image.shape
        
        # [추가] 얼굴(코) 랜드마크 추출 (0번)
        nose = [landmarks[0].x, landmarks[0].y]

        # 좌측 관절 추출 (11:어깨, 13:팔꿈치, 15:손목, 23:골반)
        l_sh = [landmarks[11].x, landmarks[11].y]
        l_el = [landmarks[13].x, landmarks[13].y]
        l_wr = [landmarks[15].x, landmarks[15].y]
        l_hip = [landmarks[23].x, landmarks[23].y]
        
        # 우측 관절 추출 (12:어깨, 14:팔꿈치, 16:손목, 24:골반)
        r_sh = [landmarks[12].x, landmarks[12].y]
        r_el = [landmarks[14].x, landmarks[14].y]
        r_wr = [landmarks[16].x, landmarks[16].y]
        r_hip = [landmarks[24].x, landmarks[24].y]

        # 픽셀 좌표 변환 및 시각화용 포인트
        l_pts = [(int(l_sh[0]*w), int(l_sh[1]*h)), (int(l_el[0]*w), int(l_el[1]*h)), 
                 (int(l_wr[0]*w), int(l_wr[1]*h)), (int(l_hip[0]*w), int(l_hip[1]*h))]
        r_pts = [(int(r_sh[0]*w), int(r_sh[1]*h)), (int(r_el[0]*w), int(r_el[1]*h)), 
                 (int(r_wr[0]*w), int(r_wr[1]*h)), (int(r_hip[0]*w), int(r_hip[1]*h))]
        nose_pt = (int(nose[0]*w), int(nose[1]*h))

        # 시각화 (뼈대)
        cv2.line(image, l_pts[3], l_pts[0], (0, 255, 0), 3)
        cv2.line(image, l_pts[0], l_pts[1], (0, 255, 0), 3)
        cv2.line(image, l_pts[1], l_pts[2], (0, 255, 0), 3)
        cv2.line(image, r_pts[3], r_pts[0], (255, 0, 0), 3)
        cv2.line(image, r_pts[0], r_pts[1], (255, 0, 0), 3)
        cv2.line(image, r_pts[1], r_pts[2], (255, 0, 0), 3)
        
        for pt in l_pts + r_pts + [nose_pt]:
            cv2.circle(image, pt, 8, (0, 0, 255), -1)

        # 1. 팔꿈치 각도 (손목-팔꿈치-어깨)
        l_elbow_angle = self.calculate_angle(l_sh, l_el, l_wr)
        r_elbow_angle = self.calculate_angle(r_sh, r_el, r_wr)

        # 2. 몸통 기준 어깨 각도 (골반-어깨-팔꿈치)
        l_shoulder_angle = self.calculate_angle(l_hip, l_sh, l_el)
        r_shoulder_angle = self.calculate_angle(r_hip, r_sh, r_el)

        # 3. [추가] 몸통(얼굴~골반) 휨 정도 계산
        # 양쪽 골반의 중앙점을 구하고, 코(Nose)와의 수직 정렬 상태를 확인
        mid_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
        vertical_ref = [mid_hip[0], mid_hip[1] - 0.1] # 골반에서 수직 위로 올린 가상의 점
        trunk_angle = self.calculate_angle(vertical_ref, mid_hip, nose)

        # ---------------- 상태 판별 및 카운팅 로직 ----------------
        is_correct_posture = True
        feedback = "Good Form!"
        color = (0, 255, 0) # 기본 초록색

        # [1] 에러 체크 (잘못된 자세 감지)
        if trunk_angle > 15: # 몸통이 15도 이상 좌우로 기울어짐
            feedback = "Warning: Keep your body straight!"
            color = (0, 165, 255)
            is_correct_posture = False
        elif abs(l_elbow_angle - r_elbow_angle) > 30: # 양팔 밸런스 무너짐
            feedback = "Warning: Balance your arms!"
            color = (0, 0, 255)
            is_correct_posture = False
        elif l_shoulder_angle < 70 or r_shoulder_angle < 70: # 너무 많이 내림
            feedback = "Warning: Don't go too low!"
            color = (0, 0, 255)
            is_correct_posture = False
        # [지적하신 T자 자세 에러] 팔 각도는 90도 부근인데 팔꿈치를 140도 이상 쭉 편 경우
        elif (l_shoulder_angle < 120 and l_elbow_angle > 140) or (r_shoulder_angle < 120 and r_elbow_angle > 140):
            feedback = "Warning: Bend elbows at bottom!"
            color = (0, 0, 255)
            is_correct_posture = False

        # [2] 상태 전환 및 카운팅 (올바른 자세를 유지 중일 때만 작동)
        if is_correct_posture:
            # DOWN 자세: 몸통-팔 각도 약 90도, 팔꿈치 각도 약 90도 (여유를 두어 70~120도 구간으로 설정)
            is_down_pose = (70 <= l_shoulder_angle <= 120) and (70 <= l_elbow_angle <= 120) and \
                           (70 <= r_shoulder_angle <= 120) and (70 <= r_elbow_angle <= 120)
            
            # UP 자세: 몸통-팔 각도 150도 이상, 팔꿈치 각도 150도 이상 (쭉 핀 상태)
            is_up_pose = (l_shoulder_angle >= 150 and l_elbow_angle >= 150 and \
                          r_shoulder_angle >= 150 and r_elbow_angle >= 150)

            if is_down_pose:
                if self.state == "UP":
                    self.state = "DOWN"
                feedback = "Ready... Push UP!"
                color = (0, 255, 255) # 노란색 (준비)
                
            elif is_up_pose:
                if self.state == "DOWN":
                    self.state = "UP"
                    self.count += 1  # ⭐️ 다운 -> 업으로 전환될 때 1회 카운트 증가!
                feedback = "Perfect! Keep going!"
                color = (255, 0, 0)  # 파란색 (성공)

        # 화면 텍스트 출력
        cv2.putText(image, feedback, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f"L Elbow: {int(l_elbow_angle)} | L Shld: {int(l_shoulder_angle)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"R Elbow: {int(r_elbow_angle)} | R Shld: {int(r_shoulder_angle)}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # ⭐️ 우측 상단에 횟수(Count) 크게 출력
        cv2.putText(image, f"Count: {self.count}", (w - 280, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        return (l_elbow_angle + r_elbow_angle) / 2, feedback

# ==========================================
# 2. ROS2 메인 노드
# ==========================================
class PoseTrackingNode(Node):
    def __init__(self, exercise_type='shoulder_press'): # 기본값을 shoulder_press로 변경
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

        # 딕셔너리에 숄더 프레스 추가
        self.analyzers = {
            'shoulder_flexion': ShoulderFlexionAnalyzer(),
            'shoulder_press': ShoulderPressAnalyzer()
        }
        self.current_analyzer = self.analyzers.get(exercise_type, ShoulderPressAnalyzer())

        self.get_logger().info(f"💪 [{exercise_type}] 트레이닝 모드로 비전 노드가 시작되었습니다.")

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
    # 인자를 shoulder_press로 주어 숄더 프레스 모드로 실행되게 합니다.
    node = PoseTrackingNode(exercise_type='shoulder_press') 
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()