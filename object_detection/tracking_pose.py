import sys
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np

# [핵심] solutions 대신 최신 Tasks API 사용
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseTrackingNode(Node):
    def __init__(self):
        super().__init__('pose_tracking_node')
        self.bridge = CvBridge()
        
        # 실제 카메라 토픽 주소로 수정됨
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        
        # Tasks API 설정 및 모델 로드 (다운로드 받은 .task 파일 경로 지정)
        model_path = '/home/rokey/cobot_ws/src/cobot2_ws/rehab_assist_robot/object_detection/pose_landmarker_lite.task'
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.get_logger().info("💪 [Tasks API] MediaPipe 트레이닝 비전 노드가 시작되었습니다.")

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # MediaPipe 이미지 객체로 변환 후 추론
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = self.detector.detect(mp_image)

            # 포즈가 감지되었을 경우
            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0] # 첫 번째 사람의 관절 데이터
                
                # 오른쪽 관절 좌표 추출 (12: 어깨, 14: 팔꿈치, 16: 손목)
                # 좌표는 0.0 ~ 1.0 사이의 정규화된 값입니다.
                shoulder = [landmarks[12].x, landmarks[12].y]
                elbow = [landmarks[14].x, landmarks[14].y]
                wrist = [landmarks[16].x, landmarks[16].y]
                
                angle = self.calculate_angle(shoulder, elbow, wrist)
                
                angle_msg = Float32()
                angle_msg.data = float(angle)
                self.angle_pub.publish(angle_msg)
                
                self.get_logger().info(f"현재 팔꿈치 각도: {int(angle)}도")
                
                # --- 시각화 코드 수정 부분 ---
                h, w, _ = image.shape
                
                # 정규화된 좌표를 픽셀 좌표로 변환
                shoulder_px = (int(shoulder[0]*w), int(shoulder[1]*h))
                elbow_px = (int(elbow[0]*w), int(elbow[1]*h))
                wrist_px = (int(wrist[0]*w), int(wrist[1]*h))

                # [요청사항] 12번-14번-16번 선으로 연결 (초록색, 두께 3)
                cv2.line(image, shoulder_px, elbow_px, (0, 255, 0), 3)
                cv2.line(image, elbow_px, wrist_px, (0, 255, 0), 3)

                # 기존 팔꿈치 점 그리기 유지
                cv2.circle(image, elbow_px, 10, (0, 255, 0), -1)
                # --- 시각화 코드 수정 끝 ---

            cv2.imshow('MediaPipe PT Trainer', image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"에러 발생: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PoseTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()