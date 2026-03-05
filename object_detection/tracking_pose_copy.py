import sys
import os
import time
import csv
import psutil
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np

# MediaPipe Tasks API 임포트
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseTrackingNode(Node):
    def __init__(self):
        # 노드 이름을 파일명과 일치하게 수정
        super().__init__('tracking_pose')
        self.bridge = CvBridge()
        
        # 1. 카메라 구독 및 각도 퍼블리셔
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        
        # 2. Tasks API 설정 및 모델 로드
        model_path = "/home/rokey/cobot_ws/src/cobot2_ws/rehab_assist_robot/object_detection/pose_landmarker_lite.task"
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # 3. 로깅 및 녹화 관련 초기화
        self.prev_time = time.time()
        self.prev_angle = None
        self.video_writer = None
        self.log_file_path = '/tmp/pose_metrics.csv'
        self.video_file_path = '/tmp/pose_output.avi'
        
        # CSV 파일 초기화 및 헤더 작성
        with open(self.log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "FPS", "Inference_ms", "Angle", "Jitter", "CPU_Usage"])

        self.get_logger().info(f"💪 [Tasks API] 로깅 및 녹화 노드가 시작되었습니다.")
        self.get_logger().info(f"📁 로그: {self.log_file_path} | 영상: {self.video_file_path}")

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def image_callback(self, msg):
        try:
            inf_start = time.time() # 추론 시작 시간
            
            # ROS 이미지를 OpenCV BGR로 변환
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # MediaPipe 추론
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = self.detector.detect(mp_image)

            current_angle = 0.0
            jitter = 0.0

            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                
                # 오른쪽 관절 좌표 (12, 14, 16)
                shoulder = [landmarks[12].x, landmarks[12].y]
                elbow = [landmarks[14].x, landmarks[14].y]
                wrist = [landmarks[16].x, landmarks[16].y]
                
                current_angle = self.calculate_angle(shoulder, elbow, wrist)
                
                # 각도 퍼블리시
                angle_msg = Float32()
                angle_msg.data = float(current_angle)
                self.angle_pub.publish(angle_msg)
                
                # 지터(Jitter) 계산
                if self.prev_angle is not None:
                    jitter = abs(current_angle - self.prev_angle)
                self.prev_angle = current_angle

                # 시각화 (기존 로직 유지)
                h, w, _ = image.shape
                shoulder_px = (int(shoulder[0]*w), int(shoulder[1]*h))
                elbow_px = (int(elbow[0]*w), int(elbow[1]*h))
                wrist_px = (int(wrist[0]*w), int(wrist[1]*h))

                cv2.line(image, shoulder_px, elbow_px, (0, 255, 0), 3)
                cv2.line(image, elbow_px, wrist_px, (0, 255, 0), 3)
                cv2.circle(image, elbow_px, 10, (0, 255, 0), -1)

            # --- 성능 지표 계산 ---
            inf_end = time.time()
            inf_time_ms = (inf_end - inf_start) * 1000.0
            fps = 1.0 / (inf_end - self.prev_time)
            self.prev_time = inf_end
            cpu_usage = psutil.Process(os.getpid()).cpu_percent()

            # --- 데이터 로깅 (CSV) ---
            with open(self.log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"{inf_end:.3f}", f"{fps:.1f}", f"{inf_time_ms:.1f}", 
                                 f"{current_angle:.2f}", f"{jitter:.3f}", f"{cpu_usage:.1f}"])

            # --- 영상 녹화 (VideoWriter) ---
            if self.video_writer is None:
                h, w, _ = image.shape
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(self.video_file_path, fourcc, 20.0, (w, h))
            
            self.video_writer.write(image)

            # 실시간 출력
            cv2.imshow('MediaPipe Evaluation', image)
            cv2.waitKey(1)
            
            if int(current_angle) > 0:
                self.get_logger().info(f"FPS: {fps:.1f} | 각도: {int(current_angle)}° | Jitter: {jitter:.2f}")
                
        except Exception as e:
            self.get_logger().error(f"에러 발생: {e}")

    def destroy_node(self):
        # 종료 시 자원 해제
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PoseTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()