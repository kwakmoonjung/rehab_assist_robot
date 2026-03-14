import os
import sys

# [추가] TensorFlow와 JAX의 로그 충돌 및 초기화 에러 방지
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # CPU 모드 강제 (안정성 확보)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import csv
import mediapipe as mp

from rclpy.qos import qos_profile_sensor_data # [추가] 센서 데이터용 QoS 프로필 임포트

class MediaPipePoseNode(Node):
    def __init__(self):
        super().__init__('mediapipe_pose_node')
        self.bridge = CvBridge()
        
        # [수정] 토픽명을 bag 파일에 맞춰 '/camera/camera/color/image_raw'로 복구
        self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.image_callback, 
            qos_profile_sensor_data
        )
        self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        
        # [수정] MediaPipe 초기화 (API 키 미사용)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_file_path = os.path.join(current_dir, 'mediapipe_metrics.csv')

        with open(self.log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "FPS", "Angle"])

        self.prev_time = time.time()
        self.get_logger().info("MediaPipe-Pose 분석 엔진 가동 완료!")

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 추론 실행
            results = self.pose.process(rgb_frame)
            
            current_angle = 0.0
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # R-Shoulder(12), R-Elbow(14), R-Wrist(16)
                shoulder = [lm[12].x * w, lm[12].y * h]
                elbow = [lm[14].x * w, lm[14].y * h]
                wrist = [lm[16].x * w, lm[16].y * h]
                
                if lm[14].visibility > 0.5:
                    current_angle = self.calculate_angle(shoulder, elbow, wrist)
                    self.angle_pub.publish(Float32(data=float(current_angle)))
                
                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            inf_end = time.time()
            fps = 1.0 / (inf_end - self.prev_time)
            self.prev_time = inf_end

            with open(self.log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"{inf_end:.3f}", f"{fps:.1f}", f"{current_angle:.2f}"])

            cv2.putText(frame, f"MediaPipe FPS: {fps:.1f}", (20, 50), 2, 1, (0, 255, 0), 2)
            cv2.imshow("MediaPipe Collector", frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"실행 중 에러 발생: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = MediaPipePoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()