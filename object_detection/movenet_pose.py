import os
import time
import csv
import numpy as np
import cv2

# [추가] GPU 가속 라이브러리(cuDNN) 충돌 방지를 위해 CPU 사용 강제 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_hub as hub
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge

class MoveNetPoseNode(Node):
    def __init__(self):
        super().__init__('movenet_pose_node')
        self.bridge = CvBridge()
        
        self.get_logger().info("⏳ MoveNet 모델 로드 중...")
        # [수정] 모델 로드 방식 유지
        self.module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        self.model = self.module.signatures['serving_default']
        
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_file_path = os.path.join(current_dir, 'movenet_metrics.csv')

        with open(self.log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "FPS", "Angle"])

        self.prev_time = time.time()
        self.get_logger().info("🚀 MoveNet-Pose 분석 및 로그 기록 엔진 가동 완료!")

    def calculate_angle(self, a, b, c):
        # 관절 각도 계산 공식: 
        # $$angle = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}| |\vec{BC}|}\right)$$
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w, _ = frame.shape
            
            # MoveNet 추론 전처리
            input_image = cv2.resize(frame, (192, 192))
            input_image = tf.expand_dims(input_image, axis=0)
            input_image = tf.cast(input_image, dtype=tf.int32)
            
            # [수정] TypeError 방지를 위해 키워드 인자(input=) 명시적 사용
            outputs = self.model(input=input_image)
            keypoints = outputs['output_0'].numpy()[0, 0, :, :]

            current_angle = 0.0
            # Index: 6: R-Shoulder, 8: R-Elbow, 10: R-Wrist
            ry, rx, rs = keypoints[6]
            ey, ex, es = keypoints[8]
            wy, wx, ws = keypoints[10]

            if all(score > 0.2 for score in [rs, es, ws]):
                shoulder = [rx * w, ry * h]
                elbow = [ex * w, ey * h]
                wrist = [wx * w, wy * h]
                
                current_angle = self.calculate_angle(shoulder, elbow, wrist)
                self.angle_pub.publish(Float32(data=float(current_angle)))
                
                for pt in [shoulder, elbow, wrist]:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

            inf_end = time.time()
            fps = 1.0 / (inf_end - self.prev_time)
            self.prev_time = inf_end

            with open(self.log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"{inf_end:.3f}", f"{fps:.1f}", f"{current_angle:.2f}"])

            cv2.putText(frame, f"MoveNet FPS: {fps:.1f}", (20, 50), 2, 1, (0, 255, 0), 2)
            cv2.imshow("MoveNet Metrics Collector", frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"이미지 처리 중 오류 발생: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = MoveNetPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()