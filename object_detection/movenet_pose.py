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
import tensorflow as tf

class MoveNetNode(Node):
    def __init__(self):
        super().__init__('movenet_pose')
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        
        # MoveNet 모델 로드 (다운로드 받은 경로)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = '/home/rokey/cobot_ws/src/cobot2_ws/rehab_assist_robot/object_detection/movenet_lightning.tflite'
        
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 로깅 관련 초기화
        self.prev_time = time.time()
        self.prev_angle = None
        self.log_file_path = '/tmp/movenet_metrics.csv'
        
        with open(self.log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "FPS", "Inference_ms", "Angle", "Jitter", "CPU_Usage"])

        self.get_logger().info("⚡ [MoveNet] 초고속 로깅 노드가 시작되었습니다.")

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def image_callback(self, msg):
        try:
            inf_start = time.time()
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # MoveNet은 192x192 사이즈의 이미지를 입력으로 받습니다
            input_image = tf.image.resize_with_pad(np.expand_dims(image_rgb, axis=0), 192, 192)
            input_image = tf.cast(input_image, dtype=tf.float32)

            # 추론 실행
            self.interpreter.set_tensor(self.input_details[0]['index'], input_image.numpy())
            self.interpreter.invoke()
            keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])

            current_angle = 0.0
            jitter = 0.0
            
            h, w, _ = image.shape
            keypoints = keypoints_with_scores[0][0] # 17개의 관절 [y, x, score]

            # 오른쪽 어깨(5), 오른쪽 팔꿈치(7), 오른쪽 손목(9) 추출 (score가 0.3 이상일 때만)
            if keypoints[5][2] > 0.3 and keypoints[7][2] > 0.3 and keypoints[9][2] > 0.3:
                # 좌표가 정규화되어 있으므로 해상도를 곱해줍니다 (y, x 순서 주의)
                shoulder = [keypoints[5][1], keypoints[5][0]]
                elbow = [keypoints[7][1], keypoints[7][0]]
                wrist = [keypoints[9][1], keypoints[9][0]]
                
                current_angle = self.calculate_angle(shoulder, elbow, wrist)
                
                angle_msg = Float32()
                angle_msg.data = float(current_angle)
                self.angle_pub.publish(angle_msg)
                
                if self.prev_angle is not None:
                    jitter = abs(current_angle - self.prev_angle)
                self.prev_angle = current_angle

                # 시각화
                shoulder_px = (int(shoulder[0]*w), int(shoulder[1]*h))
                elbow_px = (int(elbow[0]*w), int(elbow[1]*h))
                wrist_px = (int(wrist[0]*w), int(wrist[1]*h))

                cv2.line(image, shoulder_px, elbow_px, (0, 255, 255), 3) # 노란색 선
                cv2.line(image, elbow_px, wrist_px, (0, 255, 255), 3)
                cv2.circle(image, elbow_px, 10, (0, 255, 255), -1)

            # 성능 지표 계산
            inf_end = time.time()
            inf_time_ms = (inf_end - inf_start) * 1000.0
            fps = 1.0 / (inf_end - self.prev_time)
            self.prev_time = inf_end
            cpu_usage = psutil.Process(os.getpid()).cpu_percent()

            # CSV 기록
            with open(self.log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"{inf_end:.3f}", f"{fps:.1f}", f"{inf_time_ms:.1f}", 
                                 f"{current_angle:.2f}", f"{jitter:.3f}", f"{cpu_usage:.1f}"])

            display_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('MoveNet Evaluation', display_image)
            cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = MoveNetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()