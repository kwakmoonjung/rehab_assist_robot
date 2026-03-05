import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import csv
import psutil
import os
from ultralytics import YOLO

class YOLOPoseNode(Node):
    def __init__(self):
        super().__init__('yolo_pose')
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        
        # 모델 로드 (가장 가벼운 Nano 버전 사용)
        self.model = YOLO('yolov8n-pose.pt') 
        
        self.prev_time = time.time()
        self.prev_angle = None
        self.log_file_path = '/tmp/yolo_metrics.csv'
        
        with open(self.log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "FPS", "Inference_ms", "Angle", "Jitter", "CPU_Usage"])

        self.get_logger().info("🚀 [YOLOv8-Pose] 정밀 추적 노드가 시작되었습니다.")

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
            
            # YOLO 추론 (CPU 사용)
            results = self.model(image, verbose=False, device='cpu')[0]
            
            current_angle = 0.0
            jitter = 0.0
            
            if results.keypoints is not None and len(results.keypoints.data) > 0:
                # 첫 번째 사람의 키포인트 [N, 17, 3] -> x, y, conf
                kpts = results.keypoints.data[0].cpu().numpy()
                
                # 오른쪽 어깨(5), 팔꿈치(7), 손목(9)
                if kpts[5][2] > 0.5 and kpts[7][2] > 0.5 and kpts[9][2] > 0.5:
                    shoulder = kpts[5][:2]
                    elbow = kpts[7][:2]
                    wrist = kpts[9][:2]
                    
                    current_angle = self.calculate_angle(shoulder, elbow, wrist)
                    
                    angle_msg = Float32()
                    angle_msg.data = float(current_angle)
                    self.angle_pub.publish(angle_msg)
                    
                    if self.prev_angle is not None:
                        jitter = abs(current_angle - self.prev_angle)
                    self.prev_angle = current_angle

                    # 시각화 (초록색)
                    cv2.line(image, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), (0, 255, 0), 3)
                    cv2.line(image, tuple(elbow.astype(int)), tuple(wrist.astype(int)), (0, 255, 0), 3)

            inf_end = time.time()
            inf_time_ms = (inf_end - inf_start) * 1000.0
            fps = 1.0 / (inf_end - self.prev_time)
            self.prev_time = inf_end
            cpu_usage = psutil.Process(os.getpid()).cpu_percent()

            with open(self.log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"{inf_end:.3f}", f"{fps:.1f}", f"{inf_time_ms:.1f}", 
                                 f"{current_angle:.2f}", f"{jitter:.3f}", f"{cpu_usage:.1f}"])

            display_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('YOLOv8-Pose Evaluation', display_image)
            cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YOLOPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()