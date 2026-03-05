import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import time
import csv # 로그 저장을 위해 추가

class YOLO11PoseNode(Node):
    def __init__(self):
        super().__init__('yolo11_pose_node')
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        
        # 모델 로드
        self.model = YOLO('yolo11n-pose.pt')
        
        # [핵심] 로그 파일 설정
        self.log_file_path = '/tmp/yolo11n_metrics.csv'
        with open(self.log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "FPS", "Angle"]) # 헤더 작성

        self.prev_time = time.time()
        self.get_logger().info("🚀 YOLOv11n-Pose 분석 및 로그 기록 엔진 가동!")

    def calculate_angle(self, a, b, c):
        # 관절 각도 계산 공식: 
        # $$\theta = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}| |\vec{BC}|}\right)$$
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def image_callback(self, msg):
        start_time = time.time()
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # YOLOv11 추론
        results = self.model(frame, verbose=False, device='cpu')[0]
        
        current_angle = 0.0
        if results.keypoints:
            # COCO: 6: R-Shoulder, 8: R-Elbow, 10: R-Wrist
            kpts = results.keypoints.xyn[0].cpu().numpy()
            if len(kpts) > 10:
                shoulder, elbow, wrist = kpts[6], kpts[8], kpts[10]
                if all(shoulder + elbow + wrist):
                    current_angle = self.calculate_angle(shoulder, elbow, wrist)
                    self.angle_pub.publish(Float32(data=float(current_angle)))

        # 성능 계산
        inf_end = time.time()
        fps = 1.0 / (inf_end - self.prev_time)
        self.prev_time = inf_end

        # [핵심] CSV 파일에 실시간 기록
        with open(self.log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"{inf_end:.3f}", f"{fps:.1f}", f"{current_angle:.2f}"])

        # 시각화
        annotated_frame = results.plot()
        cv2.putText(annotated_frame, f"YOLO11 FPS: {fps:.1f}", (20, 50), 2, 1, (0, 255, 0), 2)
        cv2.imshow("YOLOv11 Metrics Collector", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YOLO11PoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()