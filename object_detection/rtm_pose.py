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
from openvino.runtime import Core

class RTMPoseNode(Node):
    def __init__(self):
        super().__init__('rtmpose_node')
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        
        # 1. OpenVINO 엔진 및 모델 로드
        self.core = Core()
        model_path = '/home/rokey/cobot_ws/src/cobot2_ws/rehab_assist_robot/object_detection/rtmpose-m.onnx'
        self.model = self.core.read_model(model=model_path)
        self.compiled_model = self.core.compile_model(model=self.model, device_name="CPU")
        self.infer_request = self.compiled_model.create_infer_request()
        
        # 로깅 설정
        self.prev_time = time.time()
        self.prev_angle = None
        self.log_file_path = '/tmp/rtm_metrics.csv'
        with open(self.log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "FPS", "Inference_ms", "Angle", "Jitter", "CPU_Usage"])

        self.get_logger().info("💎 [RTMPose + OpenVINO] 지능형 추적 노드가 시작되었습니다.")

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        # 관절 각도 계산 공식: 
        # $$\theta = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}| |\vec{BC}|}\right)$$
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def image_callback(self, msg):
        try:
            inf_start = time.time()
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w, _ = image.shape
            
            # 2. 전처리 (256x192 입력 사이즈 조절)
            input_img = cv2.resize(image, (192, 256))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img = input_img.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_img = np.expand_dims(input_img, axis=0)

            # 3. OpenVINO 추론
            results = self.compiled_model([input_img])
            simcc_x = results[self.compiled_model.output(0)]
            simcc_y = results[self.compiled_model.output(1)]

            # 4. 좌표 추출 (간이 디코딩)
            # RTMPose-m (COCO) 기준: 5: R-Shoulder, 7: R-Elbow, 9: R-Wrist
            def get_kpt(idx):
                x = np.argmax(simcc_x[0, idx]) / simcc_x.shape[2] * w
                y = np.argmax(simcc_y[0, idx]) / simcc_y.shape[2] * h
                return [x, y]

            shoulder = get_kpt(6) # Right Shoulder
            elbow = get_kpt(8)    # Right Elbow
            wrist = get_kpt(10)   # Right Wrist

            current_angle = self.calculate_angle(shoulder, elbow, wrist)
            
            # 성능 지표 및 로깅
            inf_end = time.time()
            fps = 1.0 / (inf_end - self.prev_time)
            self.prev_time = inf_end
            jitter = abs(current_angle - self.prev_angle) if self.prev_angle else 0.0
            self.prev_angle = current_angle
            cpu_usage = psutil.Process(os.getpid()).cpu_percent()

            with open(self.log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"{inf_end:.3f}", f"{fps:.1f}", f"{(inf_end-inf_start)*1000:.1f}", 
                                 f"{current_angle:.2f}", f"{jitter:.3f}", f"{cpu_usage:.1f}"])

            # 시각화
            cv2.line(image, tuple(map(int, shoulder)), tuple(map(int, elbow)), (255, 0, 255), 3)
            cv2.line(image, tuple(map(int, elbow)), tuple(map(int, wrist)), (255, 0, 255), 3)
            cv2.imshow('RTMPose OpenVINO', cv2.resize(image, (w//2, h//2)))
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RTMPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()