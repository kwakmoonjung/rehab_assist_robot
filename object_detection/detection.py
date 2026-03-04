########## detection.py ##########
import rclpy
from rclpy.node import Node
import cv2
import time
import numpy as np
from datetime import datetime
from collections import deque
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory
from od_msg.srv import SrvDepthPosition
from sensor_msgs.msg import Image
from object_detection.realsense import ImgNode
from object_detection.yolo import YoloModel

PACKAGE_NAME = 'rehab_assist_robot'

class ObjectDetectionNode(Node):
    def __init__(self, model_name='yolo'):
        super().__init__('object_detection_node')
        self.img_node = ImgNode()
        self.cv_bridge = CvBridge()
        
        # 모드 설정: 'performance' (속도 중심) / 'rviz' (화면+정보) / 'service' (요청 대기)
        self.declare_parameter('mode', 'performance') 
        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        
        self.model = self._load_model(model_name)
        
        # Rviz 디버그 이미지 퍼블리셔
        self.debug_pub = self.create_publisher(Image, '/yolo/debug_image', 10)

        self.intrinsics = self._wait_for_valid_data(
            self.img_node.get_camera_intrinsic, "camera intrinsics"
        )
        self.create_service(SrvDepthPosition, 'get_3d_position', self.handle_get_depth)
        
        # 스켈레톤 연결 정보 (그리기용)
        self.SKELETON_CONNECTIONS = [
            (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (0, 1), (0, 2), (1, 3), (2, 4)
        ]
        
        self.get_logger().info(f"Initialized in [{self.mode.upper()}] mode.")
        
        # 'performance' 또는 'rviz' 모드이면 타이머로 무한 반복 실행
        if self.mode in ['performance', 'rviz']:
            # 0.01초마다 루프 실행 (최대 속도)
            self.timer = self.create_timer(0.01, self._continuous_loop)

    def _load_model(self, name):
        if name.lower() == 'yolo':
            return YoloModel()
        raise ValueError(f"Unsupported model: {name}")

    def _continuous_loop(self):
        """서비스 호출 없이 스스로 계속 실행하는 루프"""
        self._compute_position('person')

    def handle_get_depth(self, request, response):
        """로봇 서비스 요청 처리"""
        coords = self._compute_position(request.target)
        response.depth_position = [float(x) for x in coords]
        return response

    def _compute_position(self, target):
        # 1. 이미지 갱신
        rclpy.spin_once(self.img_node)
        
        t_start = time.perf_counter()
        
        # 2. YOLO 추론 (1초 대기 없음)
        box, score, keypoints = self.model.get_best_detection(self.img_node, target)
        
        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000.0
        fps = 1000.0 / inference_ms if inference_ms > 0 else 0

        # 데이터 초기화
        right_elbow_angle = 0.0
        depth_val = 0.0
        target_coords = (0, 0, 0)

        # 3. 데이터 계산
        if box is not None and keypoints:
            # (1) Depth 및 좌표 계산 (코 기준)
            nose_x, nose_y, conf = keypoints[0]
            if conf > 0.5:
                tx, ty = int(nose_x), int(nose_y)
            else:
                tx, ty = map(int, [(box[0]+box[2])/2, (box[1]+box[3])/2])
            
            cz = self._get_depth(tx, ty)
            if cz:
                depth_val = float(cz)
                target_coords = self._pixel_to_camera_coords(tx, ty, cz)

            # (2) 오른쪽 팔꿈치 각도 계산 (R_Shoulder:6, R_Elbow:8, R_Wrist:10)
            kp_shoulder = keypoints[6]
            kp_elbow = keypoints[8]
            kp_wrist = keypoints[10]

            # 세 점 모두 신뢰도가 0.5 이상일 때만 계산
            if kp_shoulder[2] > 0.5 and kp_elbow[2] > 0.5 and kp_wrist[2] > 0.5:
                right_elbow_angle = self._calculate_angle(
                    (kp_shoulder[0], kp_shoulder[1]), 
                    (kp_elbow[0], kp_elbow[1]), 
                    (kp_wrist[0], kp_wrist[1])
                )

        # 4. 터미널 출력 (Performance, Rviz 모드 모두 출력)
        if self.mode in ['performance', 'rviz']:
            print(f"\r[Mode:{self.mode}] FPS: {fps:4.1f} | Angle(R_Elbow): {right_elbow_angle:5.1f}° | Depth: {depth_val:.0f}mm", end="")

        # 5. Rviz 시각화 (Rviz 모드일 때만 실행)
        if self.mode == 'rviz':
            self._process_visualization(box, keypoints, right_elbow_angle)

        return target_coords

    def _process_visualization(self, box, keypoints, angle):
        """이미지에 뼈대와 각도를 그리고 ROS 토픽으로 발행"""
        if self.img_node.color_frame is None: return
        
        debug_img = self.img_node.color_frame.copy()
        
        if box is not None:
            # 박스 그리기
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 스켈레톤 그리기
            if keypoints:
                for p1, p2 in self.SKELETON_CONNECTIONS:
                    if keypoints[p1][2] > 0.5 and keypoints[p2][2] > 0.5:
                        pt1 = (int(keypoints[p1][0]), int(keypoints[p1][1]))
                        pt2 = (int(keypoints[p2][0]), int(keypoints[p2][1]))
                        cv2.line(debug_img, pt1, pt2, (0, 255, 255), 2)
            
            # 오른쪽 팔꿈치 각도 텍스트 표시 (R_Elbow: index 8)
            if keypoints and keypoints[8][2] > 0.5:
                ex, ey = int(keypoints[8][0]), int(keypoints[8][1])
                cv2.circle(debug_img, (ex, ey), 5, (0, 0, 255), -1) # 팔꿈치에 빨간 점
                cv2.putText(debug_img, f"{angle:.1f} deg", 
                           (ex + 15, ey), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 이미지 발행 (/yolo/debug_image)
        try:
            ros_image = self.cv_bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            self.debug_pub.publish(ros_image)
        except Exception:
            pass

    def _calculate_angle(self, a, b, c):
        """점 b를 중심으로 a-b-c의 각도를 계산"""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return 0.0

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def _get_depth(self, x, y):
        frame = self._wait_for_valid_data(self.img_node.get_depth_frame, "depth frame")
        try:
            region = frame[max(0, y-1):y+2, max(0, x-1):x+2]
            valid = region[region > 0]
            if valid.size == 0: return None
            return np.median(valid)
        except Exception: return None

    def _wait_for_valid_data(self, getter, desc):
        data = getter()
        while data is None or (isinstance(data, np.ndarray) and not data.any()):
            rclpy.spin_once(self.img_node)
            data = getter()
        return data

    def _pixel_to_camera_coords(self, x, y, z):
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        ppx = self.intrinsics['ppx']
        ppy = self.intrinsics['ppy']
        return ((x - ppx) * z / fx, (y - ppy) * z / fy, z)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()