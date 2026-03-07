import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
from geometry_msgs.msg import Point  # 3D 좌표 퍼블리시용
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. 운동 분석기 모듈
# ==========================================
class ExerciseAnalyzer:
    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def analyze(self, kpts, image):
        raise NotImplementedError

class ShoulderPressAnalyzer(ExerciseAnalyzer):
    def __init__(self):
        self.count = 0         
        self.state = "UP"      

    def analyze(self, kpts, image):
        h, w, _ = image.shape
        nose = [kpts[0][0], kpts[0][1]]
        
        l_sh, l_el, l_wr, l_hip = [kpts[5][0], kpts[5][1]], [kpts[7][0], kpts[7][1]], [kpts[9][0], kpts[9][1]], [kpts[11][0], kpts[11][1]]
        r_sh, r_el, r_wr, r_hip = [kpts[6][0], kpts[6][1]], [kpts[8][0], kpts[8][1]], [kpts[10][0], kpts[10][1]], [kpts[12][0], kpts[12][1]]

        l_pts = [(int(l_sh[0]*w), int(l_sh[1]*h)), (int(l_el[0]*w), int(l_el[1]*h)), 
                 (int(l_wr[0]*w), int(l_wr[1]*h)), (int(l_hip[0]*w), int(l_hip[1]*h))]
        r_pts = [(int(r_sh[0]*w), int(r_sh[1]*h)), (int(r_el[0]*w), int(r_el[1]*h)), 
                 (int(r_wr[0]*w), int(r_wr[1]*h)), (int(r_hip[0]*w), int(r_hip[1]*h))]

        # 1. 중앙점 좌표 계산
        mid_x = int((l_pts[2][0] + r_pts[2][0]) / 2)
        mid_y = int((l_pts[2][1] + r_pts[2][1]) / 2)
        
        # 2. 왼쪽 손목 좌표 추출
        l_wr_x, l_wr_y = l_pts[2][0], l_pts[2][1]
        
        # 시각화 (중앙점: 노란색, 왼쪽 손목: 핑크색)
        cv2.circle(image, (mid_x, mid_y), 10, (0, 255, 255), -1)
        cv2.circle(image, (l_wr_x, l_wr_y), 10, (255, 0, 255), -1)

        l_elbow_angle = self.calculate_angle(l_sh, l_el, l_wr)
        r_elbow_angle = self.calculate_angle(r_sh, r_el, r_wr)

        # [수정] 중앙점(mid_x, mid_y)과 왼쪽손목(l_wr_x, l_wr_y) 두 개를 모두 반환
        return (l_elbow_angle + r_elbow_angle) / 2, "Tracking...", (mid_x, mid_y), (l_wr_x, l_wr_y)


# ==========================================
# 2. ROS2 메인 노드
# ==========================================
class PoseTrackingNode(Node):
    def __init__(self, exercise_type='shoulder_press'):
        super().__init__('pose_tracking_node')
        self.bridge = CvBridge()
        
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        
        # [수정] 중앙점과 왼쪽 손목 퍼블리셔 모두 생성
        self.wrist_3d_pub = self.create_publisher(Point, '/wrist_midpoint_3d', 10)
        self.left_wrist_3d_pub = self.create_publisher(Point, '/left_wrist_3d', 10)
        
        self.model = YOLO('yolov8n-pose.pt')
        self.current_analyzer = ShoulderPressAnalyzer()

        self.depth_frame = None
        self.intrinsics = None

        self.get_logger().info(f"🚀 YOLOv11 & 3D Depth Tracker 시작됨. (OpenCV 창 클릭 후 [스페이스바]를 누르면 좌표 발행)")

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {"fx": msg.k[0], "fy": msg.k[4], "ppx": msg.k[2], "ppy": msg.k[5]}

    def _get_depth(self, x, y):
        if self.depth_frame is None: return None
        try: return float(self.depth_frame[y, x])
        except IndexError: return None

    def _pixel_to_camera_coords(self, x, y, z):
        fx, fy, ppx, ppy = self.intrinsics['fx'], self.intrinsics['fy'], self.intrinsics['ppx'], self.intrinsics['ppy']
        return ((x - ppx) * z / fx, (y - ppy) * z / fy, z)

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.model(image, verbose=False, device='cpu')[0]
            
            target_3d_coord = None       # 중앙점 3D 좌표 보관용
            left_wrist_3d_coord = None   # 왼쪽 손목 3D 좌표 보관용

            if results.keypoints is not None and len(results.keypoints.xyn) > 0:
                kpts = results.keypoints.xyn[0].cpu().numpy()
                
                if len(kpts) >= 13: 
                    # 반환값 튜플 언패킹 수정 (왼쪽 손목 좌표 추가로 받음)
                    target_angle, feedback, (mid_x, mid_y), (l_wr_x, l_wr_y) = self.current_analyzer.analyze(kpts, image)
                    
                    angle_msg = Float32()
                    angle_msg.data = float(target_angle)
                    self.angle_pub.publish(angle_msg)

                    if self.intrinsics is not None:
                        # 1. 중앙점 3D 좌표 변환
                        if mid_x > 0 and mid_y > 0:
                            cz = self._get_depth(mid_x, mid_y)
                            if cz is not None and cz > 0:
                                cx, cy, cz = self._pixel_to_camera_coords(mid_x, mid_y, cz)
                                target_3d_coord = (cx, cy, cz)
                                cv2.putText(image, f"Mid 3D X:{int(cx)} Y:{int(cy)} Z:{int(cz)}", 
                                            (mid_x - 70, mid_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        # 2. 왼쪽 손목 3D 좌표 변환
                        if l_wr_x > 0 and l_wr_y > 0:
                            l_cz = self._get_depth(l_wr_x, l_wr_y)
                            if l_cz is not None and l_cz > 0:
                                l_cx, l_cy, l_cz = self._pixel_to_camera_coords(l_wr_x, l_wr_y, l_cz)
                                left_wrist_3d_coord = (l_cx, l_cy, l_cz)
                                cv2.putText(image, f"L-Wr 3D X:{int(l_cx)} Y:{int(l_cy)} Z:{int(l_cz)}", 
                                            (l_wr_x - 70, l_wr_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            cv2.imshow('YOLOv11 PT Trainer', image)
            key = cv2.waitKey(1) & 0xFF
            
            # 스페이스바(아스키코드 32) 눌렀을 때 두 개의 토픽을 각각 발행
            if key == 32:
                # 중앙점 퍼블리시
                if target_3d_coord is not None:
                    point_msg = Point()
                    point_msg.x = float(target_3d_coord[0])
                    point_msg.y = float(target_3d_coord[1])
                    point_msg.z = float(target_3d_coord[2])
                    self.wrist_3d_pub.publish(point_msg)
                    self.get_logger().info(f"✅ [중앙점] 목표 좌표 발행: X:{int(target_3d_coord[0])}, Y:{int(target_3d_coord[1])}, Z:{int(target_3d_coord[2])}")
                else:
                    self.get_logger().warn("[중앙점] 유효한 3D 좌표가 없습니다.")

                # 왼쪽 손목 퍼블리시
                if left_wrist_3d_coord is not None:
                    l_point_msg = Point()
                    l_point_msg.x = float(left_wrist_3d_coord[0])
                    l_point_msg.y = float(left_wrist_3d_coord[1])
                    l_point_msg.z = float(left_wrist_3d_coord[2])
                    self.left_wrist_3d_pub.publish(l_point_msg)
                    self.get_logger().info(f"✅ [왼쪽 손목] 목표 좌표 발행: X:{int(left_wrist_3d_coord[0])}, Y:{int(left_wrist_3d_coord[1])}, Z:{int(left_wrist_3d_coord[2])}")
                else:
                    self.get_logger().warn("[왼쪽 손목] 유효한 3D 좌표가 없습니다.")
            
        except Exception as e:
            self.get_logger().error(f"에러 발생: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PoseTrackingNode(exercise_type='shoulder_press') 
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()