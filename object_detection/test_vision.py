import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class FrontElbowTestNode(Node):
    def __init__(self):
        super().__init__('front_elbow_test_node')
        self.bridge = CvBridge()

        # 오직 로봇 카메라만 구독 (자세 평가 제외)
        self.create_subscription(Image, '/robot/camera/color/image_raw', self.color_callback, qos_profile_sensor_data)
        self.create_subscription(Image, '/robot/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/robot/camera/color/camera_info', self.info_callback, qos_profile_sensor_data)

        self.elbow_pub = self.create_publisher(Point, '/right_elbow_3d', 10)

        self.color_img = None
        self.depth_img = None
        self.intrinsics = None
        self.pose_model = YOLO('yolo11n-pose.pt')

        self.timer = self.create_timer(0.033, self.process_frame)
        self.get_logger().info("🔍 [테스트 모드] 로봇 카메라 정면 우측 팔꿈치 추적 시작!")
        self.get_logger().info("👉 로봇 카메라를 사용자 정면에 두고, 스페이스바를 누르세요.")

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {"fx": msg.k[0], "fy": msg.k[4], "ppx": msg.k[2], "ppy": msg.k[5]}

    def color_callback(self, msg):
        self.color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_callback(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _pixel_to_camera_coords(self, x, y, z):
        fx, fy, ppx, ppy = self.intrinsics['fx'], self.intrinsics['fy'], self.intrinsics['ppx'], self.intrinsics['ppy']
        return ((x - ppx) * z / fx, (y - ppy) * z / fy, z)

    def process_frame(self):
        if self.color_img is None or self.depth_img is None or self.intrinsics is None:
            return

        # 1. 로봇 카메라 원본 해상도(예: 1280x720) 기억해두기
        orig_h, orig_w = self.color_img.shape[:2]

        # 2. YOLO 연산을 위해 화면만 640x480으로 줄이기 (뎁스는 안 줄임!)
        img = cv2.resize(self.color_img, (640, 480))
        
        results = self.pose_model(img, verbose=False)[0]
        elbow_3d = None

        if results.keypoints is not None and len(results.keypoints.data) > 0:
            kpts = results.keypoints.data[0].cpu().numpy()
            r_el = kpts[8][:2] # 8번 인덱스: 우측 팔꿈치
            r_el_pt = (int(r_el[0]), int(r_el[1]))

            # ---------------------------------------------------------
            # ⭐️ [핵심 수정] 640x480 좌표를 원본 1280x720 스케일로 복원
            # ---------------------------------------------------------
            scale_x = orig_w / 640.0
            scale_y = orig_h / 480.0
            
            orig_x = int(r_el_pt[0] * scale_x)
            orig_y = int(r_el_pt[1] * scale_y)

            # 원본 해상도 범위 안에 있는지 확인
            if 0 <= orig_x < orig_w and 0 <= orig_y < orig_h:
                # 3. 리사이즈 하지 않은 '원본 뎁스 이미지'에서 깊이(Z) 가져오기
                cz = float(self.depth_img[orig_y, orig_x])
                
                if cz > 0:
                    # 4. 복원된 원본 좌표(orig_x, orig_y)를 카메라 공식에 대입! (뻥튀기 제거)
                    cx, cy, cz = self._pixel_to_camera_coords(orig_x, orig_y, cz)
                    elbow_3d = (cx, cy, cz)
                    
                    # 화면 시각화 (그림은 640 이미지에 그리는 거라 원래 좌표 r_el_pt 사용)
                    cv2.circle(img, r_el_pt, 8, (0, 255, 255), -1)
                    cv2.putText(img, f"X:{int(cx)} Y:{int(cy)} Z:{int(cz)}", (r_el_pt[0]+15, r_el_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Robot Camera - Front Elbow Tracking", img)
        key = cv2.waitKey(1) & 0xFF
        
        # 스페이스바로 좌표 발행
        if key == 32 and elbow_3d is not None:
            msg = Point(x=elbow_3d[0], y=elbow_3d[1], z=elbow_3d[2])
            self.elbow_pub.publish(msg)
            self.get_logger().info(f"🚀 퍼블리시 완료 -> 카메라 좌표: X={msg.x:.1f}, Y={msg.y:.1f}, Z={msg.z:.1f}")
def main(args=None):
    rclpy.init(args=args)
    node = FrontElbowTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()