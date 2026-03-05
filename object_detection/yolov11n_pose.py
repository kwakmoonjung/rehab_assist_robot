import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. 운동 분석기 모듈 (YOLO Keypoints 기준)
# ==========================================
class ExerciseAnalyzer:
    """모든 운동 분석기가 상속받을 기본 뼈대 클래스"""
    def calculate_angle(self, a, b, c):
        # 관절 각도 계산 공식
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def analyze(self, kpts, image):
        raise NotImplementedError

class ShoulderFlexionAnalyzer(ExerciseAnalyzer):
    """밴드 숄더플렉션 분석기 (측면 기준)"""
    def analyze(self, kpts, image):
        h, w, _ = image.shape
        
        # YOLO COCO Keypoints (Right side)
        # 6: R-Shoulder, 8: R-Elbow, 10: R-Wrist, 12: R-Hip
        shoulder = [kpts[6][0], kpts[6][1]]
        elbow = [kpts[8][0], kpts[8][1]]
        wrist = [kpts[10][0], kpts[10][1]]
        hip = [kpts[12][0], kpts[12][1]]

        pts = {
            'shoulder': (int(shoulder[0]*w), int(shoulder[1]*h)),
            'elbow': (int(elbow[0]*w), int(elbow[1]*h)),
            'wrist': (int(wrist[0]*w), int(wrist[1]*h)),
            'hip': (int(hip[0]*w), int(hip[1]*h))
        }

        cv2.line(image, pts['hip'], pts['shoulder'], (255, 0, 0), 3)
        cv2.line(image, pts['shoulder'], pts['elbow'], (0, 255, 0), 3)
        cv2.line(image, pts['elbow'], pts['wrist'], (0, 255, 0), 3)
        for pt in pts.values():
            cv2.circle(image, pt, 8, (0, 0, 255), -1)

        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        shoulder_angle = self.calculate_angle(hip, shoulder, elbow)
        vertical_ref = [hip[0], hip[1] - 0.1]
        trunk_angle = self.calculate_angle(vertical_ref, hip, shoulder)

        feedback = "Good Posture!"
        color = (0, 255, 0)
        
        if elbow_angle < 150:
            feedback = "Warning: Straighten your elbow!"
            color = (0, 0, 255)
        elif trunk_angle > 15:
            feedback = "Warning: Do not lean back!"
            color = (0, 165, 255)

        cv2.putText(image, feedback, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f"Shoulder Angle: {int(shoulder_angle)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return shoulder_angle, feedback

class ShoulderPressAnalyzer(ExerciseAnalyzer):
    """숄더 프레스 분석기 (정면 기준, 양팔 동시 측정 및 카운팅)"""
    def __init__(self):
        self.count = 0         
        self.state = "UP"      

    def analyze(self, kpts, image):
        h, w, _ = image.shape
        
        # YOLO COCO Keypoints
        # 0: Nose
        # Left -> 5: Shoulder, 7: Elbow, 9: Wrist, 11: Hip
        # Right -> 6: Shoulder, 8: Elbow, 10: Wrist, 12: Hip
        nose = [kpts[0][0], kpts[0][1]]

        l_sh = [kpts[5][0], kpts[5][1]]
        l_el = [kpts[7][0], kpts[7][1]]
        l_wr = [kpts[9][0], kpts[9][1]]
        l_hip = [kpts[11][0], kpts[11][1]]
        
        r_sh = [kpts[6][0], kpts[6][1]]
        r_el = [kpts[8][0], kpts[8][1]]
        r_wr = [kpts[10][0], kpts[10][1]]
        r_hip = [kpts[12][0], kpts[12][1]]

        l_pts = [(int(l_sh[0]*w), int(l_sh[1]*h)), (int(l_el[0]*w), int(l_el[1]*h)), 
                 (int(l_wr[0]*w), int(l_wr[1]*h)), (int(l_hip[0]*w), int(l_hip[1]*h))]
        r_pts = [(int(r_sh[0]*w), int(r_sh[1]*h)), (int(r_el[0]*w), int(r_el[1]*h)), 
                 (int(r_wr[0]*w), int(r_wr[1]*h)), (int(r_hip[0]*w), int(r_hip[1]*h))]
        nose_pt = (int(nose[0]*w), int(nose[1]*h))

        # 시각화 (뼈대)
        cv2.line(image, l_pts[3], l_pts[0], (0, 255, 0), 3)
        cv2.line(image, l_pts[0], l_pts[1], (0, 255, 0), 3)
        cv2.line(image, l_pts[1], l_pts[2], (0, 255, 0), 3)
        cv2.line(image, r_pts[3], r_pts[0], (255, 0, 0), 3)
        cv2.line(image, r_pts[0], r_pts[1], (255, 0, 0), 3)
        cv2.line(image, r_pts[1], r_pts[2], (255, 0, 0), 3)
        
        for pt in l_pts + r_pts + [nose_pt]:
            cv2.circle(image, pt, 8, (0, 0, 255), -1)

        # 1. 팔꿈치 각도 
        l_elbow_angle = self.calculate_angle(l_sh, l_el, l_wr)
        r_elbow_angle = self.calculate_angle(r_sh, r_el, r_wr)

        # 2. 몸통 기준 어깨 각도
        l_shoulder_angle = self.calculate_angle(l_hip, l_sh, l_el)
        r_shoulder_angle = self.calculate_angle(r_hip, r_sh, r_el)

        # 3. 몸통(얼굴~골반) 휨 정도 계산
        mid_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
        vertical_ref = [mid_hip[0], mid_hip[1] - 0.1]
        trunk_angle = self.calculate_angle(vertical_ref, mid_hip, nose)

        # 상태 판별 및 카운팅 로직
        is_correct_posture = True
        feedback = "Good Form!"
        color = (0, 255, 0)

        if trunk_angle > 15: 
            feedback = "Warning: Keep your body straight!"
            color = (0, 165, 255)
            is_correct_posture = False
        elif abs(l_elbow_angle - r_elbow_angle) > 30: 
            feedback = "Warning: Balance your arms!"
            color = (0, 0, 255)
            is_correct_posture = False
        elif l_shoulder_angle < 70 or r_shoulder_angle < 70: 
            feedback = "Warning: Don't go too low!"
            color = (0, 0, 255)
            is_correct_posture = False
        elif (l_shoulder_angle < 120 and l_elbow_angle > 140) or (r_shoulder_angle < 120 and r_elbow_angle > 140):
            feedback = "Warning: Bend elbows at bottom!"
            color = (0, 0, 255)
            is_correct_posture = False

        if is_correct_posture:
            is_down_pose = (70 <= l_shoulder_angle <= 120) and (70 <= l_elbow_angle <= 120) and \
                           (70 <= r_shoulder_angle <= 120) and (70 <= r_elbow_angle <= 120)
            is_up_pose = (l_shoulder_angle >= 150 and l_elbow_angle >= 150 and \
                          r_shoulder_angle >= 150 and r_elbow_angle >= 150)

            if is_down_pose:
                if self.state == "UP":
                    self.state = "DOWN"
                feedback = "Ready... Push UP!"
                color = (0, 255, 255) 
                
            elif is_up_pose:
                if self.state == "DOWN":
                    self.state = "UP"
                    self.count += 1 
                feedback = "Perfect! Keep going!"
                color = (255, 0, 0)  

        cv2.putText(image, feedback, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f"L Elbow: {int(l_elbow_angle)} | L Shld: {int(l_shoulder_angle)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"R Elbow: {int(r_elbow_angle)} | R Shld: {int(r_shoulder_angle)}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, f"Count: {self.count}", (w - 280, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        return (l_elbow_angle + r_elbow_angle) / 2, feedback

# ==========================================
# 2. ROS2 메인 노드 (YOLOv11 적용)
# ==========================================
class PoseTrackingNode(Node):
    def __init__(self, exercise_type='shoulder_press'):
        super().__init__('pose_tracking_node')
        self.bridge = CvBridge()
        
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        
        # YOLOv11 Pose 모델 로드
        self.model = YOLO('yolov8n-pose.pt') # 또는 'yolo11n-pose.pt' 명칭에 맞게 사용
        
        self.analyzers = {
            'shoulder_flexion': ShoulderFlexionAnalyzer(),
            'shoulder_press': ShoulderPressAnalyzer()
        }
        self.current_analyzer = self.analyzers.get(exercise_type, ShoulderPressAnalyzer())

        self.get_logger().info(f"🚀 [{exercise_type}] YOLOv11 트레이닝 모드로 비전 노드가 시작되었습니다.")

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # YOLO 추론
            results = self.model(image, verbose=False, device='cpu')[0]

            # Keypoints가 인식되었을 때만 처리
            if results.keypoints is not None and len(results.keypoints.xyn) > 0:
                # xyn[0]는 정규화된(0~1) 첫 번째 사람의 키포인트 좌표입니다.
                kpts = results.keypoints.xyn[0].cpu().numpy()
                
                # YOLO가 17개의 관절을 모두 감지했는지 확인
                if len(kpts) >= 13: 
                    target_angle, feedback = self.current_analyzer.analyze(kpts, image)
                    
                    angle_msg = Float32()
                    angle_msg.data = float(target_angle)
                    self.angle_pub.publish(angle_msg)

            cv2.imshow('YOLOv11 PT Trainer', image)
            cv2.waitKey(1)
            
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