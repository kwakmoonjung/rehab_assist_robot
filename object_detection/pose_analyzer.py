import os
import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

# [추가] 서비스 통신을 위한 메시지 임포트
from std_srvs.srv import SetBool, Trigger

# [원래 경로 유지]
LOG_FILE = os.path.expanduser("~/exercise_session_log.json")

# ==========================================
# 1. JSON 로깅 모듈
# ==========================================
class ExerciseSessionLogger:
    def __init__(self, log_file, exercise_type):
        self.log_file = log_file
        self.exercise_type = exercise_type
        self.reset()

    def reset(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_started_at = now
        self.last_updated_at = now
        self.rep_count = 0
        self.frame_count = 0
        self.good_frame_count = 0
        self.elbow_angle_sum = 0.0
        self.shoulder_angle_sum = 0.0
        self.trunk_angle_sum = 0.0
        self.last_feedback = "No data yet"
        self.warning_counts = {
            "body_not_straight": 0,
            "arm_balance_issue": 0,
            "too_low": 0,
            "bend_elbows_at_bottom": 0,
        }
        self.save()

    def update_frame(self, elbow_angle, shoulder_angle, trunk_angle, feedback, is_correct):
        self.frame_count += 1
        if is_correct:
            self.good_frame_count += 1

        self.elbow_angle_sum += float(elbow_angle)
        self.shoulder_angle_sum += float(shoulder_angle)
        self.trunk_angle_sum += float(trunk_angle)
        self.last_feedback = feedback
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._count_warning(feedback)

        if self.frame_count % 15 == 0:
            self.save()

    def increment_rep(self, rep_count):
        self.rep_count = int(rep_count)
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def _count_warning(self, feedback):
        if feedback == "Warning: Keep your body straight!":
            self.warning_counts["body_not_straight"] += 1
        elif feedback == "Warning: Balance your arms!":
            self.warning_counts["arm_balance_issue"] += 1
        elif feedback == "Warning: Don't go too low!":
            self.warning_counts["too_low"] += 1
        elif feedback == "Warning: Bend elbows at bottom!":
            self.warning_counts["bend_elbows_at_bottom"] += 1

    def _safe_avg(self, value_sum):
        if self.frame_count == 0:
            return 0.0
        return round(value_sum / self.frame_count, 2)

    def save(self):
        good_ratio = 0.0
        if self.frame_count > 0:
            good_ratio = round((self.good_frame_count / self.frame_count) * 100.0, 2)

        data = {
            "exercise_type": self.exercise_type,
            "session_started_at": self.session_started_at,
            "last_updated_at": self.last_updated_at,
            "rep_count": self.rep_count,
            "frame_count": self.frame_count,
            "good_frame_count": self.good_frame_count,
            "good_posture_ratio": good_ratio,
            "avg_elbow_angle": self._safe_avg(self.elbow_angle_sum),
            "avg_shoulder_angle": self._safe_avg(self.shoulder_angle_sum),
            "avg_trunk_angle": self._safe_avg(self.trunk_angle_sum),
            "warning_counts": self.warning_counts,
            "last_feedback": self.last_feedback,
        }

        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"log save error: {e}")


# ==========================================
# 2. 운동 분석기 모듈 (YOLO Keypoints 적용)
# ==========================================
class ExerciseAnalyzer:
    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

class ShoulderPressAnalyzer(ExerciseAnalyzer):
    def __init__(self):
        self.count = 0
        self.state = "UP"
        self.logger = ExerciseSessionLogger(LOG_FILE, "shoulder_press")

    def analyze(self, kpts, image):
        h, w, _ = image.shape

        nose = [kpts[0][0], kpts[0][1]]
        
        l_sh = [kpts[5][0], kpts[5][1]]
        l_el = [kpts[7][0], kpts[7][1]]
        l_wr = [kpts[9][0], kpts[9][1]]
        l_hip = [kpts[11][0], kpts[11][1]]

        r_sh = [kpts[6][0], kpts[6][1]]
        r_el = [kpts[8][0], kpts[8][1]]
        r_wr = [kpts[10][0], kpts[10][1]]
        r_hip = [kpts[12][0], kpts[12][1]]

        l_pts = [
            (int(l_sh[0] * w), int(l_sh[1] * h)),
            (int(l_el[0] * w), int(l_el[1] * h)),
            (int(l_wr[0] * w), int(l_wr[1] * h)),
            (int(l_hip[0] * w), int(l_hip[1] * h)),
        ]
        r_pts = [
            (int(r_sh[0] * w), int(r_sh[1] * h)),
            (int(r_el[0] * w), int(r_el[1] * h)),
            (int(r_wr[0] * w), int(r_wr[1] * h)),
            (int(r_hip[0] * w), int(r_hip[1] * h)),
        ]
        nose_pt = (int(nose[0] * w), int(nose[1] * h))

        # 뼈대 시각화
        cv2.line(image, l_pts[3], l_pts[0], (0, 255, 0), 3)
        cv2.line(image, l_pts[0], l_pts[1], (0, 255, 0), 3)
        cv2.line(image, l_pts[1], l_pts[2], (0, 255, 0), 3)
        cv2.line(image, r_pts[3], r_pts[0], (255, 0, 0), 3)
        cv2.line(image, r_pts[0], r_pts[1], (255, 0, 0), 3)
        cv2.line(image, r_pts[1], r_pts[2], (255, 0, 0), 3)

        for pt in l_pts + r_pts + [nose_pt]:
            cv2.circle(image, pt, 8, (0, 0, 255), -1)

        l_elbow_angle = self.calculate_angle(l_sh, l_el, l_wr)
        r_elbow_angle = self.calculate_angle(r_sh, r_el, r_wr)

        l_shoulder_angle = self.calculate_angle(l_hip, l_sh, l_el)
        r_shoulder_angle = self.calculate_angle(r_hip, r_sh, r_el)

        mid_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
        vertical_ref = [mid_hip[0], mid_hip[1] - 0.1]
        trunk_angle = self.calculate_angle(vertical_ref, mid_hip, nose)

        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0
        avg_shoulder_angle = (l_shoulder_angle + r_shoulder_angle) / 2.0

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
                    self.logger.increment_rep(self.count)
                feedback = "Perfect! Keep going!"
                color = (255, 0, 0)

        self.logger.update_frame(avg_elbow_angle, avg_shoulder_angle, trunk_angle, feedback, is_correct_posture)

        cv2.putText(image, feedback, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f"L Elbow: {int(l_elbow_angle)} | L Shld: {int(l_shoulder_angle)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"R Elbow: {int(r_elbow_angle)} | R Shld: {int(r_shoulder_angle)}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, f"Count: {self.count}", (w - 280, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        return avg_elbow_angle, feedback


# ==========================================
# 3. ROS2 메인 노드 (YOLO + 3D 퍼블리시)
# ==========================================
class PoseAnalyzerNode(Node):
    def __init__(self, exercise_type='shoulder_press'):
        super().__init__('pose_analyzer_node')
        self.bridge = CvBridge()
        
        # [추가] 상태 관리 변수
        self.is_exercising = False
        self.publish_trigger = False
        
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        
        # self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        self.target_3d_pub = self.create_publisher(Point, '/target_correction_3d', 10)
        
        self.model = YOLO('yolo11n-pose.pt')
        self.current_analyzer = ShoulderPressAnalyzer()

        self.depth_frame = None
        self.intrinsics = None

        # [추가] 서비스 서버 2개 생성
        self.srv_set_exercise = self.create_service(SetBool, '/set_exercise_state', self.set_exercise_cb)
        self.srv_publish_3d = self.create_service(Trigger, '/publish_target_3d', self.publish_3d_cb)

        self.get_logger().info(f"🚀 YOLO 자세 분석 및 3D 보정 노드 시작. (대기 상태)")
        self.get_logger().info(f"📝 운동 로그 경로: {LOG_FILE}")

    # [추가] Service 1 콜백: 운동 분석 On/Off 제어
    def set_exercise_cb(self, request, response):
        self.is_exercising = request.data
        if self.is_exercising:
            self.current_analyzer.logger.reset() # 새 운동 시작 시 기록 초기화
            self.current_analyzer.count = 0      # 횟수 초기화
            self.current_analyzer.state = "UP"
            msg = "✅ 운동 분석을 시작합니다."
        else:
            self.current_analyzer.logger.save()  # 종료 시 최종 기록 저장
            msg = "⏸️ 운동 분석을 멈추고 대기(IDLE) 상태로 전환합니다."
        
        self.get_logger().info(msg)
        response.success = True
        response.message = msg
        return response

    # [추가] Service 2 콜백: 3D 교정 좌표 1회 발행
    def publish_3d_cb(self, request, response):
        self.publish_trigger = True
        msg = "🎯 3D 교정 좌표 발행 요청 수신됨."
        self.get_logger().info(msg)
        response.success = True
        response.message = msg
        return response

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
            
            target_3d_coord = None       
            mid_x, mid_y = -1, -1

            if results.keypoints is not None and len(results.keypoints.xyn) > 0:
                kpts = results.keypoints.xyn[0].cpu().numpy()
                
                if len(kpts) >= 13: 
                    # [수정] is_exercising 상태에 따른 분기 처리
                    if self.is_exercising:
                        # On 상태: 각도 분석, 카운트, 로깅 및 UI 업데이트
                        target_angle, feedback = self.current_analyzer.analyze(kpts, image)
                        
                        angle_msg = Float32()
                        angle_msg.data = float(target_angle)
                        # self.angle_pub.publish(angle_msg)
                    else:
                        # Off 상태: 대기 모드 UI 표시 (연산 및 로깅 쉼)
                        cv2.putText(image, "IDLE MODE - Waiting to start", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # 3D 교정 및 좌표 발행을 위해 중앙점 계산은 항상 수행 (스페이스바 대체용)
                    h, w, _ = image.shape
                    l_wr = [kpts[9][0], kpts[9][1]]
                    r_wr = [kpts[10][0], kpts[10][1]]
                    mid_x = int((l_wr[0] * w + r_wr[0] * w) / 2)
                    mid_y = int((l_wr[1] * h + r_wr[1] * h) / 2)
                    cv2.circle(image, (mid_x, mid_y), 10, (0, 255, 255), -1)

                    if self.intrinsics is not None:
                        # 중앙점 3D 좌표 변환
                        if mid_x > 0 and mid_y > 0:
                            cz = self._get_depth(mid_x, mid_y)
                            if cz is not None and cz > 0:
                                cx, cy, cz = self._pixel_to_camera_coords(mid_x, mid_y, cz)
                                target_3d_coord = (cx, cy, cz)
                                cv2.putText(image, f"Target 3D X:{int(cx)} Y:{int(cy)} Z:{int(cz)}", 
                                            (mid_x - 70, mid_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow('YOLO PT Trainer', image)
            cv2.waitKey(1) # [수정] 스페이스바 트리거 삭제
            
            # [추가] 서비스로 호출되었을 때만 퍼블리시 진행
            if self.publish_trigger:
                if target_3d_coord is not None:
                    point_msg = Point()
                    point_msg.x = float(target_3d_coord[0])
                    point_msg.y = float(target_3d_coord[1])
                    point_msg.z = float(target_3d_coord[2])
                    self.target_3d_pub.publish(point_msg)
                    self.get_logger().info(f"✅ [교정 목표점] 좌표 발행: X:{int(target_3d_coord[0])}, Y:{int(target_3d_coord[1])}, Z:{int(target_3d_coord[2])}")
                else:
                    self.get_logger().warn("⚠️ 유효한 3D 좌표가 없어 퍼블리시하지 못했습니다.")
                
                # 발행 여부와 상관없이 트리거는 1회 동작 후 다시 Off
                self.publish_trigger = False
            
        except Exception as e:
            self.get_logger().error(f"에러 발생: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PoseAnalyzerNode(exercise_type='shoulder_press') 
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()