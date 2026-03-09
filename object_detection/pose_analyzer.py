import os
import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32, String # [추가] String 타입 임포트
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

from std_srvs.srv import SetBool, Trigger

# ==========================================
# 1. 상태 추적 모듈 (파일 저장 기능 제거, Publish 전용)
# ==========================================
class ExerciseSessionTracker:
    def __init__(self, exercise_type, publish_callback):
        self.exercise_type = exercise_type
        self.publish_callback = publish_callback # [추가] 퍼블리시용 콜백 함수
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
        self.emit_data()

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

        # 15프레임마다 토픽 퍼블리시 (실시간 업데이트 효과)
        if self.frame_count % 15 == 0:
            self.emit_data()

    def increment_rep(self, rep_count):
        self.rep_count = int(rep_count)
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.emit_data()

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

    def emit_data(self):
        """[수정] 파일 저장 대신 딕셔너리를 생성하여 콜백으로 전달"""
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
        
        if self.publish_callback:
            self.publish_callback(data)


# ==========================================
# 2. 운동 분석기 모듈
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
    def __init__(self, publish_callback): # [수정] 콜백 함수 받음
        self.count = 0
        self.state = "UP"
        self.tracker = ExerciseSessionTracker("shoulder_press", publish_callback)

    def analyze(self, kpts, image):
        # (기존 analyze 로직과 완벽히 동일하여 생략 없이 유지)
        h, w, _ = image.shape
        nose = [kpts[0][0], kpts[0][1]]
        
        l_sh, l_el, l_wr, l_hip = [kpts[5][0], kpts[5][1]], [kpts[7][0], kpts[7][1]], [kpts[9][0], kpts[9][1]], [kpts[11][0], kpts[11][1]]
        r_sh, r_el, r_wr, r_hip = [kpts[6][0], kpts[6][1]], [kpts[8][0], kpts[8][1]], [kpts[10][0], kpts[10][1]], [kpts[12][0], kpts[12][1]]

        l_pts = [(int(l_sh[0] * w), int(l_sh[1] * h)), (int(l_el[0] * w), int(l_el[1] * h)), (int(l_wr[0] * w), int(l_wr[1] * h)), (int(l_hip[0] * w), int(l_hip[1] * h))]
        r_pts = [(int(r_sh[0] * w), int(r_sh[1] * h)), (int(r_el[0] * w), int(r_el[1] * h)), (int(r_wr[0] * w), int(r_wr[1] * h)), (int(r_hip[0] * w), int(r_hip[1] * h))]
        nose_pt = (int(nose[0] * w), int(nose[1] * h))

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
            feedback = "Warning: Keep your body straight!"; color = (0, 165, 255); is_correct_posture = False
        elif abs(l_elbow_angle - r_elbow_angle) > 30:
            feedback = "Warning: Balance your arms!"; color = (0, 0, 255); is_correct_posture = False
        elif l_shoulder_angle < 70 or r_shoulder_angle < 70:
            feedback = "Warning: Don't go too low!"; color = (0, 0, 255); is_correct_posture = False
        elif (l_shoulder_angle < 120 and l_elbow_angle > 140) or (r_shoulder_angle < 120 and r_elbow_angle > 140):
            feedback = "Warning: Bend elbows at bottom!"; color = (0, 0, 255); is_correct_posture = False

        if is_correct_posture:
            is_down_pose = (70 <= l_shoulder_angle <= 120) and (70 <= l_elbow_angle <= 120) and \
                           (70 <= r_shoulder_angle <= 120) and (70 <= r_elbow_angle <= 120)
            is_up_pose = (l_shoulder_angle >= 150 and l_elbow_angle >= 150 and \
                          r_shoulder_angle >= 150 and r_elbow_angle >= 150)

            if is_down_pose:
                if self.state == "UP": self.state = "DOWN"
                feedback = "Ready... Push UP!"; color = (0, 255, 255)
            elif is_up_pose:
                if self.state == "DOWN":
                    self.state = "UP"
                    self.count += 1
                    self.tracker.increment_rep(self.count)
                feedback = "Perfect! Keep going!"; color = (255, 0, 0)

        self.tracker.update_frame(avg_elbow_angle, avg_shoulder_angle, trunk_angle, feedback, is_correct_posture)

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
        self.is_exercising = False
        self.publish_trigger = False
        
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        
        self.target_3d_pub = self.create_publisher(Point, '/target_correction_3d', 10)
        
        # [추가] 운동 결과 데이터를 쏠 퍼블리셔 생성
        self.result_pub = self.create_publisher(String, '/exercise_result', 10)
        
        self.model = YOLO('yolo11n-pose.pt')
        
        # [추가] 퍼블리시 콜백 함수를 Analyzer에 전달
        self.current_analyzer = ShoulderPressAnalyzer(publish_callback=self.publish_result_cb)

        self.depth_frame = None
        self.intrinsics = None

        self.srv_set_exercise = self.create_service(SetBool, '/set_exercise_state', self.set_exercise_cb)
        self.srv_publish_3d = self.create_service(Trigger, '/publish_target_3d', self.publish_3d_cb)

        self.get_logger().info(f"🚀 YOLO 자세 분석 및 3D 보정 노드 시작. (대기 상태)")

    # [추가] Tracker에서 넘어온 데이터를 String 토픽으로 발행
    def publish_result_cb(self, data_dict):
        msg = String()
        msg.data = json.dumps(data_dict, ensure_ascii=False)
        self.result_pub.publish(msg)

    def set_exercise_cb(self, request, response):
        self.is_exercising = request.data
        if self.is_exercising:
            self.current_analyzer.tracker.reset() 
            self.current_analyzer.count = 0      
            self.current_analyzer.state = "UP"
            msg = "✅ 운동 분석을 시작합니다."
        else:
            self.current_analyzer.tracker.emit_data() # 종료 시 최종 기록 한 번 더 발행
            msg = "⏸️ 운동 분석을 멈추고 대기(IDLE) 상태로 전환합니다."
        
        self.get_logger().info(msg)
        response.success = True
        response.message = msg
        return response

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
                    if self.is_exercising:
                        target_angle, feedback = self.current_analyzer.analyze(kpts, image)
                    else:
                        cv2.putText(image, "IDLE MODE - Waiting to start", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    h, w, _ = image.shape
                    l_wr, r_wr = [kpts[9][0], kpts[9][1]], [kpts[10][0], kpts[10][1]]
                    mid_x, mid_y = int((l_wr[0] * w + r_wr[0] * w) / 2), int((l_wr[1] * h + r_wr[1] * h) / 2)
                    cv2.circle(image, (mid_x, mid_y), 10, (0, 255, 255), -1)

                    if self.intrinsics is not None and mid_x > 0 and mid_y > 0:
                        cz = self._get_depth(mid_x, mid_y)
                        if cz is not None and cz > 0:
                            cx, cy, cz = self._pixel_to_camera_coords(mid_x, mid_y, cz)
                            target_3d_coord = (cx, cy, cz)
                            cv2.putText(image, f"Target 3D X:{int(cx)} Y:{int(cy)} Z:{int(cz)}", 
                                        (mid_x - 70, mid_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow('YOLO PT Trainer', image)
            cv2.waitKey(1)
            
            if self.publish_trigger:
                if target_3d_coord is not None:
                    point_msg = Point(x=float(target_3d_coord[0]), y=float(target_3d_coord[1]), z=float(target_3d_coord[2]))
                    self.target_3d_pub.publish(point_msg)
                    self.get_logger().info(f"✅ [교정 목표점] 좌표 발행: X:{int(target_3d_coord[0])}, Y:{int(target_3d_coord[1])}, Z:{int(target_3d_coord[2])}")
                else:
                    self.get_logger().warn("⚠️ 유효한 3D 좌표가 없어 퍼블리시하지 못했습니다.")
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