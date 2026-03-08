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
        self.upper_arm_angle_sum = 0.0
        self.trunk_angle_sum = 0.0
        self.last_feedback = "No data yet"
        self.warning_counts = {
            "body_not_straight": 0,
            "arm_balance_issue": 0,
            "elbows_not_close_to_body": 0,
            "arms_not_visible": 0,
        }
        self.save()

    def update_frame(self, elbow_angle, upper_arm_angle, trunk_angle, feedback, is_correct):
        self.frame_count += 1
        if is_correct:
            self.good_frame_count += 1

        self.elbow_angle_sum += float(elbow_angle)
        self.upper_arm_angle_sum += float(upper_arm_angle)
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
        elif feedback == "Warning: Move both arms evenly!":
            self.warning_counts["arm_balance_issue"] += 1
        elif feedback == "Warning: Keep elbows close to body!":
            self.warning_counts["elbows_not_close_to_body"] += 1
        elif feedback == "Warning: Keep both arms visible!":
            self.warning_counts["arms_not_visible"] += 1

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
            "avg_upper_arm_angle": self._safe_avg(self.upper_arm_angle_sum),
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


class BicepCurlAnalyzer(ExerciseAnalyzer):
    def __init__(self):
        self.count = 0
        self.state = "DOWN"
        self.logger = ExerciseSessionLogger(LOG_FILE, "bicep_curl")

        # ===== 기준값 =====
        self.TRUNK_THRESHOLD = 15
        self.BALANCE_THRESHOLD = 20

        # 팔꿈치가 몸통에서 떨어졌는지 판단하는 기준
        # 여유롭게 잡아달라고 했던 값 반영
        self.UPPER_ARM_THRESHOLD = 45

        # 이두컬 자세 기준
        self.DOWN_ELBOW_ANGLE = 145
        self.UP_ELBOW_ANGLE = 60

        # YOLO 키포인트 신뢰도 기준
        self.KEYPOINT_CONF_THRESHOLD = 0.35

    def _is_visible(self, idx, kpt_conf):
        if kpt_conf is None:
            return True
        if idx >= len(kpt_conf):
            return False
        return float(kpt_conf[idx]) >= self.KEYPOINT_CONF_THRESHOLD

    def analyze(self, kpts, image, kpt_conf=None):
        h, w, _ = image.shape

        # COCO keypoint index
        # 0 nose
        # 5 left shoulder, 6 right shoulder
        # 7 left elbow, 8 right elbow
        # 9 left wrist, 10 right wrist
        # 11 left hip, 12 right hip
        required_ids = [0, 5, 6, 7, 8, 9, 10, 11, 12]

        if not all(self._is_visible(i, kpt_conf) for i in required_ids):
            feedback = "Warning: Keep both arms visible!"
            color = (0, 0, 255)

            self.logger.update_frame(
                elbow_angle=0.0,
                upper_arm_angle=0.0,
                trunk_angle=0.0,
                feedback=feedback,
                is_correct=False,
            )

            cv2.putText(image, feedback, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(
                image,
                f"Count: {self.count}",
                (w - 280, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                3,
            )
            return 0.0, feedback, None

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

        # 팔꿈치 각도
        l_elbow_angle = self.calculate_angle(l_sh, l_el, l_wr)
        r_elbow_angle = self.calculate_angle(r_sh, r_el, r_wr)

        # 상완이 몸통에서 얼마나 벌어졌는지
        l_upper_arm_angle = self.calculate_angle(l_hip, l_sh, l_el)
        r_upper_arm_angle = self.calculate_angle(r_hip, r_sh, r_el)

        # 몸통 기울기
        mid_hip = [(l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0]
        vertical_ref = [mid_hip[0], mid_hip[1] - 0.1]
        trunk_angle = self.calculate_angle(vertical_ref, mid_hip, nose)

        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0
        avg_upper_arm_angle = (l_upper_arm_angle + r_upper_arm_angle) / 2.0

        is_correct_posture = True
        feedback = "Good Form!"
        color = (0, 255, 0)

        # ===== 잘못된 자세 체크 =====
        if trunk_angle > self.TRUNK_THRESHOLD:
            feedback = "Warning: Keep your body straight!"
            color = (0, 165, 255)
            is_correct_posture = False

        elif abs(l_elbow_angle - r_elbow_angle) > self.BALANCE_THRESHOLD:
            feedback = "Warning: Move both arms evenly!"
            color = (0, 0, 255)
            is_correct_posture = False

        elif (
            l_upper_arm_angle > self.UPPER_ARM_THRESHOLD
            or r_upper_arm_angle > self.UPPER_ARM_THRESHOLD
        ):
            feedback = "Warning: Keep elbows close to body!"
            color = (0, 0, 255)
            is_correct_posture = False

        # ===== 상태 전환 및 카운팅 =====
        if is_correct_posture:
            is_down_pose = (
                l_elbow_angle >= self.DOWN_ELBOW_ANGLE
                and r_elbow_angle >= self.DOWN_ELBOW_ANGLE
                and l_upper_arm_angle <= self.UPPER_ARM_THRESHOLD
                and r_upper_arm_angle <= self.UPPER_ARM_THRESHOLD
            )

            is_up_pose = (
                l_elbow_angle <= self.UP_ELBOW_ANGLE
                and r_elbow_angle <= self.UP_ELBOW_ANGLE
                and l_upper_arm_angle <= self.UPPER_ARM_THRESHOLD + 15
                and r_upper_arm_angle <= self.UPPER_ARM_THRESHOLD + 15
            )

            if is_down_pose:
                if self.state == "UP":
                    self.state = "DOWN"
                feedback = "Ready... Curl UP!"
                color = (0, 255, 255)

            elif is_up_pose:
                if self.state == "DOWN":
                    self.state = "UP"
                    self.count += 1
                    self.logger.increment_rep(self.count)
                feedback = "Great Curl!"
                color = (255, 0, 0)

            else:
                if self.state == "DOWN":
                    feedback = "Curl the weights up!"
                else:
                    feedback = "Slowly lower your arms!"
                color = (255, 255, 255)

        self.logger.update_frame(
            elbow_angle=avg_elbow_angle,
            upper_arm_angle=avg_upper_arm_angle,
            trunk_angle=trunk_angle,
            feedback=feedback,
            is_correct=is_correct_posture,
        )

        cv2.putText(image, feedback, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(
            image,
            f"L Elbow: {int(l_elbow_angle)} | L UpperArm: {int(l_upper_arm_angle)}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            image,
            f"R Elbow: {int(r_elbow_angle)} | R UpperArm: {int(r_upper_arm_angle)}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            image,
            f"Body Tilt: {int(trunk_angle)}",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            image,
            f"Count: {self.count}",
            (w - 280, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            3,
        )

        # 교정 목표점: 양 손목의 중앙점
        mid_x = int((l_pts[2][0] + r_pts[2][0]) / 2)
        mid_y = int((l_pts[2][1] + r_pts[2][1]) / 2)
        cv2.circle(image, (mid_x, mid_y), 10, (0, 255, 255), -1)

        return avg_elbow_angle, feedback, (mid_x, mid_y)


# ==========================================
# 3. ROS2 메인 노드 (YOLO + 3D 퍼블리시)
# ==========================================
class PoseAnalyzerNode(Node):
    def __init__(self, exercise_type='bicep_curl'):
        super().__init__('pose_analyzer_node')
        self.bridge = CvBridge()

        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)

        self.angle_pub = self.create_publisher(Float32, '/patient_elbow_angle', 10)
        self.target_3d_pub = self.create_publisher(Point, '/target_correction_3d', 10)

        self.model = YOLO('yolo11n-pose.pt')
        self.current_analyzer = BicepCurlAnalyzer()

        self.depth_frame = None
        self.intrinsics = None

        self.get_logger().info("🚀 YOLO 이두컬 자세 분석 및 3D 보정 노드 시작. (스페이스바 누를 시 좌표 발행)")
        self.get_logger().info(f"📝 운동 로그 경로: {LOG_FILE}")
        self.get_logger().info(f"[{exercise_type}] pose analyzer node started.")

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {
                "fx": msg.k[0],
                "fy": msg.k[4],
                "ppx": msg.k[2],
                "ppy": msg.k[5],
            }

    def _get_depth(self, x, y):
        if self.depth_frame is None:
            return None
        try:
            return float(self.depth_frame[y, x])
        except IndexError:
            return None

    def _pixel_to_camera_coords(self, x, y, z):
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        ppx = self.intrinsics['ppx']
        ppy = self.intrinsics['ppy']
        return (
            (x - ppx) * z / fx,
            (y - ppy) * z / fy,
            z
        )

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.model(image, verbose=False, device='cpu')[0]

            target_3d_coord = None

            if results.keypoints is not None and len(results.keypoints.xyn) > 0:
                kpts = results.keypoints.xyn[0].cpu().numpy()

                kpt_conf = None
                if hasattr(results.keypoints, "conf") and results.keypoints.conf is not None:
                    if len(results.keypoints.conf) > 0:
                        kpt_conf = results.keypoints.conf[0].cpu().numpy()

                if len(kpts) >= 13:
                    target_angle, feedback, target_pixel = self.current_analyzer.analyze(kpts, image, kpt_conf)

                    angle_msg = Float32()
                    angle_msg.data = float(target_angle)
                    self.angle_pub.publish(angle_msg)

                    if target_pixel is not None and self.intrinsics is not None:
                        mid_x, mid_y = target_pixel

                        if mid_x > 0 and mid_y > 0:
                            cz = self._get_depth(mid_x, mid_y)
                            if cz is not None and cz > 0:
                                cx, cy, cz = self._pixel_to_camera_coords(mid_x, mid_y, cz)
                                target_3d_coord = (cx, cy, cz)
                                cv2.putText(
                                    image,
                                    f"Target 3D X:{int(cx)} Y:{int(cy)} Z:{int(cz)}",
                                    (mid_x - 80, mid_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 255),
                                    2,
                                )

            cv2.imshow('YOLO Bicep Curl Trainer', image)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:
                if target_3d_coord is not None:
                    point_msg = Point()
                    point_msg.x = float(target_3d_coord[0])
                    point_msg.y = float(target_3d_coord[1])
                    point_msg.z = float(target_3d_coord[2])
                    self.target_3d_pub.publish(point_msg)

                    self.get_logger().info(
                        f"✅ [교정 목표점] 좌표 발행: X:{int(target_3d_coord[0])}, "
                        f"Y:{int(target_3d_coord[1])}, Z:{int(target_3d_coord[2])}"
                    )
                else:
                    self.get_logger().warn("⚠️ 유효한 3D 좌표가 없습니다.")

        except Exception as e:
            self.get_logger().error(f"에러 발생: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PoseAnalyzerNode(exercise_type='bicep_curl')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()