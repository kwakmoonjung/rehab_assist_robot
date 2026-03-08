import os
import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LOG_FILE = os.path.expanduser("~/exercise_session_log.json")


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


class ExerciseAnalyzer:
    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def analyze(self, landmarks, image):
        raise NotImplementedError


class ShoulderFlexionAnalyzer(ExerciseAnalyzer):
    def analyze(self, landmarks, image):
        h, w, _ = image.shape
        shoulder = [landmarks[12].x, landmarks[12].y]
        elbow = [landmarks[14].x, landmarks[14].y]
        wrist = [landmarks[16].x, landmarks[16].y]
        hip = [landmarks[24].x, landmarks[24].y]

        pts = {
            "shoulder": (int(shoulder[0] * w), int(shoulder[1] * h)),
            "elbow": (int(elbow[0] * w), int(elbow[1] * h)),
            "wrist": (int(wrist[0] * w), int(wrist[1] * h)),
            "hip": (int(hip[0] * w), int(hip[1] * h)),
        }

        cv2.line(image, pts["hip"], pts["shoulder"], (255, 0, 0), 3)
        cv2.line(image, pts["shoulder"], pts["elbow"], (0, 255, 0), 3)
        cv2.line(image, pts["elbow"], pts["wrist"], (0, 255, 0), 3)
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
        cv2.putText(
            image,
            f"Shoulder Angle: {int(shoulder_angle)}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return shoulder_angle, feedback


class BicepCurlAnalyzer(ExerciseAnalyzer):
    def __init__(self):
        self.count = 0
        self.state = "DOWN"
        self.logger = ExerciseSessionLogger(LOG_FILE, "bicep_curl")

        # ===== 기준값 =====
        self.TRUNK_THRESHOLD = 15
        self.BALANCE_THRESHOLD = 20
        self.UPPER_ARM_THRESHOLD = 25
        self.DOWN_ELBOW_ANGLE = 145
        self.UP_ELBOW_ANGLE = 60

    def _is_visible(self, landmark, threshold=0.5):
        return getattr(landmark, "visibility", 1.0) >= threshold

    def analyze(self, landmarks, image):
        h, w, _ = image.shape

        required_ids = [0, 11, 12, 13, 14, 15, 16, 23, 24]
        if not all(self._is_visible(landmarks[i]) for i in required_ids):
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
            return 0.0, feedback

        nose = [landmarks[0].x, landmarks[0].y]

        l_sh = [landmarks[11].x, landmarks[11].y]
        l_el = [landmarks[13].x, landmarks[13].y]
        l_wr = [landmarks[15].x, landmarks[15].y]
        l_hip = [landmarks[23].x, landmarks[23].y]

        r_sh = [landmarks[12].x, landmarks[12].y]
        r_el = [landmarks[14].x, landmarks[14].y]
        r_wr = [landmarks[16].x, landmarks[16].y]
        r_hip = [landmarks[24].x, landmarks[24].y]

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
        mid_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
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
                and l_upper_arm_angle <= self.UPPER_ARM_THRESHOLD + 10
                and r_upper_arm_angle <= self.UPPER_ARM_THRESHOLD + 10
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
                    feedback = "Lower slowly!"
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
            f"Trunk: {int(trunk_angle)}",
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

        return avg_elbow_angle, feedback


class PoseTrackingNode(Node):
    def __init__(self, exercise_type="bicep_curl"):
        super().__init__("pose_tracking_node")
        self.bridge = CvBridge()

        self.create_subscription(Image, "/camera/camera/color/image_raw", self.image_callback, 10)
        self.angle_pub = self.create_publisher(Float32, "/patient_elbow_angle", 10)

        model_path = "/home/rokey/cobot_ws/src/cobot2_ws/rehab_assist_robot/object_detection/pose_landmarker_lite.task"
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.analyzers = {
            "shoulder_flexion": ShoulderFlexionAnalyzer(),
            "bicep_curl": BicepCurlAnalyzer(),
        }
        self.current_analyzer = self.analyzers.get(exercise_type, BicepCurlAnalyzer())

        self.get_logger().info(f"[{exercise_type}] pose tracking node started.")
        self.get_logger().info(f"exercise log path: {LOG_FILE}")

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = self.detector.detect(mp_image)

            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                target_angle, feedback = self.current_analyzer.analyze(landmarks, image)

                angle_msg = Float32()
                angle_msg.data = float(target_angle)
                self.angle_pub.publish(angle_msg)

            cv2.imshow("MediaPipe PT Trainer", image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"에러 발생: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PoseTrackingNode(exercise_type="bicep_curl")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()