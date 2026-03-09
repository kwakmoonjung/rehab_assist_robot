import cv2
import numpy as np
from datetime import datetime

class ExerciseSessionTracker:
    def __init__(self, exercise_type, publish_callback):
        self.exercise_type = exercise_type
        self.publish_callback = publish_callback # [수정] 콜백 함수 받음
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
        """파일 저장 대신 딕셔너리를 생성하여 콜백으로 전달"""
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


class ShoulderPressAnalyzer:
    def __init__(self, publish_callback): 
        self.count = 0
        self.state = "UP"
        self.tracker = ExerciseSessionTracker("shoulder_press", publish_callback)

    def calculate_angle(self, a, b, c):
        """내부에서 사용하기 위해 공통 각도 계산기를 클래스 내부에 포함시켰습니다."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def analyze(self, kpts, image):
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