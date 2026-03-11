import numpy as np
import cv2
from datetime import datetime

class ExerciseSessionTracker:
    def __init__(self, exercise_type, publish_callback):
        self.exercise_type = exercise_type
        self.publish_callback = publish_callback # [수정] 콜백 추가
        self.reset()

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def reset(self):
        now_dt = datetime.now()
        self.session_start_dt = now_dt
        self.session_started_at = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        self.last_updated_at = self.session_started_at
        self.rep_count = 0
        self.frame_count = 0
        self.good_frame_count = 0
        self.trunk_angle_sum = 0.0
        self.last_feedback = "No data yet"
        
        self.max_rom_left = 0.0   
        self.max_rom_right = 0.0  
        self.rep_start_time = None
        self.rep_durations = []   
        self.tremor_count = 0     
        self.last_l_wr_y = None   
        self.last_r_wr_y = None
        
        self.successful_peaks = [] 
        self.all_peaks = []        
        self.l_z_history = []
        self.r_z_history = []
        self.max_z_drift = 0.0
        self.pure_arom = 0.0
        self.warning_counts = {
            "lean_back_momentum": 0, "chest_down": 0, "arms_too_high": 0, "arm_balance_issue": 0,
        }
        self.emit_data()

    def update_frame(self, l_shoulder_angle, r_shoulder_angle, trunk_angle, feedback, is_correct, l_wr_y=None, r_wr_y=None):
        self.frame_count += 1
        if is_correct:
            self.good_frame_count += 1
            current_max_shoulder = max(l_shoulder_angle, r_shoulder_angle)
            if current_max_shoulder > self.pure_arom:
                self.pure_arom = round(float(current_max_shoulder), 2)

        self.trunk_angle_sum += float(trunk_angle)

        if l_shoulder_angle > self.max_rom_left: self.max_rom_left = round(float(l_shoulder_angle), 2)
        if r_shoulder_angle > self.max_rom_right: self.max_rom_right = round(float(r_shoulder_angle), 2)

        # 떨림 감지 등 기존 로직 유지
        if l_wr_y is not None and r_wr_y is not None:
            if self.last_l_wr_y is not None:
                if abs(l_wr_y - self.last_l_wr_y) > 15 or abs(r_wr_y - self.last_r_wr_y) > 15:
                    if l_shoulder_angle > 30 or r_shoulder_angle > 30: self.tremor_count += 1
            self.last_l_wr_y = l_wr_y; self.last_r_wr_y = r_wr_y

        self.last_feedback = feedback
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 경고 카운팅 간소화 생략
        if self.frame_count % 15 == 0: self.emit_data()

    def update_depth(self, l_z, r_z):
        if l_z is not None and l_z > 0: self.l_z_history.append(l_z)
        if r_z is not None and r_z > 0: self.r_z_history.append(r_z)

    def increment_rep(self, rep_count):
        self.rep_count = int(rep_count)
        now = datetime.now()
        if self.rep_start_time is not None:
            duration = (now - self.rep_start_time).total_seconds()
            self.rep_durations.append(round(duration, 2))
        self.rep_start_time = now 
        
        if self.l_z_history and self.r_z_history:
            current_drift = max(max(self.l_z_history) - min(self.l_z_history), max(self.r_z_history) - min(self.r_z_history))
            if current_drift > self.max_z_drift: self.max_z_drift = round(current_drift, 2)
        
        self.l_z_history, self.r_z_history = [], []
        self.last_updated_at = now.strftime("%Y-%m-%d %H:%M:%S")
        self.emit_data()

    def emit_data(self): # [수정] save -> emit_data
        good_ratio = round((self.good_frame_count / self.frame_count) * 100.0, 2) if self.frame_count > 0 else 0.0
        
        data = {
            "exercise_type": self.exercise_type,
            "rep_count": self.rep_count,
            "performance_stats": {
                "total_frames": self.frame_count,
                "good_posture_ratio": good_ratio,
            },
            "warning_counts": self.warning_counts,
            "last_feedback": self.last_feedback,
        }

        if self.publish_callback:
            self.publish_callback(data)


class LateralRaiseAnalyzer:
    def __init__(self, publish_callback): # [수정] 콜백 추가
        self.count = 0         
        self.state = "DOWN"    
        self.current_rep_peak = 0.0 
        self.eval_feedback = ""     
        self.eval_color = (0, 255, 0)
        self.rep_has_warning = False 
        self.tracker = ExerciseSessionTracker("lateral_raise", publish_callback)

    def analyze_dual(self, front_kpts, side_kpts, front_img, side_img):
        h, w, _ = front_img.shape

        # --- [1. 정면 카메라 데이터 처리] ---
        l_sh_f = front_kpts[5][:2]
        l_el_f = front_kpts[7][:2]
        l_hip_f = front_kpts[11][:2]
        l_wr_f = front_kpts[9][:2]   

        r_sh_f = front_kpts[6][:2]
        r_el_f = front_kpts[8][:2]
        r_hip_f = front_kpts[12][:2]
        r_wr_f = front_kpts[10][:2]  

        pts_f = {
            'l_sh': (int(l_sh_f[0]), int(l_sh_f[1])), 'l_el': (int(l_el_f[0]), int(l_el_f[1])),
            'l_hip': (int(l_hip_f[0]), int(l_hip_f[1])), 'r_sh': (int(r_sh_f[0]), int(r_sh_f[1])),
            'r_el': (int(r_el_f[0]), int(r_el_f[1])), 'r_hip': (int(r_hip_f[0]), int(r_hip_f[1]))
        }

        cv2.line(front_img, pts_f['l_hip'], pts_f['l_sh'], (0, 255, 0), 3)
        cv2.line(front_img, pts_f['l_sh'], pts_f['l_el'], (0, 255, 0), 3)
        cv2.line(front_img, pts_f['r_hip'], pts_f['r_sh'], (255, 0, 0), 3)
        cv2.line(front_img, pts_f['r_sh'], pts_f['r_el'], (255, 0, 0), 3)
        for pt in pts_f.values(): cv2.circle(front_img, pt, 8, (0, 0, 255), -1)

        l_shoulder_angle = self.calculate_angle(l_hip_f, l_sh_f, l_el_f)
        r_shoulder_angle = self.calculate_angle(r_hip_f, r_sh_f, r_el_f)
        avg_shoulder_angle = (l_shoulder_angle + r_shoulder_angle) / 2.0

        # --- [2. 측면 카메라 데이터 처리] ---
        l_sh_s = side_kpts[5][:2]
        l_hip_s = side_kpts[11][:2]
        l_knee_s = side_kpts[13][:2]

        pts_s = {
            'l_sh': (int(l_sh_s[0]), int(l_sh_s[1])),
            'l_hip': (int(l_hip_s[0]), int(l_hip_s[1])),
            'l_knee': (int(l_knee_s[0]), int(l_knee_s[1]))
        }

        cv2.line(side_img, pts_s['l_hip'], pts_s['l_sh'], (0, 255, 255), 3)
        cv2.line(side_img, pts_s['l_hip'], pts_s['l_knee'], (0, 255, 255), 3)
        for pt in pts_s.values(): cv2.circle(side_img, pt, 8, (0, 0, 255), -1)

        trunk_side_angle = self.calculate_angle(l_sh_s, l_hip_s, l_knee_s)

        # --- [3. 통합 상태 판별 및 카운팅] ---
        is_correct_posture = True
        feedback = "Good Form!"
        color = (0, 255, 0) 

        if trunk_side_angle > 185: 
            feedback = "Warning: Don't lean back! No momentum."
            color = (0, 0, 255)
            is_correct_posture = False
        elif trunk_side_angle < 150: 
            feedback = "Warning: Keep your chest up!"
            color = (0, 165, 255)
            is_correct_posture = False
        elif l_shoulder_angle > 100 or r_shoulder_angle > 100: 
            feedback = "Warning: Arms too high! Lower them."
            color = (0, 0, 255)
            is_correct_posture = False
        elif abs(l_shoulder_angle - r_shoulder_angle) > 20: 
            feedback = "Warning: Balance your arms!"
            color = (0, 165, 255)
            is_correct_posture = False

        if not is_correct_posture:
            self.rep_has_warning = True

        # 3단계 상태 머신 적용 (DOWN -> RAISING -> LOWERING)
        is_down_pose = (l_shoulder_angle < 40) and (r_shoulder_angle < 40)

        if self.state == "DOWN":
            if avg_shoulder_angle >= 40:
                self.state = "RAISING"
                self.current_rep_peak = avg_shoulder_angle
                self.rep_has_warning = False
                if self.tracker.rep_start_time is None:
                    self.tracker.rep_start_time = datetime.now()
                feedback = "Keep going up!"
                color = (0, 255, 0)
            else:
                feedback = "Ready... Raise your arms!"
                color = (0, 255, 255)

        elif self.state == "RAISING":
            # 실시간 최고점 갱신
            if avg_shoulder_angle > self.current_rep_peak:
                self.current_rep_peak = avg_shoulder_angle

            # ⭐️ 하강 시작점 포착 (최고점에서 5도 이상 떨어지는 순간) ⭐️
            if avg_shoulder_angle < self.current_rep_peak - 5.0:
                self.state = "LOWERING"
                
                # [기존] 이 찰나의 순간에 즉시 평가 및 DB 저장 진행 (성공한 횟수)
                if not self.rep_has_warning and self.current_rep_peak >= 70.0:
                    self.tracker.successful_peaks.append(round(float(self.current_rep_peak), 2))
                    
                # [신규] 자세 붕괴 및 경고 여부와 상관없이 무조건 1회 사이클 최고점 기록 (전체 평균용)
                if self.current_rep_peak >= 40.0:
                    self.tracker.all_peaks.append(round(float(self.current_rep_peak), 2))
                
                if self.current_rep_peak >= 80:
                    self.count += 1
                    self.tracker.increment_rep(self.count)
                    self.eval_feedback = "Perfect! Great job."
                    self.eval_color = (0, 255, 0)
                else:
                    needed_angle = 80 - int(self.current_rep_peak)
                    self.eval_feedback = f"Good (Peak: {int(self.current_rep_peak)}). Try {needed_angle} deg more!"
                    self.eval_color = (0, 255, 255)
                
                feedback = "Now slowly lower arms."
                color = (255, 165, 0)
            else:
                if self.current_rep_peak >= 80:
                    feedback = "Perfect height! Now slowly down."
                    color = (255, 0, 0)
                elif is_correct_posture:
                    feedback = "Keep going up!"
                    color = (0, 255, 0)

        elif self.state == "LOWERING":
            if is_down_pose:
                self.state = "DOWN"
                self.current_rep_peak = 0.0 # 다음 횟수를 위해 리셋
                self.eval_feedback = ""     # 피드백 초기화
                
            # 내려가는 동안에는 평가 피드백을 계속 유지하여 보여줌
            if self.eval_feedback != "" and is_correct_posture:
                feedback = self.eval_feedback
                color = self.eval_color
            elif is_correct_posture:
                feedback = "Slowly down..."
                color = (255, 165, 0)

        self.tracker.update_frame(
            l_shoulder_angle=l_shoulder_angle,
            r_shoulder_angle=r_shoulder_angle,
            trunk_angle=trunk_side_angle,
            feedback=feedback,
            is_correct=is_correct_posture,
            l_wr_y=l_wr_f[1],
            r_wr_y=r_wr_f[1]
        )

        cv2.putText(front_img, feedback, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(front_img, f"Front(L/R): {int(l_shoulder_angle)} / {int(r_shoulder_angle)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(front_img, f"Count: {self.count}", (w - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        cv2.putText(side_img, f"[SIDE VIEW]", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(side_img, f"Trunk Angle: {int(trunk_side_angle)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return avg_shoulder_angle, feedback, (int(l_wr_f[0]), int(l_wr_f[1])), (int(r_wr_f[0]), int(r_wr_f[1]))

