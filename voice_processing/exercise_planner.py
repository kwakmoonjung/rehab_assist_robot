#!/usr/bin/env python3
import os
import json
from datetime import datetime
from collections import defaultdict

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv
from ament_index_python.packages import get_package_share_directory


# ==========================================
# 환경 설정
# ==========================================
PACKAGE_SHARE_DIR = get_package_share_directory("rehab_assist_robot")
RESOURCE_DIR = os.path.join(PACKAGE_SHARE_DIR, "resource")

ENV_PATH = os.path.join(RESOURCE_DIR, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

FIREBASE_KEY_PATH = os.path.join(RESOURCE_DIR, "serviceAccountKey.json")
FIREBASE_DB_URL = "https://rehab-aa1ee-default-rtdb.asia-southeast1.firebasedatabase.app/"


# ==========================================
# 운동 / 부위 정보
# ==========================================
TRACKED_EXERCISE_INFO = {
    "bicep_curl": {
        "name_kr": "이두 컬",
        "main_area": "arms",
        "main_area_kr": "팔",
        "sub_areas": ["팔", "이두", "전완"],
    },
    "shoulder_press": {
        "name_kr": "숄더 프레스",
        "main_area": "shoulders",
        "main_area_kr": "어깨",
        "sub_areas": ["어깨", "삼두", "상체 밀기"],
    },
    "lateral_raise": {
        "name_kr": "사이드 레터럴 레이즈",
        "main_area": "shoulders",
        "main_area_kr": "어깨",
        "sub_areas": ["측면 어깨", "어깨 안정성"],
    },
}

AREA_LABELS = {
    "arms": "팔",
    "shoulders": "어깨",
    "unknown": "알 수 없음",
}

WARNING_LABELS = {
    "lean_back_momentum": "허리 반동",
    "elbow_flare": "팔꿈치 벌어짐",
    "asymmetry": "좌우 불균형",
    "range_limit": "가동 범위 부족",
    "shoulder_shrug": "어깨 들림",
}


class ExercisePlanner(Node):
    def __init__(self):
        super().__init__("exercise_planner")

        self._init_firebase()

        self.request_sub = self.create_subscription(
            String,
            "/exercise_planner/request",
            self.request_callback,
            10
        )

        self.response_pub = self.create_publisher(
            String,
            "/exercise_planner/response",
            10
        )

        self.get_logger().info(f"📁 resource 경로: {RESOURCE_DIR}")
        self.get_logger().info(f"📁 Firebase Key 경로: {FIREBASE_KEY_PATH}")
        self.get_logger().info("📡 exercise_planner 노드 시작")

    # ==========================================
    # 초기화
    # ==========================================
    def _init_firebase(self):
        try:
            if not os.path.exists(FIREBASE_KEY_PATH):
                raise FileNotFoundError(
                    f"serviceAccountKey.json 파일이 없습니다: {FIREBASE_KEY_PATH}"
                )

            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(
                    cred,
                    {"databaseURL": FIREBASE_DB_URL}
                )

            self.get_logger().info("🔥 Firebase 연결 완료")

        except Exception as e:
            self.get_logger().error(f"Firebase 초기화 실패: {e}")

    # ==========================================
    # 공용 유틸
    # ==========================================
    def to_int(self, value, default=0):
        try:
            return int(value)
        except Exception:
            return default

    def warning_key_to_korean(self, key):
        return WARNING_LABELS.get(key, key.replace("_", " "))

    def area_key_to_korean(self, area_key):
        return AREA_LABELS.get(area_key, area_key)

    def parse_session_datetime(self, session):
        raw = session.get("session_started_at", "")
        if not raw:
            return None

        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(raw, fmt)
            except Exception:
                continue

        return None

    def get_top_warning_info(self, warning_totals):
        if not warning_totals:
            return {"key": "", "name_kr": "", "count": 0}

        key, value = max(warning_totals.items(), key=lambda x: x[1])
        if value <= 0:
            return {"key": "", "name_kr": "", "count": 0}

        return {
            "key": key,
            "name_kr": self.warning_key_to_korean(key),
            "count": value,
        }

    def get_dominant_exercise_info(self, exercise_counts):
        if not exercise_counts:
            return {"code": "", "name_kr": "", "count": 0}

        code, count = max(exercise_counts.items(), key=lambda x: x[1])
        return {
            "code": code,
            "name_kr": TRACKED_EXERCISE_INFO.get(code, {}).get("name_kr", code),
            "count": count,
        }

    def get_dominant_area_info(self, area_volume):
        if not area_volume:
            return {
                "key": "unknown",
                "name_kr": "알 수 없음",
                "volume": 0,
            }

        key, volume = max(area_volume.items(), key=lambda x: x[1])
        return {
            "key": key,
            "name_kr": self.area_key_to_korean(key),
            "volume": volume,
        }

    def estimate_focus_keywords(self, area_volume, exercise_counts):
        shoulders = area_volume.get("shoulders", 0)
        arms = area_volume.get("arms", 0)

        focus = []

        if shoulders > arms + 10:
            focus.extend(["등", "가슴", "자세 안정", "가벼운 하체"])
        elif arms > shoulders + 10:
            focus.extend(["어깨 안정", "등", "가슴", "자세 안정"])
        else:
            focus.extend(["상체 균형", "자세 안정", "가벼운 하체"])

        has_bicep = exercise_counts.get("bicep_curl", 0) > 0
        has_shoulder_press = exercise_counts.get("shoulder_press", 0) > 0
        has_lateral_raise = exercise_counts.get("lateral_raise", 0) > 0

        if not has_bicep:
            focus.append("팔")
        if not has_shoulder_press and not has_lateral_raise:
            focus.append("어깨")

        unique_focus = []
        for item in focus:
            if item not in unique_focus:
                unique_focus.append(item)

        return unique_focus[:4]

    # ==========================================
    # DB 전체 조회
    # ==========================================
    def get_all_sessions_from_db(self, user_id):
        all_sessions = []

        try:
            # 해당 사용자의 모든 데이터 조회 (구조: user_id/date/exercise/session)
            ref = db.reference(f"{user_id}")
            user_data = ref.get()

            if not user_data or not isinstance(user_data, dict):
                return all_sessions

            # 날짜별 순회
            for date_key, date_data in user_data.items():
                if isinstance(date_data, dict):
                    # 운동 종류별 순회
                    for exercise_type, exercise_data in date_data.items():
                        if isinstance(exercise_data, dict):
                            # 세션별 순회
                            for session_key, session_data in exercise_data.items():
                                if isinstance(session_data, dict):
                                    all_sessions.append(session_data)

        except Exception as e:
            self.get_logger().warn(f"{user_id} 데이터 조회 실패: {e}")

        all_sessions.sort(
            key=lambda x: x.get("session_started_at", ""),
            reverse=True
        )
        return all_sessions

    # [수정] user_id 매개변수 추가
    def build_analysis_payload(self, user_id):
        # [수정] ID를 전달하여 특정 사용자의 데이터만 가져옴
        all_sessions = self.get_all_sessions_from_db(user_id)

        all_time_summary = self.summarize_sessions(all_sessions)
        last_workout_date, last_day_sessions = self.get_last_workout_day_sessions(all_sessions)
        last_day_summary = self.summarize_sessions(last_day_sessions)

        return {
            "basis_mode": "last_workout_day",
            "last_workout_date": last_workout_date,
            "last_day_session_count": len(last_day_sessions),
            "last_day_summary": last_day_summary,
            "all_time_summary": all_time_summary,
        }

    def get_last_workout_day_sessions(self, sessions):
        dated_sessions = []

        for session in sessions:
            dt = self.parse_session_datetime(session)
            if dt is not None:
                dated_sessions.append((dt, session))

        if not dated_sessions:
            return "", []

        dated_sessions.sort(key=lambda x: x[0], reverse=True)
        last_day = dated_sessions[0][0].date().isoformat()

        last_day_sessions = [
            session for dt, session in dated_sessions
            if dt.date().isoformat() == last_day
        ]

        last_day_sessions.sort(
            key=lambda s: self.parse_session_datetime(s) or datetime.min,
            reverse=True
        )

        return last_day, last_day_sessions
    
    def summarize_sessions(self, sessions):
        if not sessions:
            return {
                "total_sessions": 0,
                "total_reps": 0,
                "exercise_counts": {},
                "total_reps_by_exercise": {},
                "warning_totals": {},
                "area_volume": {},
                "latest_sessions": [],
                "insight": {
                    "dominant_exercise": {"code": "", "name_kr": "", "count": 0},
                    "dominant_area": {"key": "unknown", "name_kr": "알 수 없음", "volume": 0},
                    "recommended_focus_keywords": ["가벼운 전신 균형"],
                    "top_warning": {"key": "", "name_kr": "", "count": 0},
                }
            }

        exercise_counts = defaultdict(int)
        total_reps_by_exercise = defaultdict(int)
        warning_totals = defaultdict(int)
        area_volume = defaultdict(int)
        latest_sessions = []
        total_reps = 0

        for session in sessions:
            exercise_type = session.get("exercise_type", "unknown")
            rep_count = self.to_int(session.get("rep_count", 0))
            warning_counts = session.get("warning_counts", {})
            metrics = session.get("elderly_pt_metrics", {})

            exercise_counts[exercise_type] += 1
            total_reps_by_exercise[exercise_type] += rep_count
            total_reps += rep_count

            info = TRACKED_EXERCISE_INFO.get(exercise_type, {})
            main_area = info.get("main_area", "unknown")
            main_area_kr = info.get("main_area_kr", self.area_key_to_korean(main_area))

            area_volume[main_area] += rep_count

            if isinstance(warning_counts, dict):
                for key, value in warning_counts.items():
                    warning_totals[key] += self.to_int(value, 0)

            latest_sessions.append({
                "exercise_type": exercise_type,
                "exercise_name_kr": info.get("name_kr", exercise_type),
                "main_area": main_area,
                "main_area_kr": main_area_kr,
                "rep_count": rep_count,
                "session_started_at": session.get("session_started_at", ""),
                "warning_counts": warning_counts,
                "elderly_pt_metrics": metrics,
            })

        latest_sessions.sort(
            key=lambda x: x.get("session_started_at", ""),
            reverse=True
        )

        dominant_exercise = self.get_dominant_exercise_info(exercise_counts)
        dominant_area = self.get_dominant_area_info(area_volume)
        recommended_focus_keywords = self.estimate_focus_keywords(area_volume, exercise_counts)
        top_warning = self.get_top_warning_info(warning_totals)

        return {
            "total_sessions": len(sessions),
            "total_reps": total_reps,
            "exercise_counts": dict(exercise_counts),
            "total_reps_by_exercise": dict(total_reps_by_exercise),
            "warning_totals": dict(warning_totals),
            "area_volume": dict(area_volume),
            "latest_sessions": latest_sessions[:10],
            "insight": {
                "dominant_exercise": dominant_exercise,
                "dominant_area": dominant_area,
                "recommended_focus_keywords": recommended_focus_keywords,
                "top_warning": top_warning,
            }
        }


    def publish_response(self, request_type, analysis):
        payload = {
            "type": request_type,
            "analysis": analysis
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.response_pub.publish(msg)
        self.get_logger().info(f"📤 planner 응답 발행: {payload}")
    
    # ==========================================
    # 요청 처리
    # ==========================================
    def request_callback(self, msg):
        try:
            # [수정] 수신된 JSON 문자열 파싱
            data = json.loads(msg.data)
            request_type = data.get("type", "").strip()
            user_id = data.get("user_id", "unknown")
            
            self.get_logger().info(f"📥 planner 요청 수신: {request_type} (사용자: {user_id})")

            if request_type in ["exercise_log", "today_routine"]:
                # [수정] 파싱된 user_id를 분석 함수에 전달
                analysis = self.build_analysis_payload(user_id)
                self.publish_response(request_type, analysis)
            else:
                self.publish_response(
                    "unknown",
                    {"message": "요청을 이해하지 못했습니다."}
                )

        except Exception as e:
            # [추가] JSON 파싱 에러 발생 시 기존 방식(단순 문자열) 시도 혹은 에러 처리
            self.get_logger().error(f"요청 처리 중 에러: {e}")
            self.publish_response(
                "error",
                {"message": "데이터 파싱 또는 조회 중 문제가 발생했습니다."}
            )



def main(args=None):
    rclpy.init(args=args)
    node = ExercisePlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()