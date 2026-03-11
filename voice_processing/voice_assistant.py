import os
import json
import time
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from dotenv import load_dotenv
from firebase_admin import credentials, db
from openai import OpenAI
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import firebase_admin


load_dotenv()

# ==========================================
# Firebase 및 OpenAI 설정
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIREBASE_KEY_PATH = os.path.join(CURRENT_DIR, "serviceAccountKey.json")
FIREBASE_DB_URL = "https://rehab-aa1ee-default-rtdb.asia-southeast1.firebasedatabase.app/"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class RehabUserInterface(Node):
    def __init__(self):
        super().__init__('rehab_user_interface')

        # ------------------------------
        # Firebase 초기화
        # ------------------------------
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
            self.get_logger().info("🔥 Firebase 연동 성공! 클라우드 브릿지 가동.")
        except Exception as e:
            self.get_logger().error(f"Firebase 초기화 실패: {e}")

        # ------------------------------
        # OpenAI 초기화
        # ------------------------------
        self.ai_client = None
        try:
            if OPENAI_API_KEY:
                self.ai_client = OpenAI(api_key=OPENAI_API_KEY)
                self.get_logger().info("🤖 OpenAI API 연동 준비 완료.")
            else:
                self.get_logger().warn("OPENAI_API_KEY가 없어 AI 코멘트 기능은 비활성화됩니다.")
        except Exception as e:
            self.get_logger().error(f"OpenAI 초기화 실패: {e}")

        # ------------------------------
        # 얼굴 인식(경량 버전)
        # 새 라이브러리 설치 없이 OpenCV만 사용
        # ------------------------------
        self.bridge = CvBridge()
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_registry_path = os.path.join(CURRENT_DIR, "face_registry.json")
        self.face_registry = self._load_face_registry()

        self.current_person_id = "unknown_user"
        self.current_person_feature = None
        self.last_face_seen_at = 0.0
        self.face_process_interval = 15          # 모든 프레임 처리하지 않고 간격 처리
        self.face_similarity_threshold = 0.92    # 등록된 사람과 같은지 판단
        self.face_keep_threshold = 0.86          # 현재 사람 유지용 완화 기준
        self.image_counter = 0

        # ------------------------------
        # AI 분석 중복 방지
        # 사람/운동/세션 단위로 관리
        # ------------------------------
        self.last_rep_count_by_session = {}
        self.analyzing_sessions = set()

        # ------------------------------
        # ROS 구독 시작
        # ------------------------------
        self.exercise_subscription = self.create_subscription(
            String,
            '/exercise_result',
            self.exercise_result_callback,
            10
        )

        self.image_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info("📡 '/exercise_result' 구독 시작")
        self.get_logger().info("📷 '/camera/camera/color/image_raw' 구독 시작")
        self.get_logger().info(f"🙂 얼굴 레지스트리 경로: {self.face_registry_path}")

    # ==========================================
    # 얼굴 레지스트리 유틸
    # ==========================================
    def _load_face_registry(self):
        if not os.path.exists(self.face_registry_path):
            return {"next_id": 1, "users": []}

        try:
            with open(self.face_registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "next_id" not in data:
                data["next_id"] = 1
            if "users" not in data:
                data["users"] = []
            return data
        except Exception as e:
            self.get_logger().warn(f"face_registry.json 로드 실패. 새로 생성합니다: {e}")
            return {"next_id": 1, "users": []}

    def _save_face_registry(self):
        try:
            with open(self.face_registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.face_registry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.get_logger().error(f"face_registry.json 저장 실패: {e}")

    def _cosine_similarity(self, a, b):
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)

        if a.size == 0 or b.size == 0:
            return 0.0

        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _normalize_feature(self, feature):
        feature = np.array(feature, dtype=np.float32)
        norm = np.linalg.norm(feature)
        if norm == 0:
            return None
        return (feature / norm).tolist()

    def _extract_face_feature(self, frame):
        if frame is None or frame.size == 0:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            return None, None

        # 가장 큰 얼굴 하나만 사용
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        if face_roi.size == 0:
            return None, None

        face_roi = cv2.resize(face_roi, (100, 100))
        face_roi = cv2.equalizeHist(face_roi)

        feature = face_roi.astype(np.float32).flatten()
        feature = self._normalize_feature(feature)
        if feature is None:
            return None, None

        return feature, (x, y, w, h)

    def _find_best_user(self, feature):
        best_user = None
        best_score = -1.0

        for user in self.face_registry.get("users", []):
            score = self._cosine_similarity(feature, user.get("feature", []))
            if score > best_score:
                best_score = score
                best_user = user

        return best_user, best_score

    def _update_registered_feature(self, person_id, new_feature, alpha=0.85):
        for user in self.face_registry.get("users", []):
            if user.get("person_id") == person_id:
                old_feature = np.array(user.get("feature", []), dtype=np.float32)
                new_feature_np = np.array(new_feature, dtype=np.float32)
                if old_feature.size == 0 or new_feature_np.size == 0:
                    return

                blended = alpha * old_feature + (1.0 - alpha) * new_feature_np
                blended = self._normalize_feature(blended)
                if blended is not None:
                    user["feature"] = blended
                    user["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    self.current_person_feature = blended
                    self._save_face_registry()
                return

    def _register_new_user(self, feature):
        person_id = f"user_{self.face_registry['next_id']:03d}"
        self.face_registry["next_id"] += 1
        self.face_registry["users"].append({
            "person_id": person_id,
            "feature": feature,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        self._save_face_registry()
        self.get_logger().info(f"🆕 새 사용자 등록: {person_id}")
        return person_id

    def _identify_person_from_frame(self, frame):
        feature, face_box = self._extract_face_feature(frame)
        if feature is None:
            return None, None

        best_user, best_score = self._find_best_user(feature)

        # 1. 기존 등록 사용자와 충분히 유사하면 같은 사람으로 판단
        if best_user is not None and best_score >= self.face_similarity_threshold:
            person_id = best_user["person_id"]
            self._update_registered_feature(person_id, feature)
            return person_id, face_box

        # 2. 현재 추적 중인 사람과 어느 정도 비슷하면 새 사용자로 늘리지 않고 유지
        if self.current_person_id != "unknown_user" and self.current_person_feature is not None:
            current_score = self._cosine_similarity(feature, self.current_person_feature)
            if current_score >= self.face_keep_threshold:
                self._update_registered_feature(self.current_person_id, feature)
                return self.current_person_id, face_box

        # 3. 전혀 다른 얼굴이면 새 사용자 등록
        person_id = self._register_new_user(feature)
        self.current_person_feature = feature
        return person_id, face_box

    # ==========================================
    # 카메라 콜백: 현재 사람 식별
    # ==========================================
    def image_callback(self, msg):
        self.image_counter += 1
        if self.image_counter % self.face_process_interval != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"카메라 프레임 변환 실패: {e}")
            return

        person_id, face_box = self._identify_person_from_frame(frame)
        if person_id is None:
            return

        self.current_person_id = person_id
        self.last_face_seen_at = time.time()

        if face_box is not None:
            x, y, w, h = face_box
            self.get_logger().debug(
                f"얼굴 인식: {person_id}, box=({x}, {y}, {w}, {h})"
            )

    # ==========================================
    # OpenAI 분석 요청
    # ==========================================
    def request_openai_analysis(self, data, session_ref_key):
        self.analyzing_sessions.add(session_ref_key)
        try:
            if self.ai_client is None:
                return

            exercise = data.get("exercise_type", "운동")
            rep = data.get("rep_count", 0)
            metrics = data.get("elderly_pt_metrics", {})
            warns = data.get("warning_counts", {})
            person_id = data.get("person_id", "unknown_user")

            max_l = metrics.get("max_rom_left", 0)
            max_r = metrics.get("max_rom_right", 0)
            lean_back = warns.get("lean_back_momentum", 0)

            prompt = f"""
            당신은 시니어 헬스케어 전문 AI 로봇 트레이너입니다.
            사용자 ID는 {person_id}이고, '{exercise}' 운동을 {rep}회 마쳤습니다.
            좌측 최고 각도는 {max_l}도, 우측 최고 각도는 {max_r}도입니다.
            허리 반동(보상작용) 경고는 {lean_back}회 있었습니다.
            이 데이터를 바탕으로 어르신에게 따뜻하고 친절한 말투로
            칭찬 1문장 + 개선점 1문장을 합쳐 50자 이내로 답해주세요.
            """

            response = self.ai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )

            ai_comment = response.choices[0].message.content.strip()
            self.get_logger().info(f"🤖 AI 분석 완료 ({person_id}): {ai_comment}")

            live_ref = db.reference(f'live_current_session/{person_id}')
            live_ref.update({"ai_comment": ai_comment})

        except Exception as e:
            self.get_logger().error(f"OpenAI 분석 중 에러 발생: {e}")
        finally:
            self.analyzing_sessions.discard(session_ref_key)

    # ==========================================
    # 운동 결과 수신 -> 사람별 저장
    # ==========================================
    def exercise_result_callback(self, msg):
        try:
            data = json.loads(msg.data)
            exercise_type = data.get("exercise_type", "unknown_exercise")

            raw_start_time = data.get("session_started_at", "default")
            session_key = raw_start_time.replace("-", "").replace(":", "").replace(" ", "_")

            # 현재 카메라에서 식별된 사용자를 결과 데이터에 붙임
            # 얼굴이 잠깐 안 잡히더라도 최근 식별 사용자 유지
            person_id = data.get("person_id") or self.current_person_id or "unknown_user"
            data["person_id"] = person_id

            # 1. 아카이브 저장: users/{person_id}/{exercise_type}_sessions/{session_key}
            db_path = f'users/{person_id}/{exercise_type}_sessions/{session_key}'
            db_ref = db.reference(db_path)
            db_ref.set(data)

            # 2. 실시간 UI 저장: live_current_session/{person_id}
            live_ref = db.reference(f'live_current_session/{person_id}')
            live_ref.update(data)

            # 3. 사람/세션별 OpenAI 분석 트리거
            session_ref_key = f"{person_id}|{exercise_type}|{session_key}"
            current_rep = int(data.get("rep_count", 0))
            last_rep = int(self.last_rep_count_by_session.get(session_ref_key, 0))

            if current_rep > last_rep and session_ref_key not in self.analyzing_sessions:
                self.last_rep_count_by_session[session_ref_key] = current_rep
                threading.Thread(
                    target=self.request_openai_analysis,
                    args=(dict(data), session_ref_key),
                    daemon=True
                ).start()

        except Exception as e:
            self.get_logger().error(f"데이터 처리 에러: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RehabUserInterface()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            db.reference('live_current_session').delete()
            node.get_logger().info("🧹 Firebase 라이브 데이터 초기화 완료.")
        except Exception:
            pass

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()