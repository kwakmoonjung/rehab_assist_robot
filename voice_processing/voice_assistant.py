#!/usr/bin/env python3
import json
import os
import tempfile
import threading
import time

import cv2
import numpy as np
import pyaudio
import rclpy
import scipy.io.wavfile as wav
import sounddevice as sd
from cv_bridge import CvBridge
from openai import OpenAI
from rclpy.node import Node
from sensor_msgs.msg import Image

from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from std_msgs.msg import String

from voice_processing.MicController import MicController, MicConfig
from voice_processing.wakeup_word import WakeupWord
from voice_processing.stt import STT


# ==========================================
# 환경 설정
# ==========================================
package_path = get_package_share_directory("rehab_assist_robot")
load_dotenv(dotenv_path=os.path.join(package_path, "resource", ".env"))
openai_api_key = os.getenv("OPENAI_API_KEY")


# ==========================================
# 1. LLM 문장 생성 + TTS
# ==========================================
class VoiceResponseGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            openai_api_key=api_key
        )

        # 네 기존 exercise_log 분석 흐름 유지
        analysis_prompt = """
당신은 재활 운동 코치입니다.
당신은 노인 운동 코치입니다.
아래 JSON은 운동 플래너가 DB 전체 기록을 읽은 뒤, 가장 마지막 운동한 날짜의 전체 운동 기록을 기준으로 다시 요약한 데이터입니다.

중요:
- 반드시 last_workout_date 와 last_day_summary 를 우선 기준으로 설명하세요.
- 가장 최근 운동 1개가 아니라, 가장 마지막 운동한 날짜 하루 전체 운동 성향을 기준으로 말해야 합니다.
- 예를 들어 마지막 운동 날짜에 어깨 중심 운동 비중이 높았다면, 그날은 어깨 중심으로 운동했다고 설명해야 합니다.

반드시 포함:
1. 가장 마지막 운동한 날짜가 언제였는지
2. 그날 어떤 운동들을 했는지
3. 그날 어느 부위를 중심으로 운동했는지
4. 전체적인 운동량이 어땠는지
5. 다음 운동 때 보완하면 좋을 점 1개

규칙:
- 4~6문장
- 제목 금지
- 리스트 금지
- 음성으로 읽기 쉽게 작성
- 첫 문장 또는 두 번째 문장에서 날짜를 반드시 언급할 것
- 너무 딱딱하지 않게 자연스럽게 말할 것
- 데이터가 부족하면 기록이 아직 충분하지 않다고 부드럽게 안내할 것

분석 데이터:
{analysis_json}
"""
        self.analysis_prompt = PromptTemplate(
            input_variables=["analysis_json"],
            template=analysis_prompt,
        )
        self.analysis_chain = self.analysis_prompt | self.llm

        routine_prompt = """
당신은 재활 운동 코치입니다.
당신은 노인 운동 코치입니다.
아래 JSON은 운동 플래너가 DB 전체 기록을 읽은 뒤, 가장 마지막 운동한 날짜의 전체 운동 기록을 기준으로 다시 요약한 데이터입니다.

중요:
- 반드시 last_workout_date 와 last_day_summary 를 우선 기준으로 루틴을 구성하세요.
- 가장 최근 운동 1개가 아니라, 가장 마지막 운동한 날짜 하루 전체 운동 성향을 기준으로 판단해야 합니다.
- 예를 들어 마지막 운동 날짜에 어깨 중심 운동 비중이 높았다면, 오늘은 어깨를 또 몰아서 하기보다 가슴, 등, 자세 안정, 하체 보완 쪽으로 루틴을 짜세요.

반드시 포함:
1. 가장 마지막 운동한 날짜가 언제였는지
2. 그날 어느 부위를 중심으로 운동했는지
3. 그래서 오늘은 어느 부위를 중점적으로 보완할 것인지
4. 오늘 할 운동 3~4개
5. 각 운동별 횟수와 세트 수
6. 마지막에 주의사항 1문장

규칙:
- 4~6문장
- 제목 금지
- 리스트 금지
- 음성으로 읽기 쉽게 작성
- 첫 문장 또는 두 번째 문장에서 날짜와 직전 운동 부위를 반드시 언급할 것
- 너무 어렵거나 위험한 운동은 피할 것
- 장비 없어도 가능한 운동 위주
- 상체 중심으로 하되 필요하면 가벼운 하체 1개 정도 포함 가능

분석 데이터:
{analysis_json}
"""
        self.routine_prompt = PromptTemplate(
            input_variables=["analysis_json"],
            template=routine_prompt,
        )
        self.routine_chain = self.routine_prompt | self.llm

    def build_exercise_log_text(self, analysis_data):
        last_day_summary = analysis_data.get("last_day_summary", {}) if analysis_data else {}
        if not analysis_data or last_day_summary.get("total_sessions", 0) == 0:
            return (
                "아직 저장된 운동 기록이 많지 않습니다. "
                "조금 더 운동 데이터가 쌓이면 더 정확하게 분석해드릴 수 있습니다."
            )

        response = self.analysis_chain.invoke(
            {"analysis_json": json.dumps(analysis_data, ensure_ascii=False, indent=2)}
        )
        return response.content.strip()

    def build_today_routine_text(self, analysis_data):
        last_day_summary = analysis_data.get("last_day_summary", {}) if analysis_data else {}
        if not analysis_data or last_day_summary.get("total_sessions", 0) == 0:
            return (
                "기록이 아직 충분하지 않아 가벼운 기본 루틴으로 진행하겠습니다. "
                "오늘은 숄더 프레스 8회 2세트, 이두 컬 10회 2세트, 벽 푸시업 10회 2세트로 시작하세요. "
                "동작은 천천히 하고 통증이 느껴지면 바로 멈춰주세요."
            )

        response = self.routine_chain.invoke(
            {"analysis_json": json.dumps(analysis_data, ensure_ascii=False, indent=2)}
        )
        return response.content.strip()

    def build_speech_text(self, response_type, analysis_data):
        if response_type == "exercise_log":
            return self.build_exercise_log_text(analysis_data)
        if response_type == "today_routine":
            return self.build_today_routine_text(analysis_data)
        return "요청 결과를 읽을 수 없습니다."

    def speak(self, text):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_path = temp_wav.name

            with self.client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="wav",
            ) as response:
                response.stream_to_file(temp_path)

            sample_rate, audio_data = wav.read(temp_path)
            sd.play(audio_data, sample_rate)
            sd.wait()

        except Exception as e:
            print(f"TTS 출력 중 에러 발생: {e}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


# ==========================================
# 2. ROS2 메인 노드
# ==========================================
class VoiceAssistant(Node):
    def __init__(self):
        super().__init__("voice_assistant_node")

        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key=openai_api_key
        )

        prompt_content = """
당신은 음성 명령을 분류하고 필요한 키워드를 추출하는 도우미입니다.

<역할>
사용자 문장을 아래 5가지 중 하나로 분류하세요.
1. 운동 기록 조회/분석 요청
2. 오늘 운동 루틴 추천 요청
3. 운동 시작 요청
4. 자세 교정 요청
5. 그 외

<출력 형식>
반드시 아래 중 하나만 출력하세요.

1. 운동 기록 조회/분석:
exercise_log /

2. 오늘 운동 루틴 추천:
today_routine /

3. 운동 시작 요청:
start_exercise / [운동종목]

4. 자세 교정 요청:
posture_correction /

5. 해당 없음:
unknown /

<운동종목 작성 규칙>
사용자가 언급한 운동을 아래 3가지 영어 키워드 중 하나로만 변환하세요.
- 이두 운동, 팔 운동 -> bicep_curl
- 숄더 프레스, 어깨 프레스 -> shoulder_press
- 사레레, 사이드 레터럴 레이즈, 측면 어깨 -> lateral_raise
- 운동 종목이 없으면 / 뒤를 비우세요.

<예시>
"오늘 운동 기록 알려줘" -> exercise_log /
"내 운동 데이터 분석해줘" -> exercise_log /
"오늘 루틴 짜줘" -> today_routine /
"오늘 운동 루틴 추천해줘" -> today_routine /
"이두 운동 시작할게" -> start_exercise / bicep_curl
"사레레 하자" -> start_exercise / lateral_raise
"운동 시작하자" -> start_exercise /
"자세 교정해줘" -> posture_correction /
"로봇 움직여줘" -> posture_correction /
"안녕" -> unknown /

<규칙>
- 설명 절대 금지
- 반드시 한 줄만 출력
- 반드시 위 형식만 사용

<사용자 입력>
{user_input}
"""
        self.prompt_template = PromptTemplate(
            input_variables=["user_input"],
            template=prompt_content,
        )
        self.lang_chain = self.prompt_template | self.llm

        self.stt = STT(openai_api_key=openai_api_key)
        self.reporter = VoiceResponseGenerator(openai_api_key)

        mic_config = MicConfig(
            chunk=12000,
            rate=48000,
            channels=1,
            record_seconds=5,
            fmt=pyaudio.paInt16,
            device_index=10,
            buffer_size=24000,
        )

        self.mic_controller = MicController(config=mic_config)
        self.wakeup_word = WakeupWord(mic_config.buffer_size)

        self.cmd_pub = self.create_publisher(String, '/system_command', 10)
        self.mode_pub = self.create_publisher(String, '/set_exercise_mode', 10)

        self.planner_request_pub = self.create_publisher(
            String,
            '/exercise_planner/request',
            10
        )

        self.planner_response_sub = self.create_subscription(
            String,
            '/exercise_planner/response',
            self.planner_response_callback,
            10
        )

        self.pending_planner_type = None

        # ==========================================
        # 기존 상태 유지
        # ==========================================
        self.latest_exercise_data = None
        self.result_sub = self.create_subscription(
            String,
            '/exercise_result',
            self.exercise_result_callback,
            10
        )

        # ==========================================
        # 얼굴 인식 추가
        # ==========================================
        self.bridge = CvBridge()
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.face_registry_dir = os.path.join(package_path, "resource")
        os.makedirs(self.face_registry_dir, exist_ok=True)
        self.face_registry_path = os.path.join(self.face_registry_dir, "face_registry.json")
        self.face_registry = self._load_face_registry()

        self.face_lock = threading.Lock()
        self.current_person_id = "unknown_user"
        self.current_person_feature = None
        self.last_face_seen_at = 0.0

        self.image_counter = 0
        self.face_process_interval = 15
        self.face_similarity_threshold = 0.88
        self.face_keep_threshold = 0.82

        self.pending_new_face_feature = None
        self.pending_new_face_count = 0
        self.new_user_confirm_count = 2

        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.person_pub = self.create_publisher(String, '/current_person_id', 10)

        self.get_logger().info("🎙️ 상시 대기형 음성 비서 노드 시작! (planner 연동 완료)")
        self.get_logger().info("📷 얼굴 인식 활성화 완료")
        self.get_logger().info(f"🙂 face_registry 경로: {self.face_registry_path}")

        self.listen_thread = threading.Thread(
            target=self.continuous_listening_loop,
            daemon=True
        )
        self.listen_thread.start()

    # ==========================================
    # 얼굴 레지스트리
    # ==========================================
    def _load_face_registry(self):
        if not os.path.exists(self.face_registry_path):
            return {"next_id": 1, "users": []}

        try:
            with open(self.face_registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "next_id" not in data:
                data["next_id"] = 1
            if "users" not in data:
                data["users"] = []

            return data

        except Exception as e:
            self.get_logger().warn(f"face_registry 로드 실패. 새로 생성합니다: {e}")
            return {"next_id": 1, "users": []}

    def _save_face_registry(self):
        try:
            with open(self.face_registry_path, "w", encoding="utf-8") as f:
                json.dump(self.face_registry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.get_logger().error(f"face_registry 저장 실패: {e}")

    def _normalize_feature(self, feature):
        feature = np.array(feature, dtype=np.float32)
        norm = np.linalg.norm(feature)
        if norm == 0:
            return None
        return (feature / norm).tolist()

    def _cosine_similarity(self, a, b):
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)

        if a.size == 0 or b.size == 0:
            return 0.0

        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0

        return float(np.dot(a, b) / denom)

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

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y + h, x:x + w]

        if face_roi.size == 0:
            return None, None

        face_roi = cv2.resize(face_roi, (96, 96))
        face_roi = cv2.equalizeHist(face_roi)

        small = cv2.resize(face_roi, (24, 24)).astype(np.float32) / 255.0
        small_feature = small.flatten()

        hist = cv2.calcHist([face_roi], [0], None, [32], [0, 256]).flatten()
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist = hist / hist_sum

        feature = np.concatenate([small_feature, hist]).astype(np.float32)
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

    def _update_registered_feature(self, person_id, new_feature, alpha=0.90):
        for user in self.face_registry.get("users", []):
            if user.get("person_id") != person_id:
                continue

            old_feature = np.array(user.get("feature", []), dtype=np.float32)
            new_feature_np = np.array(new_feature, dtype=np.float32)

            if old_feature.size == 0 or new_feature_np.size == 0:
                return

            blended = alpha * old_feature + (1.0 - alpha) * new_feature_np
            blended = self._normalize_feature(blended)
            if blended is None:
                return

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
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        self._save_face_registry()
        self.get_logger().info(f"🆕 새 사용자 등록: {person_id}")
        return person_id

    def _reset_pending_new_face(self):
        self.pending_new_face_feature = None
        self.pending_new_face_count = 0

    def _identify_person_from_frame(self, frame):
        feature, face_box = self._extract_face_feature(frame)
        if feature is None:
            return None, None, None

        best_user, best_score = self._find_best_user(feature)

        # 1) 등록된 사용자와 충분히 유사하면 기존 사용자
        if best_user is not None and best_score >= self.face_similarity_threshold:
            person_id = best_user["person_id"]
            self._update_registered_feature(person_id, feature)
            self._reset_pending_new_face()
            return person_id, feature, face_box

        # 2) 현재 보고 있던 사람과 비슷하면 현재 사람 유지
        if self.current_person_id != "unknown_user" and self.current_person_feature is not None:
            current_score = self._cosine_similarity(feature, self.current_person_feature)
            if current_score >= self.face_keep_threshold:
                self._update_registered_feature(self.current_person_id, feature)
                self._reset_pending_new_face()
                return self.current_person_id, feature, face_box

        # 3) 새 얼굴 후보 버퍼링
        if self.pending_new_face_feature is None:
            self.pending_new_face_feature = feature
            self.pending_new_face_count = 1
            return None, feature, face_box

        pending_score = self._cosine_similarity(feature, self.pending_new_face_feature)
        if pending_score >= self.face_keep_threshold:
            self.pending_new_face_count += 1
        else:
            self.pending_new_face_feature = feature
            self.pending_new_face_count = 1
            return None, feature, face_box

        # 4) 연속으로 비슷한 새 얼굴이면 신규 등록
        if self.pending_new_face_count >= self.new_user_confirm_count:
            person_id = self._register_new_user(feature)
            self.current_person_feature = feature
            self._reset_pending_new_face()
            return person_id, feature, face_box

        return None, feature, face_box

    def _publish_current_person(self, person_id):
        msg = String()
        msg.data = person_id
        self.person_pub.publish(msg)

    def get_current_person_id(self):
        with self.face_lock:
            return self.current_person_id

    # ==========================================
    # planner 응답 처리
    # ==========================================
    def planner_response_callback(self, msg):
        try:
            data = json.loads(msg.data)
            response_type = data.get("type", "")
            analysis = data.get("analysis", {})

            if not self.pending_planner_type:
                return

            if response_type != self.pending_planner_type:
                return

            speech_text = self.reporter.build_speech_text(response_type, analysis)
            self.get_logger().info(f"📝 planner 기반 음성 문장 생성 완료: {speech_text}")
            self.reporter.speak(speech_text)

            self.pending_planner_type = None

        except Exception as e:
            self.get_logger().error(f"planner 응답 처리 오류: {e}")

    # ==========================================
    # 운동 결과 최신화
    # ==========================================
    def exercise_result_callback(self, msg):
        try:
            data = json.loads(msg.data)
            with self.face_lock:
                if not data.get("person_id"):
                    data["person_id"] = self.current_person_id
            self.latest_exercise_data = data

        except Exception as e:
            self.get_logger().error(f"운동 결과 데이터 파싱 오류: {e}")

    # ==========================================
    # 얼굴 인식 카메라 콜백
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

        with self.face_lock:
            person_id, feature, face_box = self._identify_person_from_frame(frame)

            if person_id is None:
                return

            if person_id != self.current_person_id:
                self.get_logger().info(f"🙂 현재 사용자 인식: {person_id}")

            self.current_person_id = person_id
            self.current_person_feature = feature
            self.last_face_seen_at = time.time()

        self._publish_current_person(person_id)

        if face_box is not None:
            x, y, w, h = face_box
            self.get_logger().debug(f"face box=({x}, {y}, {w}, {h})")

    # ==========================================
    # 명령 파싱
    # ==========================================
    def parse_command(self, output_message):
        response = self.lang_chain.invoke({"user_input": output_message})
        result = response.content.strip()

        if "/" not in result:
            return ["unknown"], []

        objects, targets = result.split("/", 1)
        object_list = objects.split()
        target_list = targets.split()

        self.get_logger().info(f"LLM 원시 분석 결과: {result}")
        return object_list, target_list

    # ==========================================
    # planner 요청
    # ==========================================
    def request_planner(self, request_type):
        msg = String()
        msg.data = request_type
        self.pending_planner_type = request_type
        self.planner_request_pub.publish(msg)
        self.get_logger().info(f"📤 planner 요청 전송: {request_type}")

    # ==========================================
    # 상시 음성 루프
    # ==========================================
    def continuous_listening_loop(self):
        while rclpy.ok():
            try:
                self.get_logger().info("⏳ 웨이크업 워드 대기 중...")
                self.mic_controller.open_stream()
                self.wakeup_word.set_stream(self.mic_controller.stream)

                while rclpy.ok() and not self.wakeup_word.is_wakeup():
                    pass

                self.mic_controller.close_stream()
                self.get_logger().info("👂 듣고 있습니다. 명령을 말씀해 주세요...")

                output_message = self.stt.speech2text()

                if not output_message or not output_message.strip():
                    continue

                self.get_logger().info(f"🗣️ 인식된 문장: {output_message}")

                keywords, targets = self.parse_command(output_message)
                cmd_msg = String()

                current_person_id = self.get_current_person_id()
                self._publish_current_person(current_person_id)

                if "start_exercise" in keywords:
                    exercise_name_kor = "운동"

                    if targets:
                        mode_msg = String()
                        mode_str = targets[0]
                        mode_msg.data = mode_str
                        self.mode_pub.publish(mode_msg)
                        self.get_logger().info(f"✅ [모드 변경 전달] {mode_str}")

                        if mode_str == "bicep_curl":
                            exercise_name_kor = "이두 운동"
                        elif mode_str == "shoulder_press":
                            exercise_name_kor = "숄더 프레스"
                        elif mode_str == "lateral_raise":
                            exercise_name_kor = "사이드 레터럴 레이즈"

                    cmd_msg.data = "START_EXERCISE"
                    self.cmd_pub.publish(cmd_msg)
                    self.get_logger().info("✅ [상태 변경 전달] START_EXERCISE")
                    self.get_logger().info(f"🙂 현재 사용자: {current_person_id}")

                    self.reporter.speak(
                        f"네, {exercise_name_kor}을 시작합니다. 자세를 잡아주세요."
                    )

                elif "posture_correction" in keywords:
                    cmd_msg.data = "CORRECTION"
                    self.cmd_pub.publish(cmd_msg)
                    self.get_logger().info("✅ [명령 전달] CORRECTION")
                    self.get_logger().info(f"🙂 현재 사용자: {current_person_id}")

                    self.reporter.speak(
                        "네, 로봇을 이동시켜 자세를 교정하겠습니다. 가만히 계셔주세요."
                    )

                elif "exercise_log" in keywords:
                    cmd_msg.data = "REPORT_EXERCISE"
                    self.cmd_pub.publish(cmd_msg)
                    self.get_logger().info("✅ [명령 전달] REPORT_EXERCISE")
                    self.get_logger().info(f"🙂 현재 사용자: {current_person_id}")
                    self.request_planner("exercise_log")

                elif "today_routine" in keywords:
                    cmd_msg.data = "TODAY_ROUTINE"
                    self.cmd_pub.publish(cmd_msg)
                    self.get_logger().info("✅ [명령 전달] TODAY_ROUTINE")
                    self.get_logger().info(f"🙂 현재 사용자: {current_person_id}")
                    self.request_planner("today_routine")

                else:
                    self.get_logger().warn("❓ 명령을 이해하지 못했습니다.")
                    self.reporter.speak("잘 못 들었어요. 다시 한 번 말씀해 주세요.")

            except Exception as e:
                self.get_logger().error(f"❌ 음성 처리 중 에러: {e}")

            finally:
                try:
                    self.mic_controller.close_stream()
                except Exception:
                    pass


def main(args=None):
    rclpy.init(args=args)
    node = VoiceAssistant()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()