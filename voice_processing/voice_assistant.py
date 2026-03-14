#!/usr/bin/env python3
import json
import os
import tempfile
import threading
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pyaudio
import rclpy
import scipy.io.wavfile as wav
from scipy.io.wavfile import WavFileWarning
import sounddevice as sd
from openai import OpenAI
from rclpy.node import Node

warnings.filterwarnings("ignore", category=WavFileWarning)

from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from std_msgs.msg import String
from std_msgs.msg import Bool  # [추가] 긴급 정지 토픽용 Bool 임포트

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

        self.is_speaking = False
        self.speak_lock = threading.Lock()

        single_session_prompt = """
        당신은 노인 운동 보조 코치입니다.
        아래 JSON은 어르신이 방금 완료한 단일 운동 세션 결과입니다.

        반드시 포함:
        1. 방금 수행한 운동 종목과 총 횟수
        2. 정자세 비율 또는 전반적인 자세 평가
        3. 가장 많이 발생한 경고(warning_counts) 1가지에 대한 짧은 교정 피드백 (경고가 없으면 칭찬)

        규칙:
        - 2~3문장
        - 쉬운 한국어, 격려하는 따뜻한 말투
        - 제목 금지, 리스트 금지
        - 자연스럽게 말하듯 작성

        분석 데이터:
        {analysis_json}
        """

        self.single_prompt = PromptTemplate(
            input_variables=["analysis_json"],
            template=single_session_prompt,
        )
        self.single_chain = self.single_prompt | self.llm

        routine_prompt = """
        당신은 노인 운동 코치입니다.
        아래 JSON은 운동 플래너가 DB 전체 기록을 읽은 뒤, 가장 마지막 운동한 날짜의 전체 운동 기록을 기준으로 다시 요약한 데이터입니다.

        중요:
        - 반드시 last_workout_date 와 last_day_summary 를 우선 기준으로 루틴을 구성하세요.
        - 가장 최근 운동 1개가 아니라, 가장 마지막 운동한 날짜 하루 전체 운동 성향을 기준으로 판단해야 합니다.

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

        분석 데이터:
        {analysis_json}
        """

        self.routine_prompt = PromptTemplate(
            input_variables=["analysis_json"],
            template=routine_prompt,
        )
        self.routine_chain = self.routine_prompt | self.llm

    def build_single_session_text(self, analysis_data):
        if not analysis_data:
            return "아직 최근에 완료한 운동 기록이 없습니다."
        response = self.single_chain.invoke(
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
        if response_type == "single_session":
            return self.build_single_session_text(analysis_data)
        if response_type == "today_routine":
            return self.build_today_routine_text(analysis_data)
        return "요청 결과를 읽을 수 없습니다."

    def speak(self, text):
        with self.speak_lock:
            self.is_speaking = True
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
                self.is_speaking = False
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
        <역할> 사용자 문장을 분류하세요: exercise_log, today_routine, start_exercise, posture_correction, end_exercise, unknown
        <출력 형식> 키워드 / [종목] (bicep_curl, shoulder_press, lateral_raise 중 선택)
        <사용자 입력> {user_input}
        """

        self.current_user_id = None        
        self.last_greeted_user = None
        self.latest_session_data = None 
        self.is_correction_mode = False  # [추가] 교정 모드 상태 변수

        self.prompt_template = PromptTemplate(input_variables=["user_input"], template=prompt_content)
        self.lang_chain = self.prompt_template | self.llm

        self.stt = STT(openai_api_key=openai_api_key)
        self.reporter = VoiceResponseGenerator(openai_api_key)

        mic_config = MicConfig(chunk=12000, rate=48000, channels=1, record_seconds=3, fmt=pyaudio.paInt16, device_index=10, buffer_size=24000)
        self.mic_controller = MicController(config=mic_config)
        self.wakeup_word = WakeupWord(mic_config.buffer_size)

        self.cmd_pub = self.create_publisher(String, '/system_command', 10)
        self.mode_pub = self.create_publisher(String, '/set_exercise_mode', 10)
        self.planner_request_pub = self.create_publisher(String, '/exercise_planner/request', 10)
        self.session_feedback_pub = self.create_publisher(String, '/session_ai_feedback', 10)
        self.routine_pub = self.create_publisher(String, '/recommended_routine', 10)
        self.emg_pub = self.create_publisher(Bool, '/emergency_stop', 10) # [추가] 긴급 정지 퍼블리셔

        self.planner_response_sub = self.create_subscription(String, '/exercise_planner/response', self.planner_response_callback, 10)
        self.recognized_user_sub = self.create_subscription(String, '/recognized_user', self.recognized_user_callback, 10)
        self.correction_sub = self.create_subscription(String, '/end_correction', self.correction_callback, 10)
        self.exercise_result_sub = self.create_subscription(String, '/exercise_result', self.exercise_result_callback, 10)

        self.pending_planner_type = None
        self.get_logger().info("음성 비서 노드 시작!")

        self.listen_thread = threading.Thread(target=self.continuous_listening_loop, daemon=True)
        self.listen_thread.start()

    def exercise_result_callback(self, msg):
        try: self.latest_session_data = json.loads(msg.data)
        except Exception as e: self.get_logger().error(f"에러: {e}")

    def correction_callback(self, msg):
        self.is_correction_mode = False # [추가] 교정 완료 시 플래그 해제
        text = msg.data.strip()
        if text: threading.Thread(target=self.reporter.speak, args=(text,), daemon=True).start()

    def recognized_user_callback(self, msg):
        user_id = msg.data.strip()
        if user_id and self.current_user_id != user_id:
            self.current_user_id = user_id
            greeting_text = f"안녕하세요 {user_id}님! 운동을 도와드릴까요?"
            threading.Thread(target=self.reporter.speak, args=(greeting_text,), daemon=True).start()

    def is_user_recognized(self):
        return self.current_user_id is not None

    def planner_response_callback(self, msg):
        try:
            data = json.loads(msg.data)
            speech_text = self.reporter.build_speech_text(data.get("type"), data.get("analysis"))
            if data.get("type") == "today_routine":
                self.routine_pub.publish(String(data=speech_text))
            threading.Thread(target=self.reporter.speak, args=(speech_text,), daemon=True).start()
        except Exception as e: self.get_logger().error(f"에러: {e}")

    def parse_command(self, output_message):
        response = self.lang_chain.invoke({"user_input": output_message})
        result = response.content.strip()
        if "/" not in result: return ["unknown"], []
        parts = result.split("/")
        return parts[0].split(), parts[1].split()

    def request_planner(self, request_type):
        msg = String(data=json.dumps({"type": request_type, "user_id": self.current_user_id}))
        self.pending_planner_type = request_type
        self.planner_request_pub.publish(msg)

    def continuous_listening_loop(self):
        while rclpy.ok():
            try:
                if not self.is_user_recognized() or self.reporter.is_speaking:
                    time.sleep(0.5); continue
                
                # 1. 웨이크업 단계 (교정 중이면 건너뜀)
                if not self.is_correction_mode:
                    self.get_logger().info("웨이크업 워드 대기 중...")
                    self.mic_controller.open_stream()
                    self.wakeup_word.set_stream(self.mic_controller.stream)
                    is_wakeup = False
                    while rclpy.ok():
                        if self.reporter.is_speaking or self.is_correction_mode: break
                        if self.wakeup_word.is_wakeup(): is_wakeup = True; break
                    self.mic_controller.close_stream()
                    if not is_wakeup and not self.is_correction_mode:
                        time.sleep(0.5); continue

                # 2. 음성 인식 및 분석
                self.get_logger().info("듣고 있습니다...")
                output_message = self.stt.speech2text()
                if not output_message or not output_message.strip(): continue
                self.get_logger().info(f"인식: {output_message}")

                # 3. 긴급 정지 즉시 처리 [추가]
                if any(kw in output_message for kw in ["멈춰", "정지", "그만", "잠깐"]):
                    self.emg_pub.publish(Bool(data=True))
                    self.is_correction_mode = False
                    self.get_logger().warn("[긴급 정지] /emergency_stop 발행")
                    threading.Thread(target=self.reporter.speak, args=("동작을 즉시 멈춥니다.",), daemon=True).start()
                    continue

                # 4. 일반 명령 분석
                keywords, targets = self.parse_command(output_message)
                cmd_msg = String()

                if "start_exercise" in keywords:
                    exercise_name = "운동"
                    if targets:
                        mode_str = targets[0]
                        self.mode_pub.publish(String(data=mode_str))
                        exercise_name = "이두 운동" if mode_str == "bicep_curl" else "숄더 프레스" if mode_str == "shoulder_press" else "사이드 레터럴 레이즈"
                    cmd_msg.data = "START_EXERCISE"
                    self.cmd_pub.publish(cmd_msg)
                    threading.Thread(target=self.reporter.speak, args=(f"{exercise_name}을 시작합니다.",), daemon=True).start()

                elif "posture_correction" in keywords:
                    self.is_correction_mode = True # [추가] 교정 모드 진입
                    threading.Thread(target=self.reporter.speak, args=("자세를 교정하겠습니다. 가만히 계셔주세요.",), daemon=True).start()
                    self.cmd_pub.publish(String(data="CORRECTION"))

                elif "exercise_log" in keywords:
                    if not self.latest_session_data:
                        threading.Thread(target=self.reporter.speak, args=("기록이 없습니다.",), daemon=True).start()
                        continue
                    speech_text = self.reporter.build_speech_text("single_session", self.latest_session_data)
                    self.session_feedback_pub.publish(String(data=speech_text))
                    threading.Thread(target=self.reporter.speak, args=(speech_text,), daemon=True).start()

                elif "today_routine" in keywords:
                    self.request_planner("today_routine")
                
                elif "end_exercise" in keywords:
                    self.cmd_pub.publish(String(data="END_EXERCISE"))
                    threading.Thread(target=self.reporter.speak, args=("운동을 종료합니다.",), daemon=True).start()

            except Exception as e: self.get_logger().error(f"에러: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VoiceAssistant()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()