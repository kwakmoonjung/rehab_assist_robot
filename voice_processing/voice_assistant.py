#!/usr/bin/env python3
import json
import os
import tempfile
import threading
import time
import warnings # [추가] 파이썬 경고 제어 모듈

# [추가] TensorFlow Lite C++ Info 로그 출력 억제
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pyaudio
import rclpy
import scipy.io.wavfile as wav
from scipy.io.wavfile import WavFileWarning # [추가] WavFileWarning 임포트
import sounddevice as sd
from openai import OpenAI
from rclpy.node import Node

warnings.filterwarnings("ignore", category=WavFileWarning)

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

        analysis_prompt = """
        당신은 노인 운동 보조 코치입니다.
        아래 JSON은 운동 플래너가 DB 전체 기록을 읽은 뒤, 가장 마지막 운동한 날짜의 전체 운동 기록을 기준으로 다시 요약한 데이터입니다.

        중요:
        - 반드시 last_workout_date 와 last_day_summary 를 우선 기준으로 설명하세요.
        - all_time_summary 는 보조 참고용입니다.
        - 가장 최근 운동 1개가 아니라, 가장 마지막 운동한 날짜 하루 전체 기록을 기준으로 설명해야 합니다.

        반드시 포함:
        1. 가장 마지막 운동한 날짜가 언제인지
        2. 그날 어떤 부위를 중심으로 운동했는지
        3. 그날 어떤 운동 비중이 높았는지
        4. 그날 기록 기준으로 가장 눈에 띄는 자세 문제 1개
        5. 오늘 보완하면 좋은 부위 1개 이상

        규칙:
        - 3~4문장
        - 쉬운 한국어
        - 제목 금지
        - 리스트 금지
        - 자연스럽게 말하듯 작성
        - 첫 문장이나 두 번째 문장 안에 날짜와 중심 부위를 꼭 언급할 것

        분석 데이터:
        {analysis_json}
        """

        self.analysis_prompt = PromptTemplate(
            input_variables=["analysis_json"],
            template=analysis_prompt,
        )
        self.analysis_chain = self.analysis_prompt | self.llm

        routine_prompt = """
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
        5. 운동 종료 요청

        <출력 형식>
        반드시 아래 6가지 중 하나만 출력하세요.

        1. 운동 기록 조회/분석:
        exercise_log /

        2. 오늘 운동 루틴 추천:
        today_routine /

        3. 운동 시작 요청:
        start_exercise / [운동종목]

        4. 자세 교정 요청:
        posture_correction /

        5. 운동 종료 요청인 경우: end_exercise /

        6. 위 5가지에 해당하지 않으면: unknown /

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
        "교정 시작할게" -> posture_correction /
        "운동 끝났어" -> end_exercise /       
        "운동 그만할래" -> end_exercise /     
        "안녕" -> unknown /

        <규칙>
        - 설명 절대 금지
        - 반드시 한 줄만 출력
        - 반드시 위 형식만 사용

        <사용자 입력>
        {user_input}
        """

        self.current_user_id = None        
        self.last_greeted_user = None


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

        self.recognized_user_sub = self.create_subscription(
            String,
            '/recognized_user',
            self.recognized_user_callback,
            10
        )

        self.pending_planner_type = None

        self.current_user_id = None
        self.current_user_time = 0.0
        self.user_valid_duration = 5.0  

        self.get_logger().info("음성 비서 노드 시작! 사용자 얼굴이 인식되기까지 대기합니다.")

        self.listen_thread = threading.Thread(
            target=self.continuous_listening_loop,
            daemon=True
        )
        self.listen_thread.start()

    # ==========================================
    # 얼굴 인식 결과 처리
    # ==========================================
    def recognized_user_callback(self, msg):
        try:
            user_id = msg.data.strip()
            if not user_id:
                return
            
            # [수정] 시간 체크 로직 삭제, 아이디가 변경되었을 때만 갱신 및 인사
            if self.current_user_id != user_id:
                self.current_user_id = user_id
                self.last_greeted_user = user_id
                
                greeting_text = (
                    f"안녕하세요 {user_id}님! 운동 루틴 추천해드릴까요? "
                    "아니면 원하시는 운동 시작한다고 말씀해주시면 자세 분석과 교정을 도와드릴게요."
                )
                self.get_logger().info(f"[{user_id}] 님 등장! 최초 1회 인사 실행 중...")
                
                import threading
                threading.Thread(target=self.reporter.speak, args=(greeting_text,), daemon=True).start()

        except Exception as e:
            self.get_logger().error(f"recognized_user 처리 오류: {e}")

    # [수정] 시간 만료 체크 함수 삭제 및 단순 ID 존재 여부 반환 함수로 변경
    def is_user_recognized(self):
        return self.current_user_id is not None

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
            self.get_logger().info(f"planner 기반 음성 문장 생성 완료: {speech_text}")
            self.reporter.speak(speech_text)

            self.pending_planner_type = None

        except Exception as e:
            self.get_logger().error(f"planner 응답 처리 오류: {e}")

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
        # [수정] string 대신 JSON 형태로 user_id를 함께 전송
        request_data = {
            "type": request_type,
            "user_id": self.current_user_id
        }
        msg.data = json.dumps(request_data)
        
        self.pending_planner_type = request_type
        self.planner_request_pub.publish(msg)
        self.get_logger().info(f"planner 요청 전송: {msg.data}")

    # ==========================================
    # 상시 음성 루프
    # ==========================================
    def continuous_listening_loop(self):
        while rclpy.ok():
            try:
                # [수정] 1. 등록된 사용자가 없을 때만 대기
                if not self.is_user_recognized():
                    time.sleep(0.5)
                    continue

                self.get_logger().info("웨이크업 워드 대기 중...")
                self.mic_controller.open_stream()
                self.wakeup_word.set_stream(self.mic_controller.stream)

                # [수정] 2. 웨이크업 대기 중 얼굴 인식 끊김(break) 로직 완전 제거
                while rclpy.ok() and not self.wakeup_word.is_wakeup():
                    pass

                self.mic_controller.close_stream()
                
                self.get_logger().info("듣고 있습니다. 명령을 말씀해 주세요...")

                output_message = self.stt.speech2text()

                self.get_logger().info(f"STT 인식 문장: {output_message}")

                if not output_message or not output_message.strip():
                    continue

                # 4. 기존 코드 유지 (명령어 파싱 및 대응)
                keywords, targets = self.parse_command(output_message)
                cmd_msg = String()

                if "start_exercise" in keywords:
                    if not self.current_user_id:
                        self.get_logger().warn("사용자 얼굴이 최근에 인식되지 않았습니다.")
                        self.reporter.speak(
                            "먼저 카메라를 바라봐 주세요. 사용자를 확인한 뒤 운동을 시작할게요."
                        )
                        continue

                    exercise_name_kor = "운동"

                    if targets:
                        mode_msg = String()
                        mode_str = targets[0]
                        mode_msg.data = mode_str
                        self.mode_pub.publish(mode_msg)
                        self.get_logger().info(f"[모드 변경 전달] {mode_str}")

                        if mode_str == "bicep_curl":
                            exercise_name_kor = "이두 운동"
                        elif mode_str == "shoulder_press":
                            exercise_name_kor = "숄더 프레스"
                        elif mode_str == "lateral_raise":
                            exercise_name_kor = "사이드 레터럴 레이즈"

                    cmd_msg.data = "START_EXERCISE"
                    self.cmd_pub.publish(cmd_msg)
                    self.get_logger().info("[상태 변경 전달] START_EXERCISE")

                    self.reporter.speak(
                        f"{self.current_user_id}님 확인되었습니다. 네, {exercise_name_kor}을 시작합니다. 자세를 잡아주세요."
                    )

                elif "posture_correction" in keywords:
                    if not self.current_user_id:
                        self.get_logger().warn("자세 교정 전 사용자 얼굴 인식 필요")
                        self.reporter.speak(
                            "먼저 카메라를 바라봐 주세요. 사용자를 확인한 뒤 자세 교정을 시작할게요."
                        )
                        continue

                    self.reporter.speak(
                        f"{self.current_user_id}님 확인되었습니다. 네, 로봇을 이동시켜 자세를 교정하겠습니다. 가만히 계셔주세요."
                    )

                    cmd_msg.data = "CORRECTION"
                    self.cmd_pub.publish(cmd_msg)
                    self.get_logger().info("[명령 전달] CORRECTION")

                    

                elif "exercise_log" in keywords:
                    if not self.current_user_id:
                        self.get_logger().warn("운동 기록 조회 전 사용자 얼굴 인식 필요")
                        self.reporter.speak(
                            "먼저 카메라를 바라봐 주세요. 사용자를 확인한 뒤 운동 기록을 안내할게요."
                        )
                        continue

                    cmd_msg.data = "REPORT_EXERCISE"
                    self.cmd_pub.publish(cmd_msg)
                    self.get_logger().info("[명령 전달] REPORT_EXERCISE")
                    self.request_planner("exercise_log")

                elif "today_routine" in keywords:
                    if not self.current_user_id:
                        self.get_logger().warn("루틴 추천 전 사용자 얼굴 인식 필요")
                        self.reporter.speak(
                            "먼저 카메라를 바라봐 주세요. 사용자를 확인한 뒤 오늘 루틴을 추천할게요."
                        )
                        continue

                    cmd_msg.data = "TODAY_ROUTINE"
                    self.cmd_pub.publish(cmd_msg)
                    self.get_logger().info("[명령 전달] TODAY_ROUTINE")
                    self.request_planner("today_routine")
                
                elif "end_exercise" in keywords:
                    cmd_msg.data = "END_EXERCISE"
                    self.cmd_pub.publish(cmd_msg)
                    self.get_logger().info("[명령 전달] END_EXERCISE")
                    self.reporter.speak("운동을 종료합니다. 수고하셨습니다.")

                else:
                    self.get_logger().warn("명령을 이해하지 못했습니다.")
                    self.reporter.speak("잘 못 들었어요. 다시 한 번 말씀해 주세요.")

            except Exception as e:
                self.get_logger().error(f"음성 처리 중 에러: {e}")

            # finally:
            #     try:
            #         self.mic_controller.close_stream()
            #     except Exception:
            #         pass


def main(args=None):
    rclpy.init(args=args)
    node = VoiceAssistant()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()