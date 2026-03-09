import json
import os
import tempfile

import pyaudio
import rclpy
import scipy.io.wavfile as wav
import sounddevice as sd
from openai import OpenAI
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from std_srvs.srv import Trigger

from std_msgs.msg import String

# 사용자가 작성해둔 커스텀 음성 처리 모듈 임포트
from voice_processing.MicController import MicController, MicConfig
from voice_processing.wakeup_word import WakeupWord
from voice_processing.stt import STT

# 환경 변수 로드 (API Key) - 패키지명은 통합된 rehab_assist_robot으로 맞췄습니다.
package_path = get_package_share_directory("rehab_assist_robot")
load_dotenv(dotenv_path=os.path.join(package_path, "resource", ".env"))
openai_api_key = os.getenv("OPENAI_API_KEY")

LOG_FILE = os.path.expanduser("~/exercise_session_log.json")

'''
voice_assistant에서 토픽으로 발행하는 값: /system_command
"START_EXERCISE"    # 운동 시작
"CORRECTION"        # 교정 시작
"REPORT_EXERCISE"   # 운동 분석 시작
'''

# ==========================================
# 1. 운동 피드백 생성 및 TTS 모듈
# ==========================================
class VoiceFeedbackGenerator:
    """운동 기록(JSON)을 읽고 LLM으로 분석하여 음성(TTS)으로 알려주는 클래스"""
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=api_key)

        analysis_prompt = """
당신은 재활 운동 보조 코치입니다.

아래 운동 기록 요약 JSON을 보고 사용자가 듣기 쉽게 한국어로 짧게 설명하세요.

반드시 포함할 내용:
1. 총 반복 횟수
2. 자세가 안정적이었는지 한 문장
3. 가장 많이 나온 문제 1~2개
4. 다음 운동 때 주의할 점 1개

규칙:
- 3~5문장으로 짧게
- 어려운 말 금지
- 환자에게 직접 말하듯 자연스럽게
- 숫자는 가능한 한 포함
- 기록이 거의 없으면 "아직 분석할 운동 데이터가 충분하지 않습니다."라고 말하세요.

운동 기록 요약:
{exercise_summary}
"""
        self.analysis_prompt = PromptTemplate(
            input_variables=["exercise_summary"],
            template=analysis_prompt,
        )
        self.analysis_chain = self.analysis_prompt | self.llm

    def load_log(self):
        if not os.path.exists(LOG_FILE):
            return None
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def build_report_text(self):
        data = self.load_log()
        if not data:
            return "운동 기록 파일이 아직 없습니다. 먼저 운동을 진행해주세요."

        if data.get("frame_count", 0) < 10:
            return "아직 분석할 운동 데이터가 충분하지 않습니다. 조금 더 운동한 뒤 다시 물어봐 주세요."

        response = self.analysis_chain.invoke(
            {"exercise_summary": json.dumps(data, ensure_ascii=False, indent=2)}
        )
        return response.content.strip()

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
# 2. ROS2 메인 노드 (음성 비서)
# ==========================================
class VoiceAssistant(Node):
    def __init__(self):
        super().__init__("voice_assistant_node")
        
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.3, openai_api_key=openai_api_key
        )

        prompt_content = """
당신은 음성 명령을 분류하고 필요한 키워드를 추출하는 도우미입니다.

<역할>
사용자 문장을 아래 4가지 중 하나로 분류하세요.

1. 도구 이동 명령
2. 운동 기록 조회/분석 요청
3. 운동 시작 요청
4. 자세 교정 요청

<출력 형식>
반드시 아래 중 하나만 출력하세요.

1. 운동 기록 조회인 경우: exercise_log /
2. 운동 시작 요청인 경우: start_exercise /
3. 자세 교정 요청인 경우: posture_correction /
4. 위 4가지에 해당하지 않으면: unknown /

<규칙>
- 설명 절대 금지, 다른 문장 절대 금지
- 도구와 위치는 공백으로 구분 (목적지가 없으면 / 뒤를 비움)
- "나 운동 시작할게", "운동하자" 등은 start_exercise로 분류
- "자세 교정해줘", "로봇 움직여줘" 등은 posture_correction으로 분류

<사용자 입력>
{user_input}
"""
        self.prompt_template = PromptTemplate(
            input_variables=["user_input"],
            template=prompt_content,
        )
        self.lang_chain = self.prompt_template | self.llm
        self.stt = STT(openai_api_key=openai_api_key)
        self.reporter = VoiceFeedbackGenerator(openai_api_key)

        # 마이크 및 웨이크업 워드 설정
        mic_config = MicConfig(
            chunk=12000,
            rate=48000,
            channels=1,
            record_seconds=5,
            fmt=pyaudio.paInt16,
            device_index=10,  # 사용자 환경에 맞는 인덱스
            buffer_size=24000,
        )
        self.mic_controller = MicController(config=mic_config)
        self.wakeup_word = WakeupWord(mic_config.buffer_size)

        self.cmd_pub = self.create_publisher(String, '/system_command', 10)

        # 트리거 서비스 생성
        self.voice_cmd_srv = self.create_service(
            Trigger, "get_keyword", self.handle_voice_command
        )
        
        self.get_logger().info("🎙️ 음성 비서 노드 초기화 완료.")
        self.get_logger().info("서비스 호출을 대기합니다 (Service: /get_keyword)...")

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

    def wait_wakeup(self):
        self.get_logger().info("⏳ 웨이크업 워드(호출어) 대기 중...")
        while rclpy.ok() and not self.wakeup_word.is_wakeup():
            pass

    def handle_voice_command(self, request, response):
        """서비스 호출 시 실행되는 메인 콜백"""
        try:
            # 1. 웨이크업 워드 대기
            self.mic_controller.open_stream()
            self.wakeup_word.set_stream(self.mic_controller.stream)
            self.wait_wakeup()
            self.mic_controller.close_stream()
            self.get_logger().info("듣고 있습니다...")

            # 2. 사용자의 음성 녹음 및 STT 변환
            output_message = self.stt.speech2text()
            self.get_logger().info(f"🗣️ 인식된 문장: {output_message}")
            
            # 3. LLM을 통한 의도 파악
            keywords, targets = self.parse_command(output_message)

            cmd_msg = String()

            # 케이스 1: 운동 시작
            if "start_exercise" in keywords:
                cmd_msg.data = "START_EXERCISE"
                self.cmd_pub.publish(cmd_msg)
                self.get_logger().info("✅ [명령 퍼블리시] START_EXERCISE")
                self.reporter.speak("네, 운동을 시작합니다. 자세를 잡아주세요.")
                response.success = True
                response.message = "운동 시작 명령 전달 완료"
                return response

            # 케이스 2: 자세 교정
            elif "posture_correction" in keywords:
                cmd_msg.data = "CORRECTION"
                self.cmd_pub.publish(cmd_msg)
                self.get_logger().info("✅ [명령 퍼블리시] CORRECTION")
                self.reporter.speak("네, 로봇을 이동시켜 자세를 교정하겠습니다. 가만히 계셔주세요.")
                response.success = True
                response.message = "자세 교정 명령 전달 완료"
                return response

            # 케이스 3: 운동 기록 조회
            elif "exercise_log" in keywords:
                cmd_msg.data = "REPORT_EXERCISE"
                self.cmd_pub.publish(cmd_msg)
                self.get_logger().info("✅ [명령 퍼블리시] REPORT_EXERCISE")
                
                report_text = self.reporter.build_report_text()
                self.get_logger().info(f"📝 운동 피드백 리포트: {report_text}")
                self.reporter.speak(report_text)
                
                response.success = True
                response.message = report_text
                return response

            # 예외: 이해하지 못함
            else:
                self.get_logger().warn("❓ 명령을 이해하지 못했습니다.")
                self.reporter.speak("잘 못 들었어요. 다시 한 번 말씀해 주시겠어요?")
                response.success = False
                response.message = "명령을 이해하지 못했습니다."
                return response

        except Exception as e:
            self.get_logger().error(f"❌ 서비스 처리 중 에러: {e}")
            response.success = False
            response.message = str(e)
            return response
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