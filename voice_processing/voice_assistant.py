import json
import os
import tempfile
import threading 

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
from std_msgs.msg import String

# 사용자가 작성해둔 커스텀 음성 처리 모듈 임포트
from voice_processing.MicController import MicController, MicConfig
from voice_processing.wakeup_word import WakeupWord
from voice_processing.stt import STT

package_path = get_package_share_directory("rehab_assist_robot")
load_dotenv(dotenv_path=os.path.join(package_path, "resource", ".env"))
openai_api_key = os.getenv("OPENAI_API_KEY")

# ==========================================
# 1. 운동 피드백 생성 및 TTS 모듈 (파일 읽기 기능 제거)
# ==========================================
class VoiceFeedbackGenerator:
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

    # 파라미터로 데이터를 직접 넘겨받도록 수정
    def build_report_text(self, data):
        if not data:
            return "아직 진행 중인 운동 데이터가 없습니다. 먼저 운동을 시작해주세요."

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
# 2. ROS2 메인 노드 (상시 대기 음성 비서)
# ==========================================
class VoiceAssistant(Node):
    def __init__(self):
        super().__init__("voice_assistant_node")
        
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.1, openai_api_key=openai_api_key
        )

        prompt_content = """
당신은 음성 명령을 분류하고 필요한 키워드를 추출하는 도우미입니다.
<역할>
사용자 문장을 아래 4가지 중 하나로 분류하세요.
1. 운동 기록 조회/분석 요청
2. 운동 시작 요청
3. 자세 교정 요청
4. 운동 종료 요청

<출력 형식>
반드시 아래 중 하나만 출력하세요. (키워드 / 타겟)

1. 운동 기록 조회인 경우: exercise_log /
2. 운동 시작 요청인 경우: start_exercise / [운동종목]
3. 자세 교정 요청인 경우: posture_correction /
4. 운동 종료 요청인 경우: end_exercise /
5. 위 4가지에 해당하지 않으면: unknown /

<운동종목 작성 규칙>
사용자가 언급한 운동을 아래 3가지 영어 키워드 중 하나로만 변환하세요.
- 이두 운동, 팔 운동 -> bicep_curl
- 숄더 프레스, 어깨 프레스 -> shoulder_press
- 사레레, 사이드 레터럴 레이즈, 측면 어깨 -> lateral_raise
* 만약 운동 종목을 말하지 않았다면 / 뒤를 비워두세요.

예시) "나 이두 운동 시작할게" -> start_exercise / bicep_curl
예시) "사레레 하자" -> start_exercise / lateral_raise
예시) "운동 시작하자" -> start_exercise / 

<규칙>
- 설명 절대 금지, 다른 문장 절대 금지
- 도구와 위치는 공백으로 구분 (목적지가 없으면 / 뒤를 비움)
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

        mic_config = MicConfig(
            chunk=12000, rate=48000, channels=1, record_seconds=5,
            fmt=pyaudio.paInt16, device_index=10, buffer_size=24000,
        )
        self.mic_controller = MicController(config=mic_config)
        self.wakeup_word = WakeupWord(mic_config.buffer_size)

        self.cmd_pub = self.create_publisher(String, '/system_command', 10)
        self.mode_pub = self.create_publisher(String, '/set_exercise_mode', 10)

        # 💡 [핵심 추가] 최신 운동 상태를 메모리에 들고 있기 위한 변수와 구독자 설정
        self.latest_exercise_data = None
        self.result_sub = self.create_subscription(
            String,
            '/exercise_result',
            self.exercise_result_callback,
            10
        )

        self.get_logger().info("🎙️ 상시 대기형 음성 비서 노드 시작! (토픽 연동 완료)")
        
        self.listen_thread = threading.Thread(target=self.continuous_listening_loop, daemon=True)
        self.listen_thread.start()

    def exercise_result_callback(self, msg):
        """ /exercise_result 토픽 데이터를 받아 메모리에 최신화 """
        try:
            self.latest_exercise_data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"운동 결과 데이터 파싱 오류: {e}")

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

    def continuous_listening_loop(self):
        while rclpy.ok(): 
            try:
                self.get_logger().info("⏳ 웨이크업 워드(호출어) 대기 중...")
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

                if "start_exercise" in keywords:
                    exercise_name_kor = "운동" 
                    if targets:
                        mode_msg = String()
                        mode_str = targets[0]
                        mode_msg.data = mode_str
                        self.mode_pub.publish(mode_msg)
                        self.get_logger().info(f"✅ [모드 변경 전달] {mode_str}")
                        
                        if mode_str == "bicep_curl": exercise_name_kor = "이두 운동"
                        elif mode_str == "shoulder_press": exercise_name_kor = "숄더 프레스"
                        elif mode_str == "lateral_raise": exercise_name_kor = "사이드 레터럴 레이즈"

                    cmd_msg.data = "START_EXERCISE"
                    self.cmd_pub.publish(cmd_msg)
                    self.get_logger().info("✅ [상태 변경 전달] START_EXERCISE")
                    
                    self.reporter.speak(f"네, {exercise_name_kor}을 시작합니다. 자세를 잡아주세요.")

                elif "posture_correction" in keywords:
                    cmd_msg.data = "CORRECTION"
                    self.cmd_pub.publish(cmd_msg)
                    self.get_logger().info("✅ [명령 전달] CORRECTION")
                    self.reporter.speak("네, 로봇을 이동시켜 자세를 교정하겠습니다. 가만히 계셔주세요.")

                elif "exercise_log" in keywords:
                    cmd_msg.data = "REPORT_EXERCISE"
                    self.cmd_pub.publish(cmd_msg)
                    self.get_logger().info("✅ [명령 전달] REPORT_EXERCISE")
                    
                    # 💡 [핵심 변경] 로컬 파일 대신 구독해둔 메모리의 데이터를 바로 전달
                    report_text = self.reporter.build_report_text(self.latest_exercise_data)
                    self.get_logger().info(f"📝 운동 리포트: {report_text}")
                    self.reporter.speak(report_text)
                
                elif "end_exercise" in keywords:
                    cmd_msg.data = "END_EXERCISE"
                    self.cmd_pub.publish(cmd_msg)
                    self.get_logger().info("[명령 전달] END_EXERCISE")
                    self.reporter.speak("운동을 종료합니다. 수고하셨습니다.")

                else:
                    self.get_logger().warn("❓ 명령을 이해하지 못했습니다.")
                    self.reporter.speak("잘 못 들었어요. 다시 한 번 말씀해 주시겠어요?")

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