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

from voice_processing.MicController import MicController, MicConfig
from voice_processing.wakeup_word import WakeupWord
from voice_processing.stt import STT


package_path = get_package_share_directory("pick_and_place_voice")
load_dotenv(dotenv_path=os.path.join(package_path, "resource", ".env"))
openai_api_key = os.getenv("OPENAI_API_KEY")
LOG_FILE = os.path.expanduser("~/exercise_session_log.json")


class ExerciseVoiceReporter:
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
            print(f"TTS error: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


class GetKeyword(Node):
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.3, openai_api_key=openai_api_key
        )

        prompt_content = """
당신은 음성 명령을 분류하고 필요한 키워드를 추출하는 도우미입니다.

<역할>
사용자 문장을 아래 2가지 중 하나로 분류하세요.

1. 도구 이동 명령
2. 운동 기록 조회/분석 요청

<도구 리스트>
- hammer, screwdriver, wrench, pos1, pos2, pos3

<출력 형식>
반드시 아래 셋 중 하나만 출력하세요.

1. 도구 이동 명령인 경우
도구1 도구2 / pos1 pos2

2. 운동 기록 조회/분석 요청인 경우
exercise_log /

3. 둘 다 아니면
unknown /

<규칙>
- 설명 절대 금지
- 다른 문장 절대 금지
- 도구와 위치는 공백으로 구분
- 목적지가 없으면 / 뒤를 비움
- 운동 횟수, 자세 분석, 오늘 기록, 피드백, 운동 결과를 물으면 모두 exercise_log로 분류

<예시>
입력: hammer를 pos1에 가져다 놔
출력: hammer / pos1

입력: 왼쪽에 있는 해머와 wrench를 pos1에 넣어줘
출력: hammer wrench / pos1

입력: 왼쪽에 있는 못 박는 걸 줘
출력: hammer /

입력: 오늘 운동 기록 알려줘
출력: exercise_log /

입력: 오늘 자세 어땠는지 분석해줘
출력: exercise_log /

입력: 내가 몇 번 했는지 말해줘
출력: exercise_log /

입력: 안녕
출력: unknown /

<사용자 입력>
{user_input}
"""

        self.prompt_template = PromptTemplate(
            input_variables=["user_input"],
            template=prompt_content,
        )
        self.lang_chain = self.prompt_template | self.llm
        self.stt = STT(openai_api_key=openai_api_key)
        self.reporter = ExerciseVoiceReporter(openai_api_key)

        super().__init__("get_keyword_node")

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

        self.get_logger().info("MicRecorderNode initialized.")
        self.get_logger().info("wait for client's request...")

        self.get_keyword_srv = self.create_service(
            Trigger, "get_keyword", self.get_keyword
        )

    def parse_command(self, output_message):
        response = self.lang_chain.invoke({"user_input": output_message})
        result = response.content.strip()

        if "/" not in result:
            return ["unknown"], []

        objects, targets = result.split("/", 1)
        object_list = objects.split()
        target_list = targets.split()

        print(f"raw llm result: {result}")
        print(f"object: {object_list}")
        print(f"target: {target_list}")
        return object_list, target_list

    def wait_wakeup(self):
        self.get_logger().info("wake word listening...")
        while rclpy.ok() and not self.wakeup_word.is_wakeup():
            pass

    def get_keyword(self, request, response):
        try:
            self.mic_controller.open_stream()
            self.wakeup_word.set_stream(self.mic_controller.stream)
            self.wait_wakeup()
            self.mic_controller.close_stream()

            output_message = self.stt.speech2text()
            keywords, targets = self.parse_command(output_message)

            if "exercise_log" in keywords:
                report_text = self.reporter.build_report_text()
                self.get_logger().info(f"Exercise report: {report_text}")
                self.reporter.speak(report_text)

                response.success = True
                response.message = report_text
                return response

            if "unknown" in keywords or not keywords:
                response.success = False
                response.message = "명령을 이해하지 못했습니다. 다시 말씀해주세요."
                return response

            self.get_logger().warn(f"Detected tools: {keywords}, targets: {targets}")
            response.success = True
            response.message = " ".join(keywords)
            return response

        except OSError:
            self.get_logger().error("Error: Failed to open audio stream")
            self.get_logger().error("please check your device index")
            response.success = False
            response.message = "audio stream error"
            return response
        except Exception as e:
            self.get_logger().error(f"service error: {e}")
            response.success = False
            response.message = str(e)
            return response
        finally:
            try:
                self.mic_controller.close_stream()
            except Exception:
                pass


def main():
    rclpy.init()
    node = GetKeyword()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
