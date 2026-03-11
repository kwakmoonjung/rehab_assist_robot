import os
import json
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  
import firebase_admin
from firebase_admin import credentials, db
from openai import OpenAI

# 🌟 dotenv 라이브러리 불러오기
from dotenv import load_dotenv

# 🌟 .env 파일에 있는 정보들을 메모리로 불러옴
load_dotenv()

# ==========================================
# Firebase 및 OpenAI 설정 
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIREBASE_KEY_PATH = os.path.join(CURRENT_DIR, "serviceAccountKey.json")
FIREBASE_DB_URL = "https://rehab-aa1ee-default-rtdb.asia-southeast1.firebasedatabase.app/"

# 🌟 [수정됨] 하드코딩된 키를 지우고, 환경 변수에서 가져오도록 변경!
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# (이하 코드 동일)
class RehabUserInterface(Node):
    def __init__(self):
        super().__init__('rehab_user_interface')
        
        # Firebase 초기화
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
            self.get_logger().info("🔥 Firebase 연동 성공! 클라우드 브릿지 가동.")
        except Exception as e:
            self.get_logger().error(f"Firebase 초기화 실패: {e}")

        # OpenAI 클라이언트 초기화
        try:
            self.ai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.get_logger().info("🤖 OpenAI API 연동 준비 완료.")
        except Exception as e:
            self.get_logger().error(f"OpenAI 초기화 실패: {e}")

        # 중복 분석을 막기 위한 이전 횟수 추적 변수
        self.last_rep_count = 0 
        self.is_analyzing = False

        self.subscription = self.create_subscription(
            String,
            '/exercise_result',
            self.exercise_result_callback,
            10
        )
        self.get_logger().info("📡 '/exercise_result' 통합 토픽 구독 시작...")

    # 🌟 [신규] OpenAI에게 분석을 요청하고 결과를 Firebase에 올리는 함수
    def request_openai_analysis(self, data):
        self.is_analyzing = True
        try:
            exercise = data.get("exercise_type", "운동")
            rep = data.get("rep_count", 0)
            metrics = data.get("elderly_pt_metrics", {})
            warns = data.get("warning_counts", {})
            
            max_l = metrics.get("max_rom_left", 0)
            max_r = metrics.get("max_rom_right", 0)
            lean_back = warns.get("lean_back_momentum", 0)

            # AI에게 명령할 프롬프트(지시서) 작성
            prompt = f"""
            당신은 시니어 헬스케어 전문 AI 로봇 트레이너입니다. 어르신이 '{exercise}' 운동을 {rep}회 마쳤습니다.
            좌측 최고 각도는 {max_l}도, 우측은 {max_r}도입니다. 허리 반동(보상작용) 경고가 {lean_back}회 있었습니다.
            이 데이터를 바탕으로 어르신에게 따뜻하고 친절한 말투로 칭찬 1문장과 개선점 1문장을 합쳐서 50자 이내로 코멘트해주세요.
            """

            response = self.ai_client.chat.completions.create(
                model="gpt-3.5-turbo", # GPT-4o-mini 등 사용 가능
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            
            ai_comment = response.choices[0].message.content.strip()
            self.get_logger().info(f"🤖 AI 분석 완료: {ai_comment}")

            # 생성된 AI 코멘트를 Firebase 'live_current_session/ai_comment' 경로에 덧붙여서 발행
            live_ref = db.reference('live_current_session')
            live_ref.update({"ai_comment": ai_comment})

        except Exception as e:
            self.get_logger().error(f"OpenAI 분석 중 에러 발생: {e}")
        finally:
            self.is_analyzing = False

    def exercise_result_callback(self, msg):
        try:
            data = json.loads(msg.data)
            exercise_type = data.get("exercise_type", "unknown_exercise")
            
            raw_start_time = data.get("session_started_at", "default")
            session_key = raw_start_time.replace("-", "").replace(":", "").replace(" ", "_")
            
            # 1. 아카이브용 DB 저장
            db_path = f'{exercise_type}_sessions/{session_key}'
            db_ref = db.reference(db_path)
            db_ref.set(data)
            
            # 2. UI 실시간 전광판용 덮어쓰기
            live_ref = db.reference('live_current_session')
            live_ref.update(data) # 기존 ai_comment를 덮어쓰지 않기 위해 set 대신 update 사용

            # 🌟 3. [OpenAI 트리거 로직] 
            # (임시 로직: 운동 횟수가 올라갈 때마다 백그라운드에서 AI 분석 실행)
            # (실전에서는 종료 버튼/플래그가 감지되었을 때 한 번만 호출하도록 변경하면 됩니다)
            current_rep = data.get("rep_count", 0)
            if current_rep > self.last_rep_count and not self.is_analyzing:
                self.last_rep_count = current_rep
                # 로봇의 실시간 통신이 멈추지 않도록 스레드(Thread)로 빼서 백그라운드 실행
                threading.Thread(target=self.request_openai_analysis, args=(data,)).start()

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