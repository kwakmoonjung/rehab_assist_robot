#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  
import firebase_admin
from firebase_admin import credentials, db
from openai import OpenAI
from dotenv import load_dotenv

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
        
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
            self.get_logger().info("🔥 Firebase 연동 성공! 클라우드 브릿지 가동.")
        except Exception as e:
            self.get_logger().error(f"Firebase 초기화 실패: {e}")

        try:
            self.ai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.get_logger().info("🤖 OpenAI API 연동 준비 완료.")
        except Exception as e:
            self.get_logger().error(f"OpenAI 초기화 실패: {e}")

        # 🌟 [수정됨] 운동 세션당 딱 1번만 분석하기 위한 상태 변수
        self.is_analysis_completed = False

        self.subscription = self.create_subscription(
            String,
            '/exercise_result',
            self.exercise_result_callback,
            10
        )
        self.get_logger().info("📡 '/exercise_result' 통합 토픽 구독 시작...")

    def calculate_report_scores(self, data):
        ex_type = data.get("exercise_type", "unknown_exercise")
        mobility_score = 0
        stability_score = 50 
        
        if ex_type == 'lateral_raise':
            metrics = data.get("elderly_pt_metrics", {})
            max_rom = max(metrics.get("max_rom_left", 0), metrics.get("max_rom_right", 0))
            mobility_score = min(50, (max_rom / 80.0) * 50)
            
            warns = data.get("warning_counts", {})
            lean_back = warns.get("lean_back_momentum", 0)
            stability_score = max(0, 50 - (lean_back * 10))

        elif ex_type == 'shoulder_press':
            avg_shoulder = data.get("avg_shoulder_angle", 0)
            mobility_score = min(50, (avg_shoulder / 145.0) * 50)

            warns = data.get("warning_counts", {})
            arm_balance = warns.get("arm_balance_issue", 0)
            body_not_straight = warns.get("body_not_straight", 0)
            stability_score = max(0, 50 - ((arm_balance + body_not_straight) * 10))

        elif ex_type == 'bicep_curl':
            min_elbow = data.get("avg_elbow_angle", 180)
            mobility_score = min(50, ((180 - min_elbow) / (180 - 50.0)) * 50)
            
            warns = data.get("warning_counts", {})
            elbow_not_close = warns.get("elbows_not_close_to_body", 0)
            stability_score = max(0, 50 - (elbow_not_close * 10))

        mob = round(mobility_score)
        stab = round(stability_score)

        return {
            "mobility_score": mob,
            "stability_score": stab,
            "total_score": mob + stab
        }

    # 🌟 [수정됨] 최종 리포트 전용 프롬프트 및 분석 요청 로직
    def request_openai_analysis(self, data, report_scores):
        try:
            exercise = data.get("exercise_type", "운동")
            rep = data.get("rep_count", 0)
            total_score = report_scores.get("total_score", 0)
            mob_score = report_scores.get("mobility_score", 0)
            stab_score = report_scores.get("stability_score", 0)

            prompt = f"""
            당신은 시니어 헬스케어 전문 AI 로봇 트레이너입니다. 어르신의 이번 '{exercise}' 운동 세션이 모두 종료되었습니다.
            총 {rep}회를 수행했으며, 종합 점수는 100점 만점에 {total_score}점입니다. 
            (세부 지표: 관절 가동성 {mob_score}/50점, 자세 안정성 {stab_score}/50점)
            
            이 최종 리포트 데이터를 바탕으로, 어르신에게 따뜻하고 격려하는 말투로 이번 운동에 대한 총평과 다음 번 개선점 1가지를 합쳐서 50자 이내로 코멘트해주세요.
            """

            response = self.ai_client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            
            ai_comment = response.choices[0].message.content.strip()
            self.get_logger().info(f"🤖 최종 AI 분석 완료: {ai_comment}")

            # 생성된 최종 코멘트를 실시간 DB와 아카이브 DB에 모두 업데이트
            live_ref = db.reference('live_current_session')
            live_ref.update({"ai_comment": ai_comment})
            
            raw_start_time = data.get("session_started_at", "default")
            session_key = raw_start_time.replace("-", "").replace(":", "").replace(" ", "_")
            db_ref = db.reference(f'{exercise}_sessions/{session_key}')
            db_ref.update({"ai_comment": ai_comment})

        except Exception as e:
            self.get_logger().error(f"OpenAI 최종 분석 중 에러 발생: {e}")

    def exercise_result_callback(self, msg):
        try:
            data = json.loads(msg.data)
            exercise_type = data.get("exercise_type", "unknown_exercise")
            
            report_scores = self.calculate_report_scores(data)
            data["report_scores"] = report_scores
            
            raw_start_time = data.get("session_started_at", "default")
            session_key = raw_start_time.replace("-", "").replace(":", "").replace(" ", "_")
            
            # 1. 아카이브 DB 저장
            db_path = f'{exercise_type}_sessions/{session_key}'
            db_ref = db.reference(db_path)
            db_ref.set(data)
            
            # 2. UI 실시간 덮어쓰기
            live_ref = db.reference('live_current_session')
            live_ref.update(data)

            # 🌟 3. [OpenAI 트리거 로직 변경] 운동 종료 시점에 딱 한 번만!
            # ROS 2에서 보내주는 JSON 데이터에 "is_finished": true 가 포함되어 들어올 때 작동합니다.
            is_finished = data.get("is_finished", False)
            
            if is_finished and not self.is_analysis_completed:
                self.get_logger().info("🏁 운동 세션 종료 감지! 최종 AI 리포트 생성을 시작합니다...")
                self.is_analysis_completed = True
                threading.Thread(target=self.request_openai_analysis, args=(data, report_scores)).start()

            # (새로운 운동이 시작되어 rep_count가 다시 0이나 1로 초기화되면 분석 플래그도 초기화)
            if data.get("rep_count", 0) <= 1 and not is_finished:
                self.is_analysis_completed = False

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