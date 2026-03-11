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

        self.is_analysis_completed = False

        self.subscription = self.create_subscription(
            String,
            '/exercise_result',
            self.exercise_result_callback,
            10
        )
        self.get_logger().info("📡 '/exercise_result' 통합 토픽 구독 시작...")

    # 🌟 [업데이트] AIT STUDIO 리포트 수준의 정량화 및 비대칭도 산출 로직
    def calculate_report_scores(self, data):
        ex_type = data.get("exercise_type", "unknown_exercise")
        mobility_score = 0
        stability_score = 50 
        max_l = 0
        max_r = 0
        
        if ex_type == 'lateral_raise':
            metrics = data.get("elderly_pt_metrics", {})
            max_l = metrics.get("max_rom_left", 0)
            max_r = metrics.get("max_rom_right", 0)
            max_rom = max(max_l, max_r)
            mobility_score = min(50, (max_rom / 80.0) * 50)
            
            warns = data.get("warning_counts", {})
            lean_back = warns.get("lean_back_momentum", 0)
            stability_score = max(0, 50 - (lean_back * 10))

        elif ex_type == 'shoulder_press':
            avg_shoulder = data.get("avg_shoulder_angle", 0)
            mobility_score = min(50, (avg_shoulder / 145.0) * 50)
            
            joints = data.get("realtime_joints", {})
            max_l = joints.get("left_shoulder", avg_shoulder)
            max_r = joints.get("right_shoulder", avg_shoulder)

            warns = data.get("warning_counts", {})
            arm_balance = warns.get("arm_balance_issue", 0)
            body_not_straight = warns.get("body_not_straight", 0)
            stability_score = max(0, 50 - ((arm_balance + body_not_straight) * 10))

        elif ex_type == 'bicep_curl':
            min_elbow = data.get("avg_elbow_angle", 180)
            mobility_score = min(50, ((180 - min_elbow) / (180 - 50.0)) * 50)
            
            joints = data.get("realtime_joints", {})
            max_l = joints.get("left_shoulder", min_elbow)
            max_r = joints.get("right_shoulder", min_elbow)
            
            warns = data.get("warning_counts", {})
            elbow_not_close = warns.get("elbows_not_close_to_body", 0)
            stability_score = max(0, 50 - (elbow_not_close * 10))

        # 🌟 정자세 정확도 (하드코딩을 실제 데이터로 교체)
        posture_accuracy = data.get("good_posture_ratio", data.get("performance_stats", {}).get("good_posture_ratio", 80))

        # 🌟 좌우 비대칭도(Asymmetry) 계산: 두 팔의 차이를 백분율로 산출
        asym_diff = abs(max_l - max_r)
        highest_rom = max(max_l, max_r)
        asym_ratio = round((asym_diff / highest_rom) * 100, 1) if highest_rom > 0 else 0.0

        return {
            "mobility_score": round(mobility_score),
            "stability_score": round(stability_score),
            "posture_accuracy": posture_accuracy,
            "asymmetry_ratio": asym_ratio,
            "total_score": round(mobility_score) + round(stability_score)
        }

    # 🌟 OpenAI 프롬프트에 비대칭도와 정자세 비율 정보 추가
    def request_openai_analysis(self, data, report_scores):
        try:
            exercise = data.get("exercise_type", "운동")
            rep = data.get("rep_count", 0)
            total = report_scores.get("total_score", 0)
            mob = report_scores.get("mobility_score", 0)
            stab = report_scores.get("stability_score", 0)
            posture = report_scores.get("posture_accuracy", 0)
            asym = report_scores.get("asymmetry_ratio", 0)

            prompt = f"""
            당신은 시니어 헬스케어 전문 AI 로봇 트레이너입니다. 어르신의 이번 '{exercise}' 운동 세션이 종료되었습니다.
            총 {rep}회를 수행했으며, 종합 점수는 100점 만점에 {total}점입니다. 
            (세부 지표: 관절 가동성 {mob}/50점, 자세 안정성 {stab}/50점, 정자세 정확도 {posture}%, 좌우 팔 비대칭도 {asym}%)
            
            이 임상적 리포트 데이터를 바탕으로, 어르신에게 따뜻하고 격려하는 말투로 이번 운동 총평과 개선점 1가지를 합쳐 50자 이내로 코멘트해주세요.
            """

            response = self.ai_client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            
            ai_comment = response.choices[0].message.content.strip()
            self.get_logger().info(f"🤖 최종 AI 분석 완료: {ai_comment}")

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
            
            db_path = f'{exercise_type}_sessions/{session_key}'
            db_ref = db.reference(db_path)
            db_ref.set(data)
            
            live_ref = db.reference('live_current_session')
            live_ref.update(data)

            is_finished = data.get("is_finished", False)
            
            if is_finished and not self.is_analysis_completed:
                self.get_logger().info("🏁 운동 세션 종료 감지! 최종 AI 리포트 생성을 시작합니다...")
                self.is_analysis_completed = True
                threading.Thread(target=self.request_openai_analysis, args=(data, report_scores)).start()

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