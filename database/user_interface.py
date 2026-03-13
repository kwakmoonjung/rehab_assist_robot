#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime  # 🌟 [추가] 날짜 및 번호 매기기를 위해 필요
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  
import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv
from ament_index_python.packages import get_package_share_directory

package_path = get_package_share_directory("rehab_assist_robot")

env_path = os.path.join(package_path, "resource", ".env")
load_dotenv(dotenv_path=env_path)

FIREBASE_KEY_PATH = os.path.join(package_path, "resource", "serviceAccountKey.json")
FIREBASE_DB_URL = "https://rehab-aa1ee-default-rtdb.asia-southeast1.firebasedatabase.app/"

class RehabUserInterface(Node):
    def __init__(self):
        super().__init__('rehab_user_interface')
        
        # Firebase 연동
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
            self.get_logger().info("🔥 Firebase 연동 성공! 클라우드 브릿지 가동.")
            
            # 노드 시작 시 UI에 실시간 연동 중 상태 알림
            db.reference('live_current_session').update({
                "system_status": "STANDBY",
                "last_feedback": "로봇 시스템이 준비되었습니다. 명령을 기다립니다."
            })
        except Exception as e:
            self.get_logger().error(f"Firebase 초기화 실패: {e}")

        # 분석 상태 관리 및 최신 데이터 저장을 위한 변수
        self.is_analysis_completed = False
        self.last_session_data = None
        self.last_report_scores = None
        self.current_user_id = "unknown_user" 

        # 🌟 1. 센서 데이터 구독
        self.subscription = self.create_subscription(
            String,
            '/exercise_result',
            self.exercise_result_callback,
            10
        )
        
        # 🌟 2. 시스템 명령어 구독
        self.cmd_subscription = self.create_subscription(
            String,
            '/system_command',
            self.system_command_callback,
            10
        )

        # 🌟 3. 사용자 인식 데이터 구독
        self.user_subscription = self.create_subscription(
            String,
            '/recognized_user',
            self.recognized_user_callback,
            10
        )

        # 🌟 4. [업데이트] 플래너 분석 결과 구독
        self.planner_subscription = self.create_subscription(
            String,
            '/exercise_planner/response',
            self.planner_response_callback,
            10
        )
        
        self.get_logger().info("📡 토픽 구독 시작.")

    # 🌟 [추가] 플래너 응답 수신 및 번호별 저장 콜백
    def planner_response_callback(self, msg):
        try:
            planner_data = json.loads(msg.data)
            request_type = planner_data.get("type", "unknown")
            
            if request_type in ["error", "unknown"]:
                return

            # 오늘 날짜 (2026-03-13 형식)
            date_only = datetime.now().strftime("%Y-%m-%d")
            
            # 경로: user_id/date/planner
            db_path = f"{self.current_user_id}/{date_only}/planner"
            planner_ref = db.reference(db_path)
            
            # 기존 데이터 개수 확인하여 번호 부여 (planner_1, planner_2...)
            existing_planners = planner_ref.get()
            next_idx = 1 if existing_planners is None else len(existing_planners) + 1
            planner_key = f"planner_{next_idx}"
            
            planner_ref.child(planner_key).set(planner_data)
            self.get_logger().info(f"✅ 플래너 저장 완료: {db_path}/{planner_key}")
            
        except Exception as e:
            self.get_logger().error(f"플래너 저장 중 에러: {e}")

    # 사용자 인식 콜백 함수
    def recognized_user_callback(self, msg):
        user_id = msg.data.strip()
        if user_id:
            self.current_user_id = user_id

    # 시스템 명령어 처리 콜백 함수
    def system_command_callback(self, msg):
        command = msg.data.strip()
        self.get_logger().info(f"시스템 명령어 수신: {command}")
        
        live_ref = db.reference('live_current_session')
        
        if command == 'START_EXERCISE':
            self.is_analysis_completed = False
            self.last_session_data = None
            self.last_report_scores = None
            
            live_ref.update({"system_status": "START_EXERCISE"})
            self.get_logger().info("▶️ Firebase 상태 업데이트 완료: START_EXERCISE")
            
        elif command == 'END_EXERCISE':
            live_ref.update({"system_status": "END_EXERCISE"})
            self.get_logger().info("🏁 END_EXERCISE 명령 감지! (PC2에서 AI 분석 결과를 Firebase에 올려줄 때까지 UI 대기)")
                
        elif command == 'REPORT_EXERCISE':
            live_ref.update({"system_status": "REPORT_EXERCISE"})
            self.get_logger().info("📊 리포트 출력 명령 수신.")
            
        elif command == 'CORRECTION':
            live_ref.update({"system_status": "CORRECTION"})
            self.get_logger().info("🛠️ 자세 교정 명령 수신.")

    # 정량화 및 비대칭도 산출 로직
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
            # 🌟 [핵심 수정] 평균(avg) 대신 최대 수축 각도(min)를 최우선으로 가져옵니다!
            min_elbow = data.get("min_elbow_angle", data.get("avg_elbow_angle", 180))
            mobility_score = max(0, min(50, ((180 - min_elbow) / (180 - 50.0)) * 50))
            
            joints = data.get("realtime_joints", {})
            max_l = joints.get("left_shoulder", min_elbow)
            max_r = joints.get("right_shoulder", min_elbow)
            
            warns = data.get("warning_counts", {})
            elbow_not_close = warns.get("elbows_not_close_to_body", 0)
            stability_score = max(0, 50 - (elbow_not_close * 10))

        # 정자세 정확도
        posture_accuracy = data.get("good_posture_ratio", data.get("performance_stats", {}).get("good_posture_ratio", 80))

        # 비대칭도
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

    # 센서 데이터 수신 콜백 함수
    def exercise_result_callback(self, msg):
        try:
            data = json.loads(msg.data)
            exercise_type = data.get("exercise_type", "unknown_exercise")
            
            report_scores = self.calculate_report_scores(data)
            data["report_scores"] = report_scores
            
            raw_start_time = data.get("session_started_at", "default")
            date_only = raw_start_time.split(" ")[0] 
            session_key = raw_start_time.replace("-", "").replace(":", "").replace(" ", "_")
            
            db_path = f'{self.current_user_id}/{date_only}/{exercise_type}/{session_key}'
            db_ref = db.reference(db_path)
            db_ref.set(data)
            
            live_ref = db.reference('live_current_session')
            live_ref.update(data)

            self.last_session_data = data
            self.last_report_scores = report_scores

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