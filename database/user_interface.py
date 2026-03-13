#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime
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

        # 🌟 4. 플래너 분석 결과 구독
        self.planner_subscription = self.create_subscription(
            String,
            '/recommended_routine',
            self.planner_response_callback,
            10
        )
        
        # 🌟 5. [수정됨] AI 피드백 데이터 구독 (토픽명 변경: /session_ai_feedback)
        self.ai_comment_subscription = self.create_subscription(
            String,
            '/session_ai_feedback',
            self.ai_comment_callback,
            10
        )
        
        self.get_logger().info("📡 토픽 구독 시작 (PC1은 중계만 담당, AI 연산은 PC2에서 수행).")

    # 🌟 [수정됨] AI 코멘트 수신 및 Firebase 저장 콜백
    def ai_comment_callback(self, msg):
        try:
            # 텍스트 형태인지 JSON 형태인지 알 수 없으므로 안전하게 처리
            feedback_text = msg.data.strip()
            # 만약 JSON으로 들어온다면 {"feedback": "..."} 형태일 수 있으니 파싱 시도
            try:
                parsed = json.loads(feedback_text)
                if isinstance(parsed, dict) and "feedback" in parsed:
                    feedback_text = parsed["feedback"]
            except:
                pass # 순수 문자열이면 그냥 사용

            self.get_logger().info(f"💬 AI 종합 처방 수신 완료: {feedback_text}")
            
            # 1. 라이브 세션에 즉시 업데이트 (웹 리포트 표출용 - 키 이름을 session_ai_feedback으로 통일)
            live_ref = db.reference('live_current_session')
            live_ref.update({"session_ai_feedback": feedback_text})
            
            # 2. 영구 저장소에 업데이트 (나중에 플래너가 조회할 수 있도록 저장)
            if self.last_session_data:
                exercise_type = self.last_session_data.get("exercise_type", "unknown_exercise")
                raw_start_time = self.last_session_data.get("session_started_at", "default")
                
                if raw_start_time != "default":
                    date_only = raw_start_time.split(" ")[0] 
                    session_key = raw_start_time.replace("-", "").replace(":", "").replace(" ", "_")
                    
                    db_path = f'{self.current_user_id}/{date_only}/{exercise_type}/{session_key}'
                    db.reference(db_path).update({"session_ai_feedback": feedback_text})
                    self.get_logger().info(f"✅ AI 코멘트 영구 저장 완료: {db_path}")

        except Exception as e:
            self.get_logger().error(f"AI 코멘트 처리 중 에러: {e}")

    # 플래너 응답 수신 및 번호별 저장 콜백
    # 플래너 응답 수신 및 번호별 저장 콜백
    def planner_response_callback(self, msg):
        try:
            raw_text = msg.data.strip()
            self.get_logger().info(f"📥 수신된 원본 플래너 데이터: {raw_text[:100]}...")
            
            # 1. 먼저 정상적인 JSON 파싱을 시도합니다.
            try:
                # 작은따옴표로 들어올 경우를 대비해 큰따옴표로 치환하는 방어 로직
                clean_text = raw_text.replace("'", '"') if raw_text.startswith("{") else raw_text
                planner_data = json.loads(clean_text)
            except Exception as parse_error:
                # 2. JSON 파싱에 실패하면 강제 저장 모드 가동
                self.get_logger().warn(f"⚠️ JSON 파싱 실패. 강제 저장 모드 가동: {parse_error}")
                planner_data = {
                    "type": "recommended_routine",
                    "analysis": {"summary": raw_text}
                }
            
            request_type = planner_data.get("type", "unknown")
            
            # 에러 메시지만 달랑 왔다면 스킵
            if request_type in ["error", "unknown"] and "analysis" not in planner_data:
                return

            # 오늘 날짜 (2026-03-13 형식)
            date_only = datetime.now().strftime("%Y-%m-%d")
            
            # 경로: user_id/date/planner
            db_path = f"{self.current_user_id}/{date_only}/planner"
            planner_ref = db.reference(db_path)
            
            # 🌟 [핵심 수정] 무조건 가장 큰 번호를 찾아서 +1 하도록 완벽하게 변경
            existing_planners = planner_ref.get()
            next_idx = 1
            
            if existing_planners and isinstance(existing_planners, dict):
                indices = []
                for key in existing_planners.keys():
                    if key.startswith("planner_"):
                        try:
                            # "planner_4"에서 "4"만 추출하여 숫자로 변환
                            idx = int(key.split("_")[1])
                            indices.append(idx)
                        except Exception:
                            pass
                
                # 가장 큰 숫자(예: 4)를 찾았다면 그 다음 번호(5)로 지정
                if indices:
                    next_idx = max(indices) + 1
            elif isinstance(existing_planners, list):
                # 파이어베이스가 가끔 리스트로 반환할 때를 대비한 방어 코드
                next_idx = len(existing_planners)
                if existing_planners[0] is None: # 인덱스 0이 비어있는 경우 보정
                    next_idx = len(existing_planners)
                else:
                    next_idx = len(existing_planners) + 1

            planner_key = f"planner_{next_idx}"
            
            # 최종 저장
            planner_ref.child(planner_key).set(planner_data)
            self.get_logger().info(f"✅ 플래너 저장 완료: {db_path}/{planner_key}")
            
        except Exception as e:
            self.get_logger().error(f"플래너 저장 중 치명적 에러: {e}")

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