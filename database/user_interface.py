#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  
import firebase_admin
from firebase_admin import credentials, db
from ament_index_python.packages import get_package_share_directory # [추가]

# ==========================================
# Firebase Realtime DB 설정 
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIREBASE_KEY_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "resource", "serviceAccountKey.json")

PACKAGE_DIR = get_package_share_directory('rehab_assist_robot') # [추가]
FIREBASE_KEY_PATH = os.path.join(PACKAGE_DIR, "resource", "serviceAccountKey.json") # [추가]

# 🌟 [수정1] 에러가 났던 미국 주소 대신, 정확한 아시아 서버 주소로 변경
FIREBASE_DB_URL = "https://rehab-aa1ee-default-rtdb.asia-southeast1.firebasedatabase.app/"

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

        self.subscription = self.create_subscription(
            String,
            '/exercise_result',
            self.exercise_result_callback,
            10
        )
        self.get_logger().info("📡 '/exercise_result' 통합 토픽 구독 시작...")

    def exercise_result_callback(self, msg):
        try:
            data = json.loads(msg.data)
            
            # 🌟 [수정2] JSON 데이터 안에서 지금 무슨 운동인지(exercise_type) 빼오기
            exercise_type = data.get("exercise_type", "unknown_exercise")
            
            raw_start_time = data.get("session_started_at", "default")
            session_key = raw_start_time.replace("-", "").replace(":", "").replace(" ", "_")
            
            # 🌟 [수정3] 하드코딩 탈피! '운동종목_sessions' 폴더로 동적 분류하여 영구 저장
            db_path = f'{exercise_type}_sessions/{session_key}'
            db_ref = db.reference(db_path)
            db_ref.set(data)
            
            # 🌟 [수정4] UI 전광판용 실시간 덮어쓰기 경로 (UI는 여기만 쳐다보면 됨)
            live_ref = db.reference('live_current_session')
            live_ref.set(data)

        except Exception as e:
            self.get_logger().error(f"데이터 처리 및 업로드 에러: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RehabUserInterface()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 🌟 노드가 종료될 때 Firebase 라이브 세션 데이터를 지우고 깔끔하게 퇴장합니다.
        try:
            db.reference('live_current_session').delete()
            node.get_logger().info("🧹 Firebase 라이브 데이터 초기화 완료.")
        except Exception:
            pass
            
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()