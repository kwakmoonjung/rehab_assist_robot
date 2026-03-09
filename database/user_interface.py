#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # [중요] 토픽 수신을 위해 추가
import firebase_admin
from firebase_admin import credentials, db

# ==========================================
# Firebase Realtime DB 설정 (rehab-aa1ee 프로젝트)
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIREBASE_KEY_PATH = os.path.join(CURRENT_DIR, "serviceAccountKey.json")
FIREBASE_DB_URL = "https://rehab-aa1ee-default-rtdb.firebaseio.com/"

class RehabUserInterface(Node):
    def __init__(self):
        super().__init__('rehab_user_interface')
        
        # 1. Firebase 초기화
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
            self.get_logger().info("🔥 Firebase 연동 성공! 클라우드 브릿지 가동.")
        except Exception as e:
            self.get_logger().error(f"Firebase 초기화 실패: {e}")

        # 2. [핵심 수정] 파일 감지 타이머 대신 토픽 구독자(Subscriber) 생성
        # pose_tracking_lr.py가 쏘는 '/exercise_result' 토픽을 기다립니다.
        self.subscription = self.create_subscription(
            String,
            '/exercise_result',
            self.exercise_result_callback,
            10
        )
        self.get_logger().info("📡 '/exercise_result' 토픽 구독 시작...")

    def exercise_result_callback(self, msg):
        """
        YOLO 분석 노드로부터 실시간 JSON 문자열을 받아 Firebase로 전송합니다.
        """
        try:
            # 수신된 문자열 메시지를 JSON 객체로 파싱
            data = json.loads(msg.data)
            
            # 세션 시작 시간을 Key로 사용 (Firebase 트리 구조 유지)
            # 예: 2026-03-09 14:52:46 -> 20260309_145246
            raw_start_time = data.get("session_started_at", "default")
            session_key = raw_start_time.replace("-", "").replace(":", "").replace(" ", "_")
            
            # Firebase 'lateral_raise_sessions' 하위의 해당 세션 노드에 실시간 업데이트
            db_ref = db.reference(f'lateral_raise_sessions/{session_key}')
            db_ref.set(data)
            
            # [디버깅용] 데이터 수신 로그 (너무 자주 뜨면 주석 처리하세요)
            # self.get_logger().info(f"🚀 Cloud Sync: {data.get('last_updated_at')} | Reps: {data.get('rep_count')}")

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
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()