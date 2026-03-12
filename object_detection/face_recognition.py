#!/usr/bin/env python3
import os
import json
import pickle
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class FaceRecognitionNode(Node):
    def __init__(self):
        super().__init__("face_recognition_node")

        self.bridge = CvBridge()
        self.camera_topic = "/camera/camera/color/image_raw"
        
        self.db_dir = os.path.expanduser("~/cobot_ws/src/cobot2_ws/rehab_assist_robot/resource")
        self.db_path = os.path.join(self.db_dir, "face_db.pkl")

        self.match_threshold = 0.7  # [추가] match_threshold: 실시간 얼굴 벡터와 DB 내 얼굴 벡터 간의 동일인 판별을 위한 최대 허용 오차(거리) 기준값
        
        self.publish_interval = 1.0  # 1초에 한 번만 발행하도록 제한

        self.last_publish_time = 0.0
        self.last_recognized_user = None

        self.face_db = self.load_face_db()

        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_path)

        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        self.recognized_user_pub = self.create_publisher(String, "/recognized_user", 10)

        self.get_logger().info("실시간 얼굴 인식 노드 (경량화 버전) 시작!")

    def load_face_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                self.get_logger().error(f"DB 로드 실패: {e}")
        else:
            self.get_logger().warn("등록된 DB가 없습니다. 얼굴을 먼저 등록해주세요.")
        return {}

    def image_callback(self, msg):
        if not self.face_db: return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        face_crop = self.extract_single_face(frame)
        if face_crop is None: return

        face_vector = self.make_face_vector(face_crop)
        if face_vector is None: return

        self.handle_recognition(face_vector)

    def extract_single_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
        if len(faces) == 0: return None
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        return frame[y:y+h, x:x+w]

    def make_face_vector(self, face_crop):
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (100, 100))
            vector = resized.astype(np.float32).flatten()
            norm = np.linalg.norm(vector)
            if norm == 0: return None
            return vector / norm
        except Exception:
            return None

    def handle_recognition(self, face_vector):
        best_name = None
        best_distance = 999.0
        # [추가] best_distance: 실시간 측정된 얼굴 특징 벡터와 DB에 저장된 얼굴 특징 벡터 간의 가장 작은 유클리디안 거리(오차). 값이 0에 가까울수록 두 벡터가 유사함을 의미합니다.

        for name, saved_vector in self.face_db.items():
            saved_vector = np.array(saved_vector, dtype=np.float32)
            dist = np.linalg.norm(face_vector - saved_vector)

            if dist < best_distance:
                best_distance = dist
                best_name = name

        if best_name is not None:
            self.get_logger().info(f"[실시간 매칭 확인] 이름: {best_name}, 오차(거리): {best_distance:.4f}")

        if best_name is not None and best_distance <= self.match_threshold:
            now = time.time()
            
            # 토픽 발행에 대한 1초 쿨다운 로직 (로깅은 프레임마다 찍히고, 토픽 전송만 1초에 한 번)
            if self.last_recognized_user == best_name and (now - self.last_publish_time) < self.publish_interval:
                return

            self.last_recognized_user = best_name
            self.last_publish_time = now

            msg = String()
            msg.data = best_name
            self.recognized_user_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()