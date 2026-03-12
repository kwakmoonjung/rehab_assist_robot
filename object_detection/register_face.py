#!/usr/bin/env python3
import os
import pickle
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class FaceRegisterNode(Node):
    def __init__(self, target_name):
        super().__init__("face_register_node")
        self.target_name = target_name
        self.bridge = CvBridge()
        
        self.db_dir = os.path.expanduser("~/cobot_ws/src/cobot2_ws/rehab_assist_robot/resource")
        self.db_path = os.path.join(self.db_dir, "face_db.pkl")
        os.makedirs(self.db_dir, exist_ok=True)

        self.face_db = self.load_face_db()
        
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_path)

        self.register_buffer = []
        self.target_count = 5

        self.image_sub = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.image_callback, 10
        )
        
        self.get_logger().info(f"📸 [{self.target_name}]님의 얼굴 등록을 시작합니다. 카메라를 정면으로 바라봐주세요...")

    def load_face_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                return pickle.load(f)
        return {}

    def save_face_db(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.face_db, f)
        self.get_logger().info(f"💾 {self.db_path} 에 성공적으로 저장되었습니다.")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))

        if len(faces) == 0: return

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face_crop = frame[y:y+h, x:x+w]
        
        gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_crop, (100, 100))
        vector = resized.astype(np.float32).flatten()
        norm = np.linalg.norm(vector)
        if norm == 0: return

        vector = vector / norm
        self.register_buffer.append(vector)
        self.get_logger().info(f"데이터 수집 중... ({len(self.register_buffer)}/{self.target_count})")

        if len(self.register_buffer) >= self.target_count:
            avg_vector = np.mean(np.stack(self.register_buffer, axis=0), axis=0)
            avg_vector = avg_vector / np.linalg.norm(avg_vector)

            self.face_db[self.target_name] = avg_vector
            self.save_face_db()
            
            self.get_logger().info(f"🎉 [{self.target_name}]님 얼굴 등록 완료! 프로그램을 종료합니다.")
            sys.exit(0)

def main():
    if len(sys.argv) < 2:
        print("사용법: python3 register_face.py [등록할_이름]")
        return
    target_name = sys.argv[1]
    
    rclpy.init()
    node = FaceRegisterNode(target_name)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()