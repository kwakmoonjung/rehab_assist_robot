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

        # =========================
        # 기본 설정
        # =========================
        self.bridge = CvBridge()

        self.camera_topic = "/camera/camera/color/image_raw"
        self.db_dir = os.path.expanduser("~/rehab_face_db")
        self.db_path = os.path.join(self.db_dir, "face_db.pkl")

        os.makedirs(self.db_dir, exist_ok=True)

        # 인식 관련 파라미터
        self.match_threshold = 0.43   # 작을수록 더 엄격
        self.publish_interval = 1.0   # 같은 결과를 너무 자주 안 보내기 위한 간격(sec)

        self.last_publish_time = 0.0
        self.last_recognized_user = None

        # 등록 모드
        self.pending_register_name = None
        self.register_buffer = []      # 여러 프레임 encoding 평균용
        self.register_target_count = 5 # 5프레임 모아서 등록

        # 얼굴 DB 로드
        self.face_db = self.load_face_db()

        # OpenCV Haar Cascade (얼굴 영역 검출)
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_path)

        # =========================
        # ROS2 통신
        # =========================
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )

        self.face_cmd_sub = self.create_subscription(
            String,
            "/face_command",
            self.face_command_callback,
            10
        )

        self.recognized_user_pub = self.create_publisher(
            String,
            "/recognized_user",
            10
        )

        self.face_status_pub = self.create_publisher(
            String,
            "/face_status",
            10
        )

        self.get_logger().info("😀 FaceRecognitionNode started.")
        self.get_logger().info(f"DB path: {self.db_path}")
        self.get_logger().info(f"Loaded users: {list(self.face_db.keys())}")

    # =========================
    # DB 처리
    # =========================
    def load_face_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                self.get_logger().error(f"얼굴 DB 로드 실패: {e}")
        return {}

    def save_face_db(self):
        try:
            with open(self.db_path, "wb") as f:
                pickle.dump(self.face_db, f)
            self.get_logger().info("얼굴 DB 저장 완료")
        except Exception as e:
            self.get_logger().error(f"얼굴 DB 저장 실패: {e}")

    # =========================
    # 명령 처리
    # =========================
    def face_command_callback(self, msg):
        command = msg.data.strip()

        # 예:
        # REGISTER:jintae
        # RESET
        # REMOVE:jintae
        if command.startswith("REGISTER:"):
            name = command.replace("REGISTER:", "").strip()
            if not name:
                self.publish_status("register_failed", "이름이 비어 있습니다.")
                return

            self.pending_register_name = name
            self.register_buffer = []
            self.publish_status("register_ready", f"{name} 등록 준비 완료. 카메라를 바라봐 주세요.")
            self.get_logger().info(f"얼굴 등록 시작: {name}")

        elif command.startswith("REMOVE:"):
            name = command.replace("REMOVE:", "").strip()
            if name in self.face_db:
                del self.face_db[name]
                self.save_face_db()
                self.publish_status("remove_success", f"{name} 삭제 완료")
            else:
                self.publish_status("remove_failed", f"{name} 사용자를 찾지 못했습니다.")

        elif command == "RESET":
            self.pending_register_name = None
            self.register_buffer = []
            self.last_recognized_user = None
            self.publish_status("reset", "얼굴 인식 상태 초기화 완료")

    # =========================
    # 이미지 콜백
    # =========================
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")
            return

        face_crop = self.extract_single_face(frame)
        if face_crop is None:
            return

        face_vector = self.make_face_vector(face_crop)
        if face_vector is None:
            return

        # 등록 모드 우선
        if self.pending_register_name is not None:
            self.handle_registration(face_vector)
            return

        # 평상시에는 계속 인식
        self.handle_recognition(face_vector)

    # =========================
    # 얼굴 검출
    # =========================
    def extract_single_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(120, 120)
        )

        if len(faces) == 0:
            return None

        # 가장 큰 얼굴 1개만 사용
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            return None

        return face_crop

    # =========================
    # 얼굴 벡터 생성
    # =========================
    def make_face_vector(self, face_crop):
        """
        아주 복잡한 딥러닝 임베딩 대신,
        구현 난이도 낮게 가기 위해 grayscale + resize + flatten 방식 사용.
        정확도는 face_recognition 라이브러리보다 낮지만 구조 확인용/빠른 테스트용으로는 좋음.
        """
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (100, 100))
            vector = resized.astype(np.float32).flatten()
            norm = np.linalg.norm(vector)
            if norm == 0:
                return None
            vector = vector / norm
            return vector
        except Exception as e:
            self.get_logger().error(f"얼굴 벡터 생성 실패: {e}")
            return None

    # =========================
    # 등록 처리
    # =========================
    def handle_registration(self, face_vector):
        self.register_buffer.append(face_vector)

        remain = self.register_target_count - len(self.register_buffer)
        self.get_logger().info(
            f"등록 중: {self.pending_register_name} ({len(self.register_buffer)}/{self.register_target_count})"
        )

        if remain > 0:
            return

        avg_vector = np.mean(np.stack(self.register_buffer, axis=0), axis=0)
        norm = np.linalg.norm(avg_vector)
        if norm != 0:
            avg_vector = avg_vector / norm

        self.face_db[self.pending_register_name] = avg_vector
        self.save_face_db()

        name = self.pending_register_name
        self.pending_register_name = None
        self.register_buffer = []

        self.publish_status("register_success", f"{name} 얼굴 등록이 완료되었습니다.")
        self.get_logger().info(f"얼굴 등록 완료: {name}")

    # =========================
    # 인식 처리
    # =========================
    def handle_recognition(self, face_vector):
        if not self.face_db:
            return

        best_name = None
        best_distance = 999.0

        for name, saved_vector in self.face_db.items():
            saved_vector = np.array(saved_vector, dtype=np.float32)
            dist = np.linalg.norm(face_vector - saved_vector)

            if dist < best_distance:
                best_distance = dist
                best_name = name

        if best_name is None:
            return

        if best_distance <= self.match_threshold:
            now = time.time()

            # 너무 자주 같은 결과를 퍼블리시하지 않기
            if (
                self.last_recognized_user == best_name
                and (now - self.last_publish_time) < self.publish_interval
            ):
                return

            self.last_recognized_user = best_name
            self.last_publish_time = now

            payload = {
                "user_id": best_name,
                "distance": float(best_distance),
                "recognized_at": now
            }

            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.recognized_user_pub.publish(msg)

            self.get_logger().info(
                f"인식 성공: user={best_name}, distance={best_distance:.4f}"
            )

    # =========================
    # 상태 메시지
    # =========================
    def publish_status(self, status, detail):
        payload = {
            "status": status,
            "detail": detail
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.face_status_pub.publish(msg)


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