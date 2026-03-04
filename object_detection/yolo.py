########## yolo.py ##########
import time
import rclpy
from ultralytics import YOLO
import numpy as np

class YoloModel:
    def __init__(self):
        # 모델 로드 (GPU 사용 권장)
        self.model = YOLO("yolov8n-pose.pt")
        
        # GPU 강제 사용 시도 (선택사항)
        # import torch
        # if torch.cuda.is_available():
        #     self.model.to('cuda')
    
    def get_best_detection(self, img_node, target=None):
        """
        1초 대기 없이, 현재 가장 최신 프레임 1장을 가져와 즉시 추론합니다.
        """
        # 1. 이미지 버퍼 비우기 (가장 최신 이미지를 얻기 위함)
        rclpy.spin_once(img_node)
        
        # 2. 현재 프레임 가져오기
        frame = img_node.get_color_frame()
        
        if frame is None:
            # 프레임이 없으면 잠깐 대기 후 리턴
            return None, None, None

        # 3. YOLO 추론 (verbose=False로 로그 제거)
        results = self.model(frame, verbose=False) 
        
        best_score = -1
        best_box = None
        best_kpts = None

        # 4. 결과 분석 (Person 클래스만 필터링)
        for res in results:
            if res.boxes:
                for i, box in enumerate(res.boxes):
                    score = box.conf.item()
                    cls = int(box.cls.item())
                    
                    # 0번 클래스(Person)이고 점수가 기존보다 높으면 선택
                    if cls == 0 and score > best_score: 
                        best_score = score
                        best_box = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                        
                        # Pose Keypoints 추출
                        if res.keypoints is not None:
                            best_kpts = res.keypoints.data[i].tolist() 

        if best_box is None:
            # print("No person detected.") # 터미널 도배 방지용 주석
            return None, None, None

        return best_box, best_score, best_kpts