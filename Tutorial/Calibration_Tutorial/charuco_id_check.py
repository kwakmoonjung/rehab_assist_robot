import cv2
import os

# 가지고 계신 데이터 중 화질이 좋은 사진 하나를 지정해주세요.
img_path = "/home/rokey/Pictures/Screenshots/Screenshot from 2026-03-07 16-55-55.png"
image = cv2.imread(img_path)

if image is None:
    print("이미지 경로를 다시 확인해주세요!")
    exit()

# 기존과 동일한 6x6 딕셔너리
try:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
except AttributeError:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 마커 검출
if hasattr(cv2.aruco, 'ArucoDetector'):
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
else:
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

# 사진 위에 마커 ID 그리기 및 저장
if ids is not None:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    cv2.imwrite("check_marker_ids.jpg", image)
    print(f"총 {len(ids)}개의 마커를 찾았습니다. 생성된 'check_marker_ids.jpg'를 확인해주세요!")
else:
    print("마커를 찾지 못했습니다.")