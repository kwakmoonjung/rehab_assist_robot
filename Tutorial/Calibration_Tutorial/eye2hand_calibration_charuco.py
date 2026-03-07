import json
from scipy.spatial.transform import Rotation
import numpy as np
import cv2
from scipy.linalg import sqrtm
from numpy.linalg import inv
import os # [추가] 경로 처리를 위한 모듈

# 1) 로봇 그리퍼의 절대 좌표를 행렬로 변환
def get_robot_pose_matrix(x, y, z, rx, ry, rz):
    R = Rotation.from_euler('ZYZ', [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

# 2) ChArUco 보드 포즈 검출
def find_charuco_pose(image, board, aruco_dict, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # OpenCV 최신 버전(4.7 이상) 호환성 대응
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    else:
        # 아루코 마커 검출 (구버전 API 완벽 호환)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
    
    if ids is not None and len(ids) > 0:
        # 체커보드 코너 보간
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board)
        
        # 최소 4개 이상의 코너가 보여야 3D 포즈 추정 가능
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) >= 4:
            valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, np.empty(1), np.empty(1))
            
            if valid:
                R, _ = cv2.Rodrigues(rvec)
                return R, tvec, charuco_corners, charuco_ids # [수정] 디버그를 위해 코너와 아이디 반환 추가
                
    return None, None, None, None # [수정] 반환 개수 일치

# 3) ChArUco 이미지를 이용한 카메라 보정 (Intrinsic)
def calibrate_camera_from_charuco(image_paths, board, aruco_dict):
    all_corners = []
    all_ids = []
    image_shape = None

    for fname in image_paths:
        img = cv2.imread(fname)
        if img is None:
            print(f"이미지 로드 실패 (경로 확인 필요): {fname}") 
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1]

        # OpenCV 최신 버전(4.7 이상) 호환성 대응
        if hasattr(cv2.aruco, 'ArucoDetector'):
            detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        
        if ids is not None and len(ids) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
                
            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)

    if len(all_corners) < 5: # [수정] 안정적인 캘리브레이션을 위해 최소 요구 개수 상향
        print(f"ChArUco 보드 코너를 충분히 찾지 못하였습니다. (현재: {len(all_corners)}장)")
        return None, None, None, None

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, image_shape, None, None)

    if not ret:
        print("캘리브레이션이 제대로 수렴하지 않았습니다.")
        return None, None, None, None

    return camera_matrix, dist_coeffs, rvecs, tvecs

# [추가] 디버그 이미지 저장 함수
def save_debug_image(image, save_path, board=None, charuco_corners=None, charuco_ids=None, detected=False):
    dbg = image.copy()

    if detected and charuco_corners is not None and charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(dbg, charuco_corners, charuco_ids, (0, 255, 0))
        label = "SUCCESS"
        color = (0, 255, 0)
    else:
        label = "FAIL"
        color = (0, 0, 255)

    cv2.putText(
        dbg,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA
    )
    cv2.imwrite(save_path, dbg)

# 4) 행렬 연산 및 Park & Martin 수학 로직
def compose_transformation_matrices(R_list, t_list):
    T_list = []
    for R, t in zip(R_list, t_list):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.ravel(t)
        T_list.append(T)
    return T_list

def logR(T):
    R = T[0:3, 0:3]
    theta = np.arccos((np.trace(R) - 1) / 2)
    logr = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) * theta / (2 * np.sin(theta))
    return logr

def Calibrate(A, B):
    n_data = len(A)
    M = np.zeros((3, 3))

    for i in range(n_data - 1):
        alpha  = logR(A[i])
        beta   = logR(B[i])
        alpha2 = logR(A[i + 1])
        beta2  = logR(B[i + 1])
        
        alpha3 = np.cross(alpha, alpha2)
        beta3  = np.cross(beta, beta2)
        
        M1 = np.dot(beta.reshape(3, 1), alpha.reshape(1, 3))
        M2 = np.dot(beta2.reshape(3, 1), alpha2.reshape(1, 3))
        M3 = np.dot(beta3.reshape(3, 1), alpha3.reshape(1, 3))
        
        M += M1 + M2 + M3

    theta = np.dot(sqrtm(inv(np.dot(M.T, M))), M.T)

    C = np.zeros((3 * n_data, 3))
    d = np.zeros((3 * n_data, 1))
    for i in range(n_data):
        rot_a = A[i][:3, :3]
        trans_a = A[i][:3, 3]
        trans_b = B[i][:3, 3]
        C[3 * i:3 * i + 3, :] = np.eye(3) - rot_a
        d[3 * i:3 * i + 3, 0] = trans_a - np.dot(theta, trans_b)
    
    b_x = np.dot(inv(np.dot(C.T, C)), np.dot(C.T, d))
    return theta, b_x

# ==========================================
# Main 실행부
# ==========================================
if __name__ == "__main__":
    # 지정된 절대 경로 변수 선언
    data_dir = "/home/rokey/cobot_ws/src/cobot2_ws/rehab_assist_robot/Tutorial/Calibration_Tutorial/data"
    
    # json 파일 절대 경로 설정 및 데이터 로드
    json_path = os.path.join(data_dir, "calibrate_data.json")
    data = json.load(open(json_path))

    robot_poses = np.array(data["poses"])
    robot_poses[:, :3] = robot_poses[:, :3]
    
    # 이미지 경로에 절대 경로 적용
    image_paths = [os.path.join(data_dir, d) for d in data["file_name"]]

    valid_indices = []
    for i, pose in enumerate(robot_poses):
        T_base2gripper = get_robot_pose_matrix(*pose)
        det_T = np.linalg.det(T_base2gripper)
        
        if np.abs(det_T) > 1e-6:
            valid_indices.append(i)

    robot_poses = robot_poses[valid_indices]
    image_paths = [image_paths[i] for i in valid_indices]

    # ---------------------------------------------------------
    # [수정 필수] 출력된 보드를 자로 직접 재서 아래 숫자를 바꿔주세요.
    # ---------------------------------------------------------
    squaresX = 8
    squaresY = 11
    square_size = 60.0  # (예: 30mm) 흑백 네모 1칸의 실제 길이
    marker_size = 45.0  # (예: 22mm) 내부 ArUco 마커 1칸의 실제 길이
    
    # 4.6 이하 버전 완벽 호환 딕셔너리 생성 명령어
    try:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)  # 내부 비트 패턴 6*6
    except AttributeError:
        # 혹시나 최신 버전일 경우를 대비한 안전장치
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        
    try:
        board = cv2.aruco.CharucoBoard_create(squaresX, squaresY, square_size, marker_size, aruco_dict)
    except AttributeError:
        # OpenCV 4.7 이상 버전을 위한 호환 코드
        board = cv2.aruco.CharucoBoard((squaresX, squaresY), square_size, marker_size, aruco_dict)
    # ---------------------------------------------------------

    # [추가] 디버그 폴더 생성
    debug_root = "debug_detect_charuco"
    success_dir = os.path.join(debug_root, "success")
    fail_dir = os.path.join(debug_root, "fail")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    print("카메라 내부 파라미터 캘리브레이션 시작...")
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera_from_charuco(
        image_paths, board, aruco_dict
    )

    if camera_matrix is None:
        print("종료합니다.")
        exit()
    print("성공!\n")

    R_gripper2base_list = []
    t_gripper2base_list = []
    R_camera2checker_list = []
    t_camera2checker_list = []
    R_checker2camera_list = []
    t_checker2camera_list = []

    success_count = 0 # [추가] 성공 카운트
    fail_count = 0    # [추가] 실패 카운트

    for img_path, pose in zip(image_paths, robot_poses):
        T_base2gripper = get_robot_pose_matrix(*pose)
        image = cv2.imread(img_path)
        base_name = os.path.basename(img_path) # [추가] 파일명 추출

        if image is None:
            print(f"[WARN] 이미지 로드 실패: {img_path}")
            fail_count += 1
            continue

        # [수정] 코너와 ID 반환받기
        R_cam2checker, t_cam2checker, charuco_corners, charuco_ids = find_charuco_pose(
            image, board, aruco_dict, camera_matrix, dist_coeffs
        )
        
        if R_cam2checker is None:
            print(f"[FAIL] 마커 검출 실패: {img_path}")
            # [추가] 실패 이미지 저장
            save_debug_image(
                image=image,
                save_path=os.path.join(fail_dir, base_name),
                detected=False
            )
            fail_count += 1
            continue

        # [추가] 성공 이미지 저장
        save_debug_image(
            image=image,
            save_path=os.path.join(success_dir, base_name),
            board=board,
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids,
            detected=True
        )
        success_count += 1

        T_gripper2base = np.linalg.inv(T_base2gripper)
        R_gripper2base = T_gripper2base[:3, :3]
        t_gripper2base = T_gripper2base[:3, 3]

        R_gripper2base_list.append(R_gripper2base.copy())
        t_gripper2base_list.append(t_gripper2base.reshape(-1, 1).copy())

        T_cam2checker = np.eye(4)
        T_cam2checker[:3, :3] = R_cam2checker
        T_cam2checker[:3, 3] = t_cam2checker.flatten()
        T_checker2cam = np.linalg.inv(T_cam2checker)

        R_checker2camera_list.append(T_checker2cam[:3, :3].copy())
        t_checker2camera_list.append(T_checker2cam[:3, 3].copy())

    print("\n[INFO] Data count check")
    print("successful pairs:", success_count)
    print("failed images    :", fail_count)

    T_gripper2base_list = compose_transformation_matrices(R_gripper2base_list, t_gripper2base_list)
    T_checker2cam_list = compose_transformation_matrices(R_checker2camera_list, t_checker2camera_list)
    
    A_list = []
    B_list = []
    num_pairs = min(len(T_gripper2base_list), len(T_checker2cam_list))

    for i in range(num_pairs - 1):
        A_i = np.dot(inv(T_gripper2base_list[i]), T_gripper2base_list[i + 1])
        B_i = np.dot(inv(T_checker2cam_list[i]), T_checker2cam_list[i + 1])
        A_list.append(A_i)
        B_list.append(B_i)

    # 최종 연산
    theta, b_x = Calibrate(A_list, B_list)
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = theta
    T_cam2base[:3, 3] = b_x.flatten()
    
    print("=======================================")
    print("[최종 캘리브레이션 결과 행렬 (T_cam2base)]")
    print("=======================================")
    print(T_cam2base)
    print("\n[위치 오프셋 (X, Y, Z mm)]")
    print(T_cam2base[:3, 3])
    np.save("T_cam2base.npy", T_cam2base)