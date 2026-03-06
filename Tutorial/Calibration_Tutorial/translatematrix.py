import json
import os
import numpy as np
import cv2
from numpy.linalg import inv

# ===============================
# 1. Robot pose -> Transformation
# ===============================
def get_robot_pose_matrix(x, y, z, rx, ry, rz):
    rvec = np.deg2rad([rx, ry, rz]).astype(np.float64)
    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


# ===============================
# 2. Checkerboard object points
# ===============================
def create_checkerboard_object_points(board_size, square_size):
    cols, rows = board_size  # 내부 코너 수
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return objp


# ===============================
# 3. Checkerboard detection (수정됨: 안정적인 기본 알고리즘 + 서브픽셀 보정)
# ===============================
def find_checkerboard_pose(image, board_size, square_size, camera_matrix, dist_coeffs):
    objp = create_checkerboard_object_points(board_size, square_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 인식률이 훨씬 좋은 기본 함수와 플래그 사용
    ret, corners = cv2.findChessboardCorners(
        gray,
        board_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    )

    if not ret or corners is None:
        return None, None, None

    # 검출된 코너의 위치를 소수점(픽셀 이하) 단위로 정밀하게 보정 (좌표 정확도 대폭 상승)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners_refined,
        camera_matrix,
        dist_coeffs
    )

    if not ok:
        return None, None, None

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, corners_refined


# ===============================
# 4. Camera intrinsic calibration (수정됨)
# ===============================
def calibrate_camera(image_paths, board_size, square_size):
    objp = create_checkerboard_object_points(board_size, square_size)

    obj_points = []
    img_points = []
    image_shape = None

    for fname in image_paths:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray,
            board_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        )

        if ret and corners is not None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)

    # 이 곳에서 5장 이상 못 찾으면 에러를 띄우고 종료되어 버림 (디버그 폴더에 안 들어가는 원인)
    if len(obj_points) < 5:
        raise RuntimeError(
            f"카메라 캘리브레이션용 체커보드 검출 성공 수가 너무 적음: {len(obj_points)}"
        )

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_shape,
        None,
        None
    )

    if not ret:
        raise RuntimeError("calibrateCamera 실패")

    return camera_matrix, dist_coeffs, len(obj_points)


# ===============================
# 5. Save debug image
# ===============================
def save_debug_image(image, save_path, board_size=None, corners=None, detected=False):
    dbg = image.copy()

    if detected and corners is not None:
        cv2.drawChessboardCorners(dbg, board_size, corners, True)
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


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    data = json.load(open("data/calibrate_data.json"))

    robot_poses = np.array(data["poses"], dtype=np.float64)
    image_paths = ["data/" + d for d in data["file_name"]]

    # 내부 코너 수 (만약 보드를 새로 뽑으셨다면, 꼭 가로세로 교차점 개수를 세서 맞춰주세요!)
    checkerboard_size = (10, 6)
    square_size = 75.0  # mm

    # debug folder
    debug_root = "debug_detect"
    success_dir = os.path.join(debug_root, "success")
    fail_dir = os.path.join(debug_root, "fail")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    # 1) camera intrinsic
    camera_matrix, dist_coeffs, intrinsic_ok_count = calibrate_camera(
        image_paths,
        checkerboard_size,
        square_size
    )

    print(f"\n[INFO] intrinsic calibration usable images: {intrinsic_ok_count}")

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    success_count = 0
    fail_count = 0

    # 2) hand-eye data collection
    for img_path, pose in zip(image_paths, robot_poses):
        image = cv2.imread(img_path)
        base_name = os.path.basename(img_path)

        if image is None:
            print(f"[WARN] image load failed: {img_path}")
            fail_count += 1
            continue

        R_cam2checker, t_cam2checker, corners = find_checkerboard_pose(
            image,
            checkerboard_size,
            square_size,
            camera_matrix,
            dist_coeffs
        )

        if R_cam2checker is None:
            print(f"[FAIL] checkerboard detection failed: {img_path}")
            save_debug_image(
                image=image,
                save_path=os.path.join(fail_dir, base_name),
                detected=False
            )
            fail_count += 1
            continue

        # [핵심 수정] inv() 제거! pose 자체가 이미 Gripper to Base 입니다.
        T_gripper2base_matrix = get_robot_pose_matrix(*pose)

        # OpenCV calibrateHandEye는 (3,1) 형태의 벡터를 선호하므로 reshape 적용
        R_gripper2base.append(T_gripper2base_matrix[:3, :3].astype(np.float64))
        t_gripper2base.append(T_gripper2base_matrix[:3, 3].astype(np.float64).reshape(3, 1))

        R_target2cam.append(R_cam2checker.astype(np.float64))
        t_target2cam.append(t_cam2checker.astype(np.float64).reshape(3, 1))

        save_debug_image(
            image=image,
            save_path=os.path.join(success_dir, base_name),
            board_size=checkerboard_size,
            corners=corners,
            detected=True
        )
        success_count += 1

    print("\n[INFO] Data count check")
    print("successful pairs:", success_count)
    print("failed images    :", fail_count)

    if success_count < 5:
        raise RuntimeError(
            "캘리브레이션 가능한 이미지가 너무 적음. "
            "checkerboard_size 또는 촬영 상태를 다시 확인해야 함."
        )

    # 3) hand-eye calibration (Eye-in-Hand 환경)
    # R_cam2gripper를 반환합니다. (카메라 -> 그리퍼 변환 행렬)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # 행렬 결합
    T_gripper2camera = np.eye(4, dtype=np.float64)
    T_gripper2camera[:3, :3] = R_cam2gripper
    T_gripper2camera[:3, 3] = t_cam2gripper.flatten()

    print("\n[RESULT] Camera -> Gripper Transform (T_gripper2camera)\n")
    print(T_gripper2camera)
    print("\nRotation determinant:", np.linalg.det(R_cam2gripper))

    # [핵심 수정] 이전 로봇 구동 코드와 호환되게 파일명 및 데이터 의미 수정
    np.save("T_gripper2camera.npy", T_gripper2camera)
    print("\n[INFO] saved: T_gripper2camera.npy")