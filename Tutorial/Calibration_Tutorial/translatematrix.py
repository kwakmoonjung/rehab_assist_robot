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
# 3. Checkerboard detection
# ===============================
def find_checkerboard_pose(image, board_size, square_size, camera_matrix, dist_coeffs):
    objp = create_checkerboard_object_points(board_size, square_size)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 너무 엄격한 SB 대신 유연하고 범용적인 표준 함수로 변경
    ret, corners = cv2.findChessboardCorners(
        gray,
        board_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not ret or corners is None:
        return None, None, None

    # 서브픽셀 단위로 정밀도 향상
    corners_sub = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners_sub,
        camera_matrix,
        dist_coeffs
    )

    if not ok:
        return None, None, None

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, corners_sub


# ===============================
# 4. Camera intrinsic calibration
# ===============================
# [수정된 부분] 함수 파라미터에 저장 경로인 success_dir, fail_dir 추가
def calibrate_camera(image_paths, board_size, square_size, success_dir, fail_dir):
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
            
        # [추가된 부분] 파일 이름 추출 (디버그 저장용)
        base_name = "intrinsic_" + os.path.basename(fname)

        # SB 대신 유연한 표준 함수로 변경
        ret, corners = cv2.findChessboardCorners(
            gray,
            board_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret and corners is not None:
            # 서브픽셀 정밀도 적용
            corners_sub = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            obj_points.append(objp)
            img_points.append(corners_sub)
            
            # [추가된 부분] 에러로 종료되기 전에 성공한 이미지 즉시 저장
            save_debug_image(img, os.path.join(success_dir, base_name), board_size, corners_sub, detected=True)
        else:
            # [추가된 부분] 에러로 종료되기 전에 실패한 이미지 즉시 저장 (눈으로 원인 파악 가능)
            save_debug_image(img, os.path.join(fail_dir, base_name), detected=False)

    # 이 곳에서 5장 이상 못 찾으면 에러를 띄우고 종료되어 버림
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

    # 내부 코너 수 기준
    checkerboard_size = (10, 6)
    square_size = 75.0  # mm

    # debug folder
    debug_root = "debug_detect"
    success_dir = os.path.join(debug_root, "success")
    fail_dir = os.path.join(debug_root, "fail")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    # 1) camera intrinsic
    # [수정된 부분] 캘리브레이션 함수에 success_dir, fail_dir 넘겨주기
    camera_matrix, dist_coeffs, intrinsic_ok_count = calibrate_camera(
        image_paths,
        checkerboard_size,
        square_size,
        success_dir,
        fail_dir
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

        T_base2gripper = get_robot_pose_matrix(*pose)
        T_gripper2base = inv(T_base2gripper)

        R_gripper2base.append(T_gripper2base[:3, :3].astype(np.float64))
        t_gripper2base.append(T_gripper2base[:3, 3].astype(np.float64))

        R_target2cam.append(R_cam2checker.astype(np.float64))
        t_target2cam.append(t_cam2checker.flatten().astype(np.float64))

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

    print(f"[INFO] success folder: {success_dir}")
    print(f"[INFO] fail folder   : {fail_dir}")

    if success_count < 5:
        raise RuntimeError(
            "캘리브레이션 가능한 이미지가 너무 적음. "
            "checkerboard_size 또는 촬영 상태를 다시 확인해야 함."
        )

    # 3) hand-eye calibration
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_cam2base = np.eye(4, dtype=np.float64)
    T_cam2base[:3, :3] = R_cam2base
    T_cam2base[:3, 3] = t_cam2base.flatten()

    print("\nCamera -> Robot Base Transform\n")
    print(T_cam2base)
    print("\nRotation determinant:", np.linalg.det(R_cam2base))

    np.save("T_cam2base.npy", T_cam2base)
    print("\n[INFO] saved: T_cam2base.npy")