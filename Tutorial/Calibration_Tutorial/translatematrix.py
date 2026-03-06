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

    ret, corners = cv2.findChessboardCornersSB(
        gray,
        board_size,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    )

    if not ret or corners is None:
        return None, None, None

    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners,
        camera_matrix,
        dist_coeffs
    )

    if not ok:
        return None, None, None

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, corners


# ===============================
# 4. Camera intrinsic calibration
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

        ret, corners = cv2.findChessboardCornersSB(
            gray,
            board_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        )

        if ret and corners is not None:
            obj_points.append(objp)
            img_points.append(corners)

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