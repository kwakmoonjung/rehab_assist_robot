import json
import numpy as np
import cv2
from numpy.linalg import inv

# ===============================
# Robot pose → Transformation
# ===============================

def get_robot_pose_matrix(x, y, z, rx, ry, rz):

    rvec = np.deg2rad([rx, ry, rz])
    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = [x,y,z]

    return T


# ===============================
# Checkerboard detection
# ===============================

def find_checkerboard_pose(image, board_size, square_size, camera_matrix, dist_coeffs):

    cols, rows = board_size

    objp = np.zeros((cols*rows,3),np.float32)
    objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)*square_size

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, board_size)

    if not ret:
        return None, None

    corners_sub = cv2.cornerSubPix(
        gray,
        corners,
        (11,11),
        (-1,-1),
        (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
    )

    ret, rvec, tvec = cv2.solvePnP(
        objp,
        corners_sub,
        camera_matrix,
        dist_coeffs
    )

    if not ret:
        return None, None

    R,_ = cv2.Rodrigues(rvec)

    return R, tvec


# ===============================
# Camera intrinsic calibration
# ===============================

def calibrate_camera(image_paths, board_size, square_size):

    cols, rows = board_size

    objp = np.zeros((cols*rows,3),np.float32)
    objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)*square_size

    obj_points=[]
    img_points=[]
    image_shape=None

    for fname in image_paths:

        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_shape is None:
            image_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, board_size)

        if ret:

            corners_sub = cv2.cornerSubPix(
                gray,
                corners,
                (11,11),
                (-1,-1),
                (
                    cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001
                )
            )

            obj_points.append(objp)
            img_points.append(corners_sub)

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_shape,
        None,
        None
    )

    return camera_matrix, dist_coeffs


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":

    data = json.load(open("data/calibrate_data.json"))

    robot_poses = np.array(data["poses"])
    image_paths = ["data/" + d for d in data["file_name"]]

    checkerboard_size = (10,6)
    square_size = 75


    # Camera intrinsic
    camera_matrix, dist_coeffs = calibrate_camera(
        image_paths,
        checkerboard_size,
        square_size
    )


    R_gripper2base = []
    t_gripper2base = []

    R_target2cam = []
    t_target2cam = []


    for img_path, pose in zip(image_paths, robot_poses):

        image = cv2.imread(img_path)

        R_cam2checker, t_cam2checker = find_checkerboard_pose(
            image,
            checkerboard_size,
            square_size,
            camera_matrix,
            dist_coeffs
        )

        if R_cam2checker is None:
            print("checkerboard detection failed:", img_path)
            continue


        T_base2gripper = get_robot_pose_matrix(*pose)
        T_gripper2base = inv(T_base2gripper)

        R_gripper2base.append(T_gripper2base[:3,:3])
        t_gripper2base.append(T_gripper2base[:3,3])

        R_target2cam.append(R_cam2checker)
        t_target2cam.append(t_cam2checker.flatten())


    # ===============================
    # DEBUG CHECK
    # ===============================

    print("\nData count check")
    print("robot poses:", len(R_gripper2base))
    print("checkerboard poses:", len(R_target2cam))


    if len(R_gripper2base) < 5:
        print("\nERROR: Not enough calibration data")
        exit()


    # ===============================
    # Hand-Eye Calibration
    # ===============================

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )


    T_cam2base = np.eye(4)

    T_cam2base[:3,:3] = R_cam2base
    T_cam2base[:3,3] = t_cam2base.flatten()


    print("\nCamera → Robot Base Transform\n")
    print(T_cam2base)

    print("\nRotation determinant:", np.linalg.det(R_cam2base))


    np.save("T_cam2base.npy", T_cam2base)