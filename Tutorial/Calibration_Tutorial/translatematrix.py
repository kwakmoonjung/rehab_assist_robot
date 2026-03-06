import json
from scipy.spatial.transform import Rotation
import numpy as np
import cv2
from scipy.linalg import sqrtm
from numpy.linalg import inv

# ===============================
# 1. Robot pose → Transformation
# ===============================

def get_robot_pose_matrix(x, y, z, rx, ry, rz):

    R = Rotation.from_euler('ZYZ', [rx, ry, rz], degrees=True).as_matrix()

    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = [x,y,z]

    return T


# ===============================
# 2. Checkerboard detection
# ===============================

def find_checkerboard_pose(image, board_size, square_size, camera_matrix, dist_coeffs):

    objp = np.zeros((board_size[0]*board_size[1],3),np.float32)

    objp[:,:2] = np.mgrid[
        0:board_size[0],
        0:board_size[1]
    ].T.reshape(-1,2)*square_size


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        board_size,
        flags=
        cv2.CALIB_CB_ADAPTIVE_THRESH+
        cv2.CALIB_CB_FAST_CHECK+
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not found:
        return None,None


    corners_sub = cv2.cornerSubPix(
        gray,
        corners,
        (11,11),
        (-1,-1),
        criteria=(
            cv2.TERM_CRITERIA_EPS+
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
        return None,None


    R,_ = cv2.Rodrigues(rvec)

    return R,tvec


# ===============================
# 3. Camera Intrinsic Calibration
# ===============================

def calibrate_camera_from_chessboard(image_paths, board_size, square_size):

    objp = np.zeros((board_size[0]*board_size[1],3),np.float32)

    objp[:,:2] = np.mgrid[
        0:board_size[0],
        0:board_size[1]
    ].T.reshape(-1,2)*square_size


    obj_points=[]
    img_points=[]
    image_shape=None


    for fname in image_paths:

        img=cv2.imread(fname)
        if img is None:
            continue

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        if image_shape is None:
            image_shape=gray.shape[::-1]


        ret,corners=cv2.findChessboardCorners(gray,board_size,None)

        if ret:

            corners_sub=cv2.cornerSubPix(
                gray,
                corners,
                (11,11),
                (-1,-1),
                (
                    cv2.TERM_CRITERIA_EPS+
                    cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001
                )
            )

            obj_points.append(objp)
            img_points.append(corners_sub)


    ret,camera_matrix,dist_coeffs,rvecs,tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_shape,
        None,
        None
    )

    return camera_matrix,dist_coeffs


# ===============================
# 4. Transformation matrix list
# ===============================

def compose_transformation_matrices(R_list,t_list):

    T_list=[]

    for R,t in zip(R_list,t_list):

        T=np.eye(4)

        T[:3,:3]=R
        T[:3,3]=np.ravel(t)

        T_list.append(T)

    return T_list


# ===============================
# 5. Log rotation
# ===============================

def logR(T):

    R=T[:3,:3]

    theta=np.arccos((np.trace(R)-1)/2)

    logr=np.array([
        R[2,1]-R[1,2],
        R[0,2]-R[2,0],
        R[1,0]-R[0,1]
    ])*theta/(2*np.sin(theta))

    return logr


# ===============================
# 6. AX = XB solver
# ===============================

def Calibrate(A,B):

    n_data=len(A)

    M=np.zeros((3,3))

    for i in range(n_data-1):

        alpha=logR(A[i])
        beta=logR(B[i])

        alpha2=logR(A[i+1])
        beta2=logR(B[i+1])


        alpha3=np.cross(alpha,alpha2)
        beta3=np.cross(beta,beta2)


        M+=np.outer(beta,alpha)
        M+=np.outer(beta2,alpha2)
        M+=np.outer(beta3,alpha3)


    theta=np.dot(sqrtm(inv(M.T@M)),M.T)


    C=np.zeros((3*n_data,3))
    d=np.zeros((3*n_data,1))


    for i in range(n_data):

        rot_a=A[i][:3,:3]

        trans_a=A[i][:3,3]
        trans_b=B[i][:3,3]

        C[3*i:3*i+3,:]=np.eye(3)-rot_a

        d[3*i:3*i+3,0]=trans_a-theta@trans_b


    b_x=inv(C.T@C)@(C.T@d)

    return theta,b_x


# ===============================
# 7. MAIN
# ===============================

if __name__=="__main__":

    data=json.load(open("data/calibrate_data.json"))

    robot_poses=np.array(data["poses"])

    image_paths=["data/"+d for d in data["file_name"]]


    checkerboard_size=(10,7)
    square_size=25


    camera_matrix,dist_coeffs = calibrate_camera_from_chessboard(
        image_paths,
        checkerboard_size,
        square_size
    )


    R_gripper2base_list=[]
    t_gripper2base_list=[]

    R_cam2checker_list=[]
    t_cam2checker_list=[]


    for img_path,pose in zip(image_paths,robot_poses):

        T_base2gripper=get_robot_pose_matrix(*pose)

        T_gripper2base=inv(T_base2gripper)

        R_gripper2base_list.append(T_gripper2base[:3,:3])
        t_gripper2base_list.append(T_gripper2base[:3,3].reshape(-1,1))


        image=cv2.imread(img_path)

        R_cam2checker,t_cam2checker = find_checkerboard_pose(
            image,
            checkerboard_size,
            square_size,
            camera_matrix,
            dist_coeffs
        )

        if R_cam2checker is None:
            continue


        R_cam2checker_list.append(R_cam2checker)
        t_cam2checker_list.append(t_cam2checker.reshape(-1,1))


    T_gripper2base_list = compose_transformation_matrices(
        R_gripper2base_list,
        t_gripper2base_list
    )

    T_cam2checker_list = compose_transformation_matrices(
        R_cam2checker_list,
        t_cam2checker_list
    )


    A_list=[]
    B_list=[]


    num_pairs=min(len(T_gripper2base_list),len(T_cam2checker_list))


    for i in range(num_pairs-1):

        A_i = inv(T_gripper2base_list[i]) @ T_gripper2base_list[i+1]

        B_i = inv(T_cam2checker_list[i]) @ T_cam2checker_list[i+1]

        A_list.append(A_i)
        B_list.append(B_i)


    theta,b_x = Calibrate(A_list,B_list)


    T_cam2base=np.eye(4)

    T_cam2base[:3,:3]=theta
    T_cam2base[:3,3]=b_x.flatten()


    print("Camera → Base Transform")
    print(T_cam2base)


    np.save("T_cam2base.npy",T_cam2base)