import os
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point 
from ament_index_python.packages import get_package_share_directory

import DR_init
from robot_control.onrobot import RG 

# --- 로봇 설정 상수 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 100, 100 
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"
MIN_DEPTH = 50.0

# ==========================================================
# 1. 두산 API 전역(Global) 세팅 
# ==========================================================
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("dsr_robot_core_node", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, movel, get_current_posj, get_current_posx, mwait
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)


# ==========================================================
# 2. 운동별 로봇 보조 전략 (Strategy Pattern) 
# ==========================================================
class ExerciseStrategy:
    def execute_assist(self, td_coord):
        pass

class LateralRaiseStrategy(ExerciseStrategy):
    def __init__(self):
        self.Z_APPROACH_OFFSET = -40.0  # 팔꿈치 4cm 아래로 접근
        self.LIFT_OFFSET = 250.0        # 접근 위치에서 위로 25cm(250mm) 들어올림
        self.Y_INWARD_OFFSET = -100.0   # Y축으로 10cm(100mm) 더 들어가기   

    def execute_assist(self, td_coord):
        print("💪 [사레레/숄더프레스] 동적 보조 시작")

        current_posx = get_current_posx()[0] 
        target_pos = list(td_coord[:3]) + list(current_posx[3:])

        target_pos[1] += self.Y_INWARD_OFFSET

        # 1. 팔꿈치 밑으로 접근 (대기 위치)
        approach_pos = list(target_pos)
        approach_pos[2] = max(approach_pos[2] + self.Z_APPROACH_OFFSET, MIN_DEPTH)

        movel(approach_pos, vel=VELOCITY, acc=ACC)
        mwait()

        # 2. 대기 위치에서 정확히 LIFT_OFFSET 만큼만 위로 들어올리기
        lift_pos = list(approach_pos)
        lift_pos[2] += self.LIFT_OFFSET
        
        movel(lift_pos, vel=VELOCITY, acc=ACC)
        mwait()

class BicepCurlStrategy(ExerciseStrategy):
    def __init__(self):
        self.Y_BACK_OFFSET = 50.0 

    def execute_assist(self, td_coord):
        print("💪 [이두컬] 정적 지지 시작")

        current_posx = get_current_posx()[0] 
        target_pos = list(td_coord[:3]) + list(current_posx[3:])

        target_pos[1] += self.Y_BACK_OFFSET
        target_pos[2] = max(target_pos[2], MIN_DEPTH)

        movel(target_pos, vel=VELOCITY, acc=ACC)
        mwait()
        # 가만히 벽처럼 지지함


# ==========================================================
# 3. 메인 자세 교정 노드
# ==========================================================
class PostureCorrector(Node):
    def __init__(self):
        super().__init__("posture_corrector_node")
        
        self.init_robot()

        self.strategies = {
            'lateral_raise': LateralRaiseStrategy(),
            'shoulder_press': LateralRaiseStrategy(),
            'bicep_curl': BicepCurlStrategy()
        }
        self.current_exercise = 'lateral_raise' 

        self.correction_sub = self.create_subscription(
            Point, 
            '/right_elbow_3d', 
            self.correction_target_callback, 
            10
        )
        self.package_path = get_package_share_directory("rehab_assist_robot") 
        self.is_moving = False 
        
        self.get_logger().info(f"🤖 노드 시작! 현재 모드: [{self.current_exercise}]")

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1) 

        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)
        base2cam = base2gripper @ gripper2cam
        
        return np.dot(base2cam, coord)[:3]

    def correction_target_callback(self, msg):
        if self.is_moving or msg.z <= 0:
            return

        self.is_moving = True
        try:
            target_cam_pos = [msg.x, msg.y, msg.z]
            gripper2cam_path = os.path.join(self.package_path, "resource", "T_gripper2camera_diff_braket.npy")
            
            robot_posx = get_current_posx()[0]
            td_coord = self.transform_to_base(target_cam_pos, gripper2cam_path, robot_posx)
            
            strategy = self.strategies.get(self.current_exercise)
            if strategy:
                strategy.execute_assist(td_coord)

        except Exception as e:
            self.get_logger().error(f"❌ 교정 동작 중 에러: {e}")
        finally:
            self.is_moving = False

    def init_robot(self):
        JReady = [0, 0, 90, 0, 90, 0]
        movej(JReady, vel=VELOCITY, acc=ACC)

        init_pos = [28.27, 33.84, 127.04, 116.85, -100.21, 76.33]
        movej(init_pos, vel=VELOCITY, acc=ACC)
        gripper.open_gripper()
        mwait()


def main(args=None):
    node = PostureCorrector()
    rclpy.spin(node) 
    
    rclpy.shutdown()
    node.destroy_node()
    dsr_node.destroy_node()

if __name__ == "__main__":
    main()