import os
import time
import sys
from scipy.spatial.transform import Rotation
import numpy as np
import rclpy
from rclpy.node import Node
import DR_init

from geometry_msgs.msg import Point # 3D 중심점 구독용
from ament_index_python.packages import get_package_share_directory
from robot_control.onrobot import RG # 패키지명에 맞게 유지

package_path = get_package_share_directory("rehab_assist_robot") # 실제 구동 패키지명으로 변경 필요 시 수정

# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 100, 100 # 추종을 위해 속도/가속도를 조금 높임 (테스트하며 조절)
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"
DEPTH_OFFSET = 100.0 # 손목에 부딪히지 않도록 Z축 위로 100mm 띄움
Y_OFFSET = 30.0       # [추가] Y축 방향 오프셋 (안전 거리, 예: 30mm)
MIN_DEPTH = 50.0

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("robot_control_node", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, movel, get_current_posj, get_current_posx, mwait, trans
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

########### Gripper Setup ############
gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

########### Robot Controller ############
class RobotController(Node):
    def __init__(self):
        super().__init__("wrist_tracking_robot")
        self.init_robot()

        # [추가] YOLO 노드에서 발행하는 손목 중심점 3D 좌표 구독
        self.wrist_sub = self.create_subscription(
            Point, 
            '/wrist_midpoint_3d', 
            self.wrist_callback, 
            10
        )
        
        self.latest_cam_pos = None # 최신 카메라 3D 좌표 저장용
        self.get_logger().info("🚀 손목 중심점 트래킹 대기 중...")

    def wrist_callback(self, msg):
        """비전 노드에서 중심점이 날아오면 최신 데이터 업데이트"""
        # 카메라 Z축 값이 0보다 클 때만 유효한 값으로 판단
        if msg.z > 0:
            self.latest_cam_pos = [msg.x, msg.y, msg.z]

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        """카메라 좌표계를 로봇 베이스 좌표계로 변환"""
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1)  # Homogeneous coordinate

        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)

        # 좌표 변환 (카메라 -> 그리퍼 -> 베이스)
        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)

        return td_coord[:3]

    def robot_control(self):
        """메인 제어 루프: 새로운 목표점이 있으면 그곳으로 이동"""
        if self.latest_cam_pos is None:
            return # 데이터가 아직 안 들어왔으면 리턴

        # 최신 데이터를 로컬 변수에 복사하고 None으로 초기화 (중복 이동 방지)
        target_cam_pos = self.latest_cam_pos.copy()
        self.latest_cam_pos = None 

        gripper2cam_path = os.path.join(package_path, "resource", "T_gripper2camera_diff_braket.npy")
        
        # 1. 로봇의 현재 위치 획득
        robot_posx = get_current_posx()[0]
        
        # 2. 좌표 변환 수행 (카메라 3D -> 로봇 Base 3D)
        td_coord = self.transform_to_base(target_cam_pos, gripper2cam_path, robot_posx)
        
        # 3. 안전 오프셋 적용
        td_coord[2] += DEPTH_OFFSET # 손목에 로봇이 닿지 않게 일정 거리 유지
        td_coord[2] = max(td_coord[2], MIN_DEPTH) # 로봇이 바닥을 치지 않게 제한
        td_coord[1] += Y_OFFSET     # [추가] Y축 보정 (봉에서 살짝 떨어지게 함)

        # 4. 목표 위치 리스트 생성 (X, Y, Z는 새로운 좌표, Rx, Ry, Rz는 기존 자세 유지)
        target_pos = list(td_coord[:3]) + robot_posx[3:]

        self.get_logger().info(f"이동 목표 (Base): X={target_pos[0]:.1f}, Y={target_pos[1]:.1f}, Z={target_pos[2]:.1f}")
        
        # 5. 로봇 이동
        try:
            movel(target_pos, vel=VELOCITY, acc=ACC)
            # mwait()를 제거하거나 짧게 대기하게 해야 연속적인 트래킹(추종)이 부드럽게 이어집니다.
            # 하지만 안전을 위해 처음에는 넣고 테스트해 보세요.
            mwait() 
        except Exception as e:
            self.get_logger().error(f"이동 중 에러 발생: {e}")

    def init_robot(self):
        JReady = [0, 0, 90, 0, 90, 0]
        #movej(JReady, vel=VELOCITY, acc=ACC)
        init_pos = get_current_posj()[0]
        print(f'init pos: {init_pos}')
        gripper.open_gripper()
        mwait()

def main(args=None):
    node = RobotController()
    
    # 주기적으로 콜백(Subscriber)을 처리하고 제어 루프를 실행하도록 변경
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1) # 토픽이 오는지 0.1초 동안 확인
        node.robot_control()                   # 목표점이 갱신되었다면 이동
        
    rclpy.shutdown()
    node.destroy_node()

if __name__ == "__main__":
    main()