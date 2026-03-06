import os
import sys
from scipy.spatial.transform import Rotation
import numpy as np
import rclpy
from rclpy.node import Node
import DR_init

from geometry_msgs.msg import Point 
from ament_index_python.packages import get_package_share_directory
from robot_control.onrobot import RG 

package_path = get_package_share_directory("rehab_assist_robot") 

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 100, 100 
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"
DEPTH_OFFSET = 100.0 
Y_OFFSET = 30.0       
MIN_DEPTH = 50.0

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("robot_control_node", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, movel, get_current_posj, get_current_posx, mwait
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

class RobotController(Node):
    def __init__(self):
        super().__init__("wrist_tracking_robot")
        self.init_robot()

        # # 인지 노드에서 '스페이스바'를 누를 때만 들어오는 토픽
        # self.wrist_sub = self.create_subscription(
        #     Point, 
        #     '/wrist_midpoint_3d', 
        #     self.wrist_callback, 
        #     10
        # )

        # 인지 노드에서 '스페이스바'를 누를 때만 들어오는 토픽
        self.wrist_sub = self.create_subscription(
            Point, 
            '/left_wrist_3d', 
            self.wrist_callback, 
            10
        )
        
        
        self.is_moving = False # 중복 이동 방지 플래그
        self.get_logger().info("🚀 로봇 대기 중... (카메라 창에서 스페이스바를 누르면 이동합니다)")

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
        td_coord = np.dot(base2cam, coord)

        return td_coord[:3]

    def wrist_callback(self, msg):
        """스페이스바로 토픽이 발행되었을 때만 호출되어 로봇 이동"""
        # 로봇이 이미 이동 중이라면 새 명령 무시 (안전 장치)
        if self.is_moving:
            self.get_logger().warn("⚠️ 현재 로봇이 이동 중입니다. 명령을 무시합니다.")
            return

        if msg.z <= 0:
            self.get_logger().warn("유효하지 않은 Depth 값입니다.")
            return

        self.is_moving = True
        
        try:
            target_cam_pos = [msg.x, msg.y, msg.z]
            gripper2cam_path = os.path.join(package_path, "resource", "T_gripper2camera_diff_braket.npy")
            
            robot_posx = get_current_posx()[0]
            td_coord = self.transform_to_base(target_cam_pos, gripper2cam_path, robot_posx)
            
            td_coord[2] += DEPTH_OFFSET 
            td_coord[2] = max(td_coord[2], MIN_DEPTH) 
            td_coord[1] += Y_OFFSET     

            target_pos = list(td_coord[:3]) + robot_posx[3:]

            self.get_logger().info(f"📍 로봇 이동 시작: X={target_pos[0]:.1f}, Y={target_pos[1]:.1f}, Z={target_pos[2]:.1f}")
            
            movel(target_pos, vel=VELOCITY, acc=ACC)
            mwait() 
            
            self.get_logger().info("✅ 목표 위치 도착 완료!")
        
        except Exception as e:
            self.get_logger().error(f"이동 중 에러 발생: {e}")
        finally:
            self.is_moving = False # 이동 완료 후 플래그 해제

    def init_robot(self):
        init_pos = get_current_posj()[0]
        print(f'init pos: {init_pos}')
        gripper.open_gripper()
        mwait()

def main(args=None):
    node = RobotController()
    rclpy.spin(node) # 콜백 대기만 수행하는 깔끔한 구조
    rclpy.shutdown()
    node.destroy_node()

if __name__ == "__main__":
    main()