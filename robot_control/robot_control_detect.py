import os
import sys
from scipy.spatial.transform import Rotation
import numpy as np
import rclpy
from rclpy.node import Node
import DR_init
import time

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
DEPTH_OFFSET = 40 # 50.0 #100.0 
Y_OFFSET = 0.0 #30.0       
MIN_DEPTH = 50.0
LIFT_OFFSET = 400.0 #50.0  # [추가] 들어올릴 Z축 높이 (50mm)

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

        # [추가 및 수정] 양 손목의 중간점(midpoint)을 받아오는 Subscriber 활성화
        self.midpoint_sub = self.create_subscription(
            Point, 
            '/wrist_midpoint_3d', 
            self.midpoint_callback, 
            10
        )

        # [기존 코드 주석 처리] 원래 왼쪽 손목을 따라가게 하던 Subscriber
        # self.wrist_sub = self.create_subscription(
        #     Point, 
        #     '/left_wrist_3d', 
        #     self.wrist_callback, 
        #     10
        # )
        
        self.is_moving = False # 중복 이동 방지 플래그
        self.get_logger().info("🚀 로봇 대기 중... (카메라 창에서 스페이스바를 누르면 중간점으로 이동합니다)")

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

    def midpoint_callback(self, msg):
        """[추가] 양 손목의 중간점(midpoint)으로 로봇을 이동시키는 콜백 함수"""
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

            # 2. movel 이동 직전, 제자리에서 6축(마지막 조인트)만 90도 회전
            self.get_logger().info("🔄 movel 전 6축 90도 회전 시작")
            current_posj = get_current_posj()
            rotated_posj = list(current_posj)
            rotated_posj[5] += 90.0  # 6축 조인트 각도 90도 증가 (반대 회전이 필요하면 -90.0)
            movej(rotated_posj, vel=VELOCITY, acc=ACC)
            mwait()
            self.get_logger().info("✅ 6축 회전 완료")

            # --- [수정된 부분 시작] ---
            # 회전이 완료된 후 변경된 툴의 자세(Rx, Ry, Rz)를 다시 읽어옵니다.
            rotated_posx = get_current_posx()[0] 
            
            # 계산해둔 목표 위치(X,Y,Z)에 새로 읽어온 회전 후의 자세(Rx, Ry, Rz)를 결합합니다.
            target_pos = list(td_coord[:3]) + list(rotated_posx[3:])
            # --- [수정된 부분 끝] ---

            self.get_logger().info(f"📍 중간점(Midpoint)으로 로봇 이동 시작: X={target_pos[0]:.1f}, Y={target_pos[1]:.1f}, Z={target_pos[2]:.1f}")
            movel(target_pos, vel=VELOCITY, acc=ACC)
            mwait() 
            self.get_logger().info("✅ 목표 위치(중간점) 도착 완료!")

            # (필요시 아래 주석을 해제하여 잡기 및 들어올리기 동작 수행)
            self.get_logger().info("✊ 중간 파지 지점 잡기 (그리퍼 닫기)")
            gripper.close_gripper(force_val=100)
            time.sleep(4.0)
            
            target_pos_up = list(target_pos)
            target_pos_up[2] += LIFT_OFFSET
            self.get_logger().info(f"⬆️ Z축으로 {LIFT_OFFSET}mm 들어올리기 시작")
            movel(target_pos_up, vel=VELOCITY, acc=ACC)
            mwait()
            self.get_logger().info("✅ 자세 교정(Z축 이동) 완료!")
        
        except Exception as e:
            self.get_logger().error(f"중간점 이동 중 에러 발생: {e}")
        finally:
            self.is_moving = False

    # [기존 코드 주석 처리] 원래 개별 손목 위치로 이동하던 콜백 함수
    # def wrist_callback(self, msg):
    #     if self.is_moving:
    #         self.get_logger().warn("⚠️ 현재 로봇이 이동 중입니다. 명령을 무시합니다.")
    #         return
    #     if msg.z <= 0:
    #         self.get_logger().warn("유효하지 않은 Depth 값입니다.")
    #         return
    #     self.is_moving = True
    #     try:
    #         target_cam_pos = [msg.x, msg.y, msg.z]
    #         gripper2cam_path = os.path.join(package_path, "resource", "T_gripper2camera_diff_braket.npy")
    #         robot_posx = get_current_posx()[0]
    #         td_coord = self.transform_to_base(target_cam_pos, gripper2cam_path, robot_posx)
    #         td_coord[2] += DEPTH_OFFSET 
    #         td_coord[2] = max(td_coord[2], MIN_DEPTH) 
    #         td_coord[1] += Y_OFFSET     
    #         target_pos = list(td_coord[:3]) + robot_posx[3:]
    #         self.get_logger().info(f"📍 로봇 이동 시작: X={target_pos[0]:.1f}, Y={target_pos[1]:.1f}, Z={target_pos[2]:.1f}")
    #         movel(target_pos, vel=VELOCITY, acc=ACC)
    #         mwait() 
    #         self.get_logger().info("✅ 목표 위치 도착 완료!")
    #     except Exception as e:
    #         self.get_logger().error(f"이동 중 에러 발생: {e}")
    #     finally:
    #         self.is_moving = False

    def init_robot(self):
        JReady = [0, 0, 90, 0, 90, 0]
        movej(JReady, vel=VELOCITY, acc=ACC)

        init_pos = [73.03, 71.08, -33.55, 23.68, -123.47, -74.72]
        movej(init_pos, vel=VELOCITY, acc=ACC)
        print(f'init pos: {init_pos}')
        gripper.open_gripper()
        mwait()

def main(args=None):
    node = RobotController()
    rclpy.spin(node) 
    rclpy.shutdown()
    node.destroy_node()

if __name__ == "__main__":
    main()