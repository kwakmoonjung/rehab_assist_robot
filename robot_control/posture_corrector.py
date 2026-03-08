import os
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point 
from ament_index_python.packages import get_package_share_directory

# 두산 로봇 및 온로봇 그리퍼 라이브러리 임포트
import DR_init
from robot_control.onrobot import RG 

# --- 로봇 설정 상수 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 100, 100 
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

# --- 위치 튜닝 상수 ---
DEPTH_OFFSET = 40.0
Y_OFFSET = 0.0       
MIN_DEPTH = 50.0
LIFT_OFFSET = 400.0  # 들어올릴 Z축 높이 (자세 교정 폭)

# 두산 로봇 초기화 환경 변수 설정
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

class PostureCorrector(Node):
    def __init__(self):
        super().__init__("posture_corrector_node")
        
        # 1. 두산 로봇 DRL API 초기화 연결
        try:
            # global 선언을 통해 로봇 제어 함수들을 노드 내에서 직접 사용 가능하게 함
            global movej, movel, get_current_posj, get_current_posx, mwait
            from DSR_ROBOT2 import movej, movel, get_current_posj, get_current_posx, mwait
        except ImportError as e:
            self.get_logger().error(f"DSR_ROBOT2 라이브러리를 불러오지 못했습니다: {e}")
            sys.exit()

        # 2. 그리퍼 초기화
        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

        # 3. 로봇 초기 자세 세팅
        self.init_robot()

        # 4. 비전 노드에서 계산한 3D 교정 목표점 구독 (토픽 이름 수정됨)
        self.correction_sub = self.create_subscription(
            Point, 
            '/target_correction_3d', 
            self.correction_target_callback, 
            10
        )
        
        # 패키지 경로 (카메라-그리퍼 캘리브레이션 행렬 로드용)
        self.package_path = get_package_share_directory("rehab_assist_robot") 
        self.is_moving = False # 중복 이동 방지 플래그
        
        self.get_logger().info("🤖 자세 교정(로봇 제어) 노드 시작! 비전 노드의 목표 좌표를 대기합니다.")

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        """로봇의 6자유도 위치(XYZ)와 회전(RxRyRz)을 4x4 변환 행렬로 변환"""
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        """카메라 좌표계의 점을 로봇 베이스 좌표계로 변환"""
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1) 

        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)

        # Base -> Cam 변환 행렬 생성
        base2cam = base2gripper @ gripper2cam
        
        # 목표점의 로봇 베이스 기준 좌표 계산
        td_coord = np.dot(base2cam, coord)
        return td_coord[:3]

    def correction_target_callback(self, msg):
        """비전 노드에서 3D 좌표를 받았을 때 실행되는 콜백 함수"""
        if self.is_moving:
            self.get_logger().warn("⚠️ 현재 로봇이 교정 동작 중입니다. 새 명령을 무시합니다.")
            return

        if msg.z <= 0:
            self.get_logger().warn("⚠️ 유효하지 않은 Depth 값입니다 (0 이하).")
            return

        self.is_moving = True
        
        try:
            target_cam_pos = [msg.x, msg.y, msg.z]
            gripper2cam_path = os.path.join(self.package_path, "resource", "T_gripper2camera_diff_braket.npy")
            
            # 1. 로봇의 현재 좌표(Task Space)를 읽어와 카메라 기준 좌표를 로봇 베이스 기준으로 변환
            robot_posx = get_current_posx()[0]
            td_coord = self.transform_to_base(target_cam_pos, gripper2cam_path, robot_posx)
            
            # 오프셋 적용 (위험 방지용 최소 Depth 설정 포함)
            td_coord[2] += DEPTH_OFFSET 
            td_coord[2] = max(td_coord[2], MIN_DEPTH) 
            td_coord[1] += Y_OFFSET     

            # 2. 이동 직전, 제자리에서 6축(마지막 조인트)만 90도 회전
            self.get_logger().info("🔄 movel 전 6축 90도 회전 시작")
            current_posj = get_current_posj()
            rotated_posj = list(current_posj)
            rotated_posj[5] += 90.0  # 6축 조인트 각도 90도 증가
            movej(rotated_posj, vel=VELOCITY, acc=ACC)
            mwait()
            self.get_logger().info("✅ 6축 회전 완료")

            # 3. 회전이 완료된 후 변경된 툴의 자세(Rx, Ry, Rz)를 다시 읽어옴
            rotated_posx = get_current_posx()[0] 
            
            # 4. 목표 위치(X,Y,Z)에 새로 읽어온 회전 후의 자세(Rx, Ry, Rz)를 결합
            target_pos = list(td_coord[:3]) + list(rotated_posx[3:])

            self.get_logger().info(f"📍 자세 교정 목표점(Midpoint)으로 로봇 이동: X={target_pos[0]:.1f}, Y={target_pos[1]:.1f}, Z={target_pos[2]:.1f}")
            movel(target_pos, vel=VELOCITY, acc=ACC)
            mwait() 
            self.get_logger().info("✅ 목표 위치(중간점) 도착 완료!")

            # 5. 그리퍼 파지 (환자의 보조 기구나 팔을 잡음)
            self.get_logger().info("✊ 자세 교정용 파지 (그리퍼 닫기)")
            self.gripper.close_gripper(force_val=100)
            time.sleep(4.0)
            
            # 6. 견인 동작 (Z축으로 들어올리기)
            target_pos_up = list(target_pos)
            target_pos_up[2] += LIFT_OFFSET
            self.get_logger().info(f"⬆️ Z축으로 {LIFT_OFFSET}mm 들어올려 견인 시작")
            movel(target_pos_up, vel=VELOCITY, acc=ACC)
            mwait()
            self.get_logger().info("✅ 환자 자세 교정(견인) 완료!")
        
        except Exception as e:
            self.get_logger().error(f"❌ 교정 동작 중 에러 발생: {e}")
        finally:
            self.is_moving = False

    def init_robot(self):
        """시작 시 로봇을 안전한 기본 위치로 이동 및 그리퍼 개방"""
        JReady = [0, 0, 90, 0, 90, 0]
        movej(JReady, vel=VELOCITY, acc=ACC)

        init_pos = [73.03, 71.08, -33.55, 23.68, -123.47, -74.72]
        movej(init_pos, vel=VELOCITY, acc=ACC)
        self.get_logger().info(f'초기 조인트 위치 세팅 완료: {init_pos}')
        
        self.gripper.open_gripper()
        mwait()


def main(args=None):
    # 두산 로봇 초기화 환경 변수는 main 실행 전에 세팅되어야 함
    rclpy.init(args=args)
    
    # 두산 로봇 노드 생성 후 글로벌 연결
    dsr_node = rclpy.create_node("dsr_robot_core_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node
    
    # 자세 교정기 노드 생성 및 스핀
    node = PostureCorrector()
    rclpy.spin(node) 
    
    rclpy.shutdown()
    node.destroy_node()
    dsr_node.destroy_node()

if __name__ == "__main__":
    main()