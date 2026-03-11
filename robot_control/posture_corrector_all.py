import os
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point 
from std_msgs.msg import String
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

# [추가] 지지 상태를 관리하는 전역 변수
g_is_supporting = False

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
    from DSR_ROBOT2 import task_compliance_ctrl, release_compliance_ctrl
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
        # 팔꿈치 3D 좌표 기준 전완근 쪽으로 이동하기 위한 오프셋 (단위: mm)
        self.X_OFFSET = -50.0           # X축으로 -5cm (-50mm) 이동 (전완근 방향)
        self.Y_INWARD_OFFSET = -150.0   # Y축으로 -15cm (-150mm) 더 들어가기 (몸쪽 밀착)
        self.Z_APPROACH_OFFSET = -100.0 # Z축으로 -10cm (-100mm) 아래로 접근 (안전한 밑바닥 지지)
        self.LIFT_OFFSET = 200.0        # 지지 위치에서 위로 20cm(200mm) 들어올림

    def execute_assist(self, td_coord):
        print("💪 [사레레/숄더프레스] 전완근 타겟 동적 보조 시작")

        current_posx = get_current_posx()[0] 
        target_pos = list(td_coord[:3]) + list(current_posx[3:])

        # XY축 오프셋 적용 (전완근 위치로 타겟 변경)
        target_pos[0] += self.X_OFFSET
        target_pos[1] += self.Y_INWARD_OFFSET

        # 1. 전완근 밑으로 접근 (대기/지지 위치)
        approach_pos = list(target_pos)
        approach_pos[2] = max(approach_pos[2] + self.Z_APPROACH_OFFSET, MIN_DEPTH)

        print(f"📍 지지 위치로 접근: X={approach_pos[0]:.1f}, Y={approach_pos[1]:.1f}, Z={approach_pos[2]:.1f}")
        movel(approach_pos, vel=VELOCITY, acc=ACC)
        mwait()

        # 2. 대기 위치에서 정확히 LIFT_OFFSET 만큼만 위로 들어올리기
        lift_pos = list(approach_pos)
        lift_pos[2] += self.LIFT_OFFSET
        
        print(f"⬆️ 위로 들어올리기: Z={lift_pos[2]:.1f}")
        movel(lift_pos, vel=VELOCITY, acc=ACC)
        mwait()

class ShoulderPressStrategy(ExerciseStrategy):
    def __init__(self):
        # 팔꿈치 바로 밑에서 받쳐주기 위한 오프셋 (단위: mm)
        self.X_OFFSET = 0.0             # X축 이동 없음 (팔꿈치 바로 아래)
        self.Y_INWARD_OFFSET = -100.0    # Y축으로 -5cm (-50mm) 밀착 (안전 거리 유지)
        self.Z_APPROACH_OFFSET = -50.0 # Z축으로 -10cm (-100mm) 아래로 접근
        self.LIFT_OFFSET = 200.0        # 지지 위치에서 위로 20cm(200mm) 들어올림

    def execute_assist(self, td_coord):
        print("💪 [숄더프레스] 팔꿈치 하단 동적 보조 시작")

        current_posx = get_current_posx()[0] 
        target_pos = list(td_coord[:3]) + list(current_posx[3:])

        target_pos[0] += self.X_OFFSET
        target_pos[1] += self.Y_INWARD_OFFSET

        # 1. 팔꿈치 밑으로 10cm 아래 접근
        approach_pos = list(target_pos)
        approach_pos[2] = max(approach_pos[2] + self.Z_APPROACH_OFFSET, MIN_DEPTH)

        print(f"📍 팔꿈치 하단 위치로 접근: X={approach_pos[0]:.1f}, Y={approach_pos[1]:.1f}, Z={approach_pos[2]:.1f}")
        movel(approach_pos, vel=VELOCITY, acc=ACC)
        mwait()

        # 2. 대기 위치에서 20cm 위로 밀어 올리기
        lift_pos = list(approach_pos)
        lift_pos[2] += self.LIFT_OFFSET
        
        print(f"⬆️ 위로 밀어 올리기: Z={lift_pos[2]:.1f}")
        movel(lift_pos, vel=VELOCITY, acc=ACC)
        mwait()

class BicepCurlStrategy(ExerciseStrategy):
    def __init__(self):
        self.X_OFFSET = 30.0
        self.Z_APPROACH_OFFSET = 10.0  
        self.Y_APPROACH_OFFSET = -50.0   
        self.Y_SUPPORT_OFFSET = -50.0    

    def execute_assist(self, td_coord):
        global g_is_supporting

        print("[이두컬] 정적 지지 시작")

        current_posx = get_current_posx()[0] 
        base_pos = list(td_coord[:3]) + list(current_posx[3:])

        approach_pos = list(base_pos)
        approach_pos[0] += self.X_OFFSET
        approach_pos[1] += self.Y_APPROACH_OFFSET
        approach_pos[2] = max(approach_pos[2] + self.Z_APPROACH_OFFSET, MIN_DEPTH)

        print(f"1차 접근 위치로 이동: X={approach_pos[0]:.1f}, Y={approach_pos[1]:.1f}, Z={approach_pos[2]:.1f}")
        movel(approach_pos, vel=VELOCITY, acc=ACC)
        mwait()

        support_pos = list(approach_pos)
        support_pos[1] += self.Y_SUPPORT_OFFSET
        
        print(f"최종 지지 위치로 밀착: X={support_pos[0]:.1f}, Y={support_pos[1]:.1f}, Z={support_pos[2]:.1f}")
        movel(support_pos, vel=VELOCITY, acc=ACC)
        mwait()

        task_compliance_ctrl(stx=[10000, 3000, 1000, 100, 100, 100], time=0.5)

        # [추가] 이동 완료 후 상태 변수 활성화 및 함수 정상 종료
        g_is_supporting = True
        print("[이두컬] 지지 위치 도달. 종료 신호 대기 중...")


# ==========================================================
# 3. 메인 자세 교정 노드
# ==========================================================
class PostureCorrector(Node):
    def __init__(self):
        super().__init__("posture_corrector_node")
        
        self.is_moving = False 
        
        self.init_robot()

        self.strategies = {
            'bicep_curl': BicepCurlStrategy()
        }
        self.current_exercise = 'bicep_curl'

        self.mode_sub = self.create_subscription(
            String, '/set_exercise_mode', self.mode_callback, 10
        )

        self.correction_sub = self.create_subscription(
            Point, '/right_elbow_3d', self.correction_target_callback, 10
        )
        
        self.sys_cmd_sub = self.create_subscription(
            String, '/system_command', self.sys_cmd_callback, 10
        )

        self.package_path = get_package_share_directory("rehab_assist_robot") 
        self.get_logger().info(f"노드 시작! 현재 로봇 모드: [{self.current_exercise}]")

    def sys_cmd_callback(self, msg):
        global g_is_supporting
        
        if msg.data == "START_EXERCISE":
            self.get_logger().info("새로운 운동 시작 명령 감지. 카메라 뷰 확보를 위해 로봇을 초기 위치로 복귀시킵니다.")
            self.move_to_init_pos()
        elif msg.data == "END_EXERCISE":
            # [추가] 지지 상태일 때만 해제 수행
            if g_is_supporting:
                self.get_logger().info("종료 신호 수신. 정적 지지를 해제합니다.")
                release_compliance_ctrl()
                g_is_supporting = False

    def move_to_init_pos(self):
        if self.is_moving:
            self.get_logger().warn("로봇이 현재 교정 동작 중입니다. 복귀 명령을 무시하거나 대기합니다.")
            return
            
        self.is_moving = True
        try:
            init_pos = [48.07, 29.12, 113.41, 131.73, -117.85, 62.66]
            self.get_logger().info("초기 위치(카메라 촬영 뷰)로 로봇 이동 시작...")
            movej(init_pos, vel=VELOCITY, acc=ACC)
            mwait()
            self.get_logger().info("초기 위치 도착 완료! 측면 뷰가 성공적으로 확보되었습니다.")
        except Exception as e:
            self.get_logger().error(f"초기 위치 이동 중 에러 발생: {e}")
        finally:
            self.is_moving = False

    def mode_callback(self, msg):
        new_mode = msg.data.lower()
        if new_mode in self.strategies:
            self.current_exercise = new_mode
            self.get_logger().info(f"로봇 보조 모드가 [{new_mode}](으)로 변경되었습니다.")
        else:
            self.get_logger().warn(f"지원하지 않는 로봇 운동 모드입니다: {new_mode}")

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
        global g_is_supporting
        # [추가] 지지 중(g_is_supporting)일 때 새로운 3D 좌표 입력을 무시하도록 조건 추가
        if self.is_moving or g_is_supporting or msg.z <= 0:
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
            self.get_logger().error(f"교정 동작 중 에러: {e}")
        finally:
            self.is_moving = False

    def init_robot(self):
        JReady = [0, 0, 90, 0, 90, 0]
        movej(JReady, vel=VELOCITY, acc=ACC)
        
        self.move_to_init_pos()
        
        gripper.close_gripper()
        mwait()

def main(args=None):
    node = PostureCorrector()
    rclpy.spin(node) 
    
    rclpy.shutdown()
    node.destroy_node()
    dsr_node.destroy_node()

if __name__ == "__main__":
    main()