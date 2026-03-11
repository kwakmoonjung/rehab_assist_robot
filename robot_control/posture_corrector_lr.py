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
    # [추가] 원호 보간 이동을 위한 movec 함수 import
    from DSR_ROBOT2 import movej, movel, movec, get_current_posj, get_current_posx, mwait
    from DSR_ROBOT2 import task_compliance_ctrl, release_compliance_ctrl
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)


# ==========================================================
# 2. 운동별 로봇 보조 전략 (Strategy Pattern) 
# ==========================================================
class ExerciseStrategy:
    # [추가] 어깨 좌표 매개변수 추가
    def execute_assist(self, td_coord, td_coord_shoulder=None):
        pass

class LateralRaiseStrategy(ExerciseStrategy):
    def __init__(self):
        self.X_OFFSET = -50.0           
        self.Y_INWARD_OFFSET = -150.0   
        self.Z_APPROACH_OFFSET = -100.0 
        # LIFT_OFFSET은 동적 각도 계산으로 대체되므로 사용하지 않습니다.

    # [수정] 어깨 좌표 매개변수 추가 및 movec 원호 궤적 로직 구현
    def execute_assist(self, td_coord, td_coord_shoulder=None):
        print("[사레레] 전완근 타겟 동적 보조 시작 (원호 궤적)")

        current_posx = get_current_posx()[0] 
        target_pos = list(td_coord[:3]) + list(current_posx[3:])

        target_pos[0] += self.X_OFFSET
        target_pos[1] += self.Y_INWARD_OFFSET

        approach_pos = list(target_pos)
        approach_pos[2] = max(approach_pos[2] + self.Z_APPROACH_OFFSET, MIN_DEPTH)

        print(f"지지 위치로 접근: X={approach_pos[0]:.1f}, Y={approach_pos[1]:.1f}, Z={approach_pos[2]:.1f}")
        movel(approach_pos, vel=VELOCITY, acc=ACC)
        mwait()

        # [추가] 어깨 기준 원호 궤적 연산
        if td_coord_shoulder is None:
            print("어깨 좌표가 없습니다. 교정을 중단합니다.")
            return

        S = np.array(td_coord_shoulder[:3])
        E_start = np.array(approach_pos[:3])
        V = E_start - S 
        
        # [추가] 몸통 벡터(Z축 아래 방향)를 기준으로 현재 팔 각도 계산
        Z_down = np.array([0.0, 0.0, -1.0])
        v_norm = np.linalg.norm(V)
        
        if v_norm < 1e-6:
            print("벡터 길이가 유효하지 않습니다.")
            return
            
        cos_phi = np.clip(np.dot(V, Z_down) / v_norm, -1.0, 1.0)
        current_angle_deg = np.degrees(np.arccos(cos_phi))
        
        # [추가] 몸통-팔 각도가 90도가 되기 위한 필요 회전량 산출
        target_angle_deg = 90.0
        theta_deg = target_angle_deg - current_angle_deg
        
        if theta_deg <= 0:
            print(f"현재 팔 각도({current_angle_deg:.1f}도)가 이미 목표 각도 이상입니다. 들어올리지 않습니다.")
            return

        # Z축(위) 방향으로 회전하기 위한 회전축 계산
        Z_up = np.array([0.0, 0.0, 1.0])
        axis = np.cross(V, Z_up)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / axis_norm

        theta = np.radians(theta_deg)
        
        # 경유점 (pos1)
        R_mid = Rotation.from_rotvec(axis * (theta / 2.0))
        V_mid = R_mid.apply(V)
        pos1 = list(S + V_mid) + list(current_posx[3:])

        # 목표점 (pos2)
        R_end = Rotation.from_rotvec(axis * theta)
        V_end = R_end.apply(V)
        pos2 = list(S + V_end) + list(current_posx[3:])

        ############ 검증 #############
        angle_mid_check = np.degrees(np.arccos(np.clip(np.dot(V, V_mid) / (np.linalg.norm(V) * np.linalg.norm(V_mid)), -1.0, 1.0)))
        angle_end_check = np.degrees(np.arccos(np.clip(np.dot(V, V_end) / (np.linalg.norm(V) * np.linalg.norm(V_end)), -1.0, 1.0)))
        
        print(f"[검증 로그] 현재 팔 각도: {current_angle_deg:.1f}도 -> 목표: {target_angle_deg:.1f}도")
        print(f"[검증 로그] 시작점 -> 경유점 회전 각도: {angle_mid_check:.1f}도 (목표: {theta_deg/2:.1f}도)")
        print(f"[검증 로그] 시작점 -> 목표점 회전 각도: {angle_end_check:.1f}도 (목표: {theta_deg:.1f}도)")
        print(f"[좌표 확인] 시작 Z: {E_start[2]:.1f} -> 경유 Z: {pos1[2]:.1f} -> 목표 Z: {pos2[2]:.1f}")
        ###############################

        # task_compliance_ctrl(stx=[3000, 3000, 1500, 100, 100, 100], time=0.5)

        print(f"위로 사레레 원호 밀어 올리기 (목표 Z={pos2[2]:.1f})")
        movec(pos1, pos2, vel=VELOCITY, acc=ACC)
        mwait()

        ############ 물리적 검증 #############
        actual_end_pos = get_current_posx()[0]
        E_actual = np.array(actual_end_pos[:3])
        V_actual = E_actual - S
        
        start_radius = np.linalg.norm(V)
        actual_radius = np.linalg.norm(V_actual)
        
        if start_radius > 0 and actual_radius > 0:
            actual_angle = np.degrees(np.arccos(np.clip(np.dot(V, V_actual) / (start_radius * actual_radius), -1.0, 1.0)))
        else:
            actual_angle = 0.0
        
        print(f"[물리적 궤적 검증] 시작 반지름(어깨-팔지점 거리): {start_radius:.1f}mm")
        print(f"[물리적 궤적 검증] 종료 반지름(어깨-로봇종단 거리): {actual_radius:.1f}mm")
        print(f"[물리적 궤적 검증] 실제 회전 각도: {actual_angle:.1f}도")
        ###############################

class ShoulderPressStrategy(ExerciseStrategy):
    def __init__(self):
        self.X_OFFSET = 0.0             
        self.Y_INWARD_OFFSET = -100.0    
        self.Z_APPROACH_OFFSET = -50.0 
        # [추가] 숄더 프레스 원호 목표 각도 설정 (60도 상승)
        self.TARGET_ANGLE_DEG = 100.0

    # [추가] 어깨 좌표 매개변수 적용 및 movec 원호 로직 구현
    def execute_assist(self, td_coord, td_coord_shoulder=None):
        print("[숄더프레스] 팔꿈치 하단 동적 보조 시작 (원호 궤적)")

        current_posx = get_current_posx()[0] 
        target_pos = list(td_coord[:3]) + list(current_posx[3:])

        target_pos[0] += self.X_OFFSET
        target_pos[1] += self.Y_INWARD_OFFSET

        # 1. 팔꿈치 밑으로 10cm 아래 접근
        approach_pos = list(target_pos)
        approach_pos[2] = max(approach_pos[2] + self.Z_APPROACH_OFFSET, MIN_DEPTH)

        print(f"팔꿈치 하단 위치로 접근: X={approach_pos[0]:.1f}, Y={approach_pos[1]:.1f}, Z={approach_pos[2]:.1f}")
        movel(approach_pos, vel=VELOCITY, acc=ACC)
        mwait()

        # [추가] 2. 어깨 기준 원호 궤적(pos1, pos2) 연산
        if td_coord_shoulder is None:
            print("어깨 좌표가 없습니다. 교정을 중단합니다.")
            return

        S = np.array(td_coord_shoulder[:3])
        E_start = np.array(approach_pos[:3])
        V = E_start - S # 어깨에서 팔꿈치를 향하는 벡터 (반지름)
        
        # Z축(위) 방향으로 회전하기 위한 회전축 계산 (V와 Z축의 외적)
        Z_up = np.array([0.0, 0.0, 1.0])
        axis = np.cross(V, Z_up)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / axis_norm

        theta = np.radians(self.TARGET_ANGLE_DEG)
        
        # 경유점 (pos1) - 30도 회전
        R_mid = Rotation.from_rotvec(axis * (theta / 2.0))
        V_mid = R_mid.apply(V)
        pos1 = list(S + V_mid) + list(current_posx[3:])

        # 목표점 (pos2) - 60도 회전
        R_end = Rotation.from_rotvec(axis * theta)
        V_end = R_end.apply(V)
        pos2 = list(S + V_end) + list(current_posx[3:])

        # [추가] 3. 컴플라이언스 활성화 및 원호 이동(movec)
        # task_compliance_ctrl(stx=[3000, 3000, 1500, 100, 100, 100], time=0.5)

        ############ 검증 #############
        # [추가] 궤적 생성 검증: 원래 벡터(V)와 생성된 벡터 간의 실제 사이각 역산
        angle_mid_check = np.degrees(np.arccos(np.clip(np.dot(V, V_mid) / (np.linalg.norm(V) * np.linalg.norm(V_mid)), -1.0, 1.0)))
        angle_end_check = np.degrees(np.arccos(np.clip(np.dot(V, V_end) / (np.linalg.norm(V) * np.linalg.norm(V_end)), -1.0, 1.0)))
        
        print(f"[검증 로그] 시작점 -> 경유점 회전 각도: {angle_mid_check:.1f}도 (목표: 30.0도)")
        print(f"[검증 로그] 시작점 -> 목표점 회전 각도: {angle_end_check:.1f}도 (목표: {self.TARGET_ANGLE_DEG}도)")
        print(f"[좌표 확인] 시작 Z: {E_start[2]:.1f} -> 경유 Z: {pos1[2]:.1f} -> 목표 Z: {pos2[2]:.1f}")
        ###############################

        # [추가] 3. 컴플라이언스 활성화 및 원호 이동(movec)
        # task_compliance_ctrl(stx=[3000, 3000, 1500, 100, 100, 100], time=0.5)

        print(f"위로 원호 밀어 올리기 (목표 Z={pos2[2]:.1f})")
        movec(pos1, pos2, vel=VELOCITY, acc=ACC)
        mwait()

        ############ 검증 #############
        # [추가] 실제 구동 궤적 검증 로직
        actual_end_pos = get_current_posx()[0]
        E_actual = np.array(actual_end_pos[:3])
        V_actual = E_actual - S
        
        start_radius = np.linalg.norm(V)
        actual_radius = np.linalg.norm(V_actual)
        
        # 0으로 나누기 방지 및 내적을 이용한 실제 회전 각도 계산
        if start_radius > 0 and actual_radius > 0:
            actual_angle = np.degrees(np.arccos(np.clip(np.dot(V, V_actual) / (start_radius * actual_radius), -1.0, 1.0)))
        else:
            actual_angle = 0.0
        
        print(f"[물리적 궤적 검증] 시작 반지름(어깨-팔꿈치 거리): {start_radius:.1f}mm")
        print(f"[물리적 궤적 검증] 종료 반지름(어깨-로봇종단 거리): {actual_radius:.1f}mm")
        print(f"[물리적 궤적 검증] 실제 회전 각도: {actual_angle:.1f}도")
        ###############################
        # release_compliance_ctrl()

class BicepCurlStrategy(ExerciseStrategy):
    def __init__(self):
        self.X_OFFSET = 30.0
        self.Z_APPROACH_OFFSET = 10.0  
        self.Y_APPROACH_OFFSET = -50.0   
        self.Y_SUPPORT_OFFSET = -50.0    

    # [추가] 어깨 좌표 매개변수 추가
    def execute_assist(self, td_coord, td_coord_shoulder=None):
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
            'bicep_curl': BicepCurlStrategy(),
            'shoulder_press': ShoulderPressStrategy(),
            'lateral_raise': LateralRaiseStrategy()
        }
        self.current_exercise = 'bicep_curl'

        self.mode_sub = self.create_subscription(
            String, '/set_exercise_mode', self.mode_callback, 10
        )

        # [수정] 팔꿈치 좌표 저장 변수 추가
        self.latest_elbow_cam_pos = None
        self.correction_sub = self.create_subscription(
            Point, '/right_elbow_3d', self.correction_target_callback, 10
        )
        
        self.latest_shoulder_cam_pos = None
        self.shoulder_sub = self.create_subscription(
            Point, '/right_shoulder_3d', self.shoulder_target_callback, 10
        )

        self.sys_cmd_sub = self.create_subscription(
            String, '/system_command', self.sys_cmd_callback, 10
        )

        self.package_path = get_package_share_directory("rehab_assist_robot") 
        self.get_logger().info(f"노드 시작! 현재 로봇 모드: [{self.current_exercise}]")

    # [수정] 좌표만 저장하고 실행 검사 함수 호출
    def shoulder_target_callback(self, msg):
        if msg.z > 0:
            self.latest_shoulder_cam_pos = [msg.x, msg.y, msg.z]
            self.try_execute_assist()

    # [수정] 좌표만 저장하고 실행 검사 함수 호출
    def correction_target_callback(self, msg):
        if msg.z > 0:
            self.latest_elbow_cam_pos = [msg.x, msg.y, msg.z]
            self.try_execute_assist()

    # [추가] 두 좌표가 모두 수신되었는지 확인 후 이동 로직 실행
    def try_execute_assist(self):
        global g_is_supporting
        if self.is_moving or g_is_supporting:
            return

        # 둘 중 하나라도 들어오지 않았다면 대기
        if self.latest_elbow_cam_pos is None or self.latest_shoulder_cam_pos is None:
            return

        self.is_moving = True
        try:
            gripper2cam_path = os.path.join(self.package_path, "resource", "T_gripper2camera_diff_braket.npy")
            robot_posx = get_current_posx()[0]
            
            td_coord_elbow = self.transform_to_base(self.latest_elbow_cam_pos, gripper2cam_path, robot_posx)
            td_coord_shoulder = self.transform_to_base(self.latest_shoulder_cam_pos, gripper2cam_path, robot_posx)
            
            strategy = self.strategies.get(self.current_exercise)
            if strategy:
                strategy.execute_assist(td_coord_elbow, td_coord_shoulder)

        except Exception as e:
            self.get_logger().error(f"교정 동작 중 에러: {e}")
        finally:
            self.is_moving = False
            # [추가] 실행 완료 후 다음 교정을 위해 저장된 좌표 초기화
            self.latest_elbow_cam_pos = None
            self.latest_shoulder_cam_pos = None

    def sys_cmd_callback(self, msg):
        global g_is_supporting
        
        if msg.data == "START_EXERCISE":
            self.get_logger().info("새로운 운동 시작 명령 감지. 카메라 뷰 확보를 위해 로봇을 초기 위치로 복귀시킵니다.")
            self.move_to_init_pos()
        elif msg.data == "END_EXERCISE":
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