import os
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
import threading 

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Point 
from std_msgs.msg import String
from std_msgs.msg import Bool 
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
g_is_recovery_triggered = False 
g_emergency_stop = False 

# ==========================================================
# 1. 두산 API 전역(Global) 세팅 
# ==========================================================
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("dsr_robot_core_node", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, get_current_posj, get_current_posx
    from DSR_ROBOT2 import task_compliance_ctrl, release_compliance_ctrl
    from DSR_ROBOT2 import amovel, amovec, amovej, check_motion
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

# ==========================================================
# 통신 충돌(Generator Error)을 방지한 안전한 비동기 감시 로직
# ==========================================================
def custom_wait_and_check():
    global g_emergency_stop
    while True:
        if g_emergency_stop:
            raise RuntimeError("EMERGENCY_STOP_TRIGGERED")
        
        try:
            status = check_motion()
            if status == 0:
                break
        except Exception:
            pass
        time.sleep(0.2)

def custom_movel(*args, **kwargs):
    amovel(*args, **kwargs)
    time.sleep(0.2)
    custom_wait_and_check()

def custom_movec(*args, **kwargs):
    amovec(*args, **kwargs)
    time.sleep(0.2)
    custom_wait_and_check()

def custom_movej(*args, **kwargs):
    amovej(*args, **kwargs)
    time.sleep(0.2)
    custom_wait_and_check()

def custom_mwait():
    pass

movel = custom_movel
movec = custom_movec
movej = custom_movej
mwait = custom_mwait

# ==========================================================
# 2. 운동별 로봇 보조 전략 (Strategy Pattern) 
# ==========================================================
class ExerciseStrategy:
    def execute_assist(self, td_coord, td_coord_shoulder=None):
        pass

class LateralRaiseStrategy(ExerciseStrategy):
    def __init__(self):
        self.X_OFFSET = -50.0           
        self.Y_APPROACH_OFFSET = 0.0 
        self.Y_SUPPORT_OFFSET = -100.0 
        self.Z_APPROACH_OFFSET = -100.0 
        self.TARGET_ANGLE_DEG = 80.0

    def execute_assist(self, td_coord, td_coord_shoulder=None):
        global g_is_recovery_triggered
        print("[사레레] 전완근 타겟 동적 보조 시작 (원호 궤적)")

        current_posx = get_current_posx()[0] 
        base_target_pos = list(td_coord[:3]) + list(current_posx[3:])

        approach_pos = list(base_target_pos)
        approach_pos[0] += self.X_OFFSET
        approach_pos[1] += self.Y_APPROACH_OFFSET
        approach_pos[2] = max(approach_pos[2] + self.Z_APPROACH_OFFSET, MIN_DEPTH)

        support_pos = list(approach_pos)
        support_pos[1] += self.Y_SUPPORT_OFFSET

        print(f"1차 접근 위치로 이동: X={approach_pos[0]:.1f}, Y={approach_pos[1]:.1f}, Z={approach_pos[2]:.1f}")
        movel(approach_pos, vel=VELOCITY, acc=ACC)
        if g_is_recovery_triggered: return 

        print(f"최종 지지 위치로 밀착: X={support_pos[0]:.1f}, Y={support_pos[1]:.1f}, Z={support_pos[2]:.1f}")
        movel(support_pos, vel=VELOCITY, acc=ACC)
        if g_is_recovery_triggered: return 

        if td_coord_shoulder is None:
            print("어깨 좌표가 없습니다. 교정을 중단합니다.")
            return

        S = np.array(td_coord_shoulder[:3])
        E_true = np.array(td_coord[:3]) 
        V_arm = E_true - S 

        E_start = np.array(support_pos[:3])
        V_robot = E_start - S 
        
        Z_up = np.array([0.0, 0.0, 1.0])
        axis = np.cross(V_arm, Z_up)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / axis_norm

        theta = np.radians(self.TARGET_ANGLE_DEG)
        
        R_mid = Rotation.from_rotvec(axis * (theta / 2.0))
        V_robot_mid = R_mid.apply(V_robot)
        pos1 = list(S + V_robot_mid) + list(current_posx[3:])

        R_end = Rotation.from_rotvec(axis * theta)
        V_robot_end = R_end.apply(V_robot)
        pos2 = list(S + V_robot_end) + list(current_posx[3:])

        print(f"위로 사레레 원호 밀어 올리기 (목표 Z={pos2[2]:.1f})")
        movec(pos1, pos2, vel=VELOCITY, acc=ACC)
        if g_is_recovery_triggered: return 

class ShoulderPressStrategy(ExerciseStrategy):
    def __init__(self):
        self.X_OFFSET = 0.0             
        self.Y_SUPPORT_OFFSET = -100.0    
        self.Z_APPROACH_OFFSET = -50.0 
        self.TARGET_ANGLE_DEG = 100.0

    def execute_assist(self, td_coord, td_coord_shoulder=None):
        global g_is_recovery_triggered
        print("[숄더프레스] 팔꿈치 하단 동적 보조 시작 (원호 궤적)")

        current_posx = get_current_posx()[0] 
        target_pos = list(td_coord[:3]) + list(current_posx[3:])

        target_pos[0] += self.X_OFFSET

        approach_pos = list(target_pos)
        approach_pos[2] = max(approach_pos[2] + self.Z_APPROACH_OFFSET, MIN_DEPTH)

        print(f"팔꿈치 하단 위치로 접근: X={approach_pos[0]:.1f}, Y={approach_pos[1]:.1f}, Z={approach_pos[2]:.1f}")
        movel(approach_pos, vel=VELOCITY, acc=ACC)
        if g_is_recovery_triggered: return 

        support_pos = list(approach_pos)
        support_pos[1] += self.Y_SUPPORT_OFFSET

        print(f"최종 지지 위치로 밀착: X={support_pos[0]:.1f}, Y={support_pos[1]:.1f}, Z={support_pos[2]:.1f}")
        movel(support_pos, vel=VELOCITY, acc=ACC)
        if g_is_recovery_triggered: return 

        if td_coord_shoulder is None:
            print("어깨 좌표가 없습니다. 교정을 중단합니다.")
            return

        S = np.array(td_coord_shoulder[:3])
        E_true = np.array(td_coord[:3]) 
        V_arm = E_true - S 

        E_start = np.array(support_pos[:3])
        V_robot = E_start - S 
        
        Z_up = np.array([0.0, 0.0, 1.0])
        axis = np.cross(V_arm, Z_up)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / axis_norm

        theta = np.radians(self.TARGET_ANGLE_DEG)
        
        R_mid = Rotation.from_rotvec(axis * (theta / 2.0))
        V_robot_mid = R_mid.apply(V_robot)
        pos1 = list(S + V_robot_mid) + list(current_posx[3:])

        R_end = Rotation.from_rotvec(axis * theta)
        V_robot_end = R_end.apply(V_robot)
        pos2 = list(S + V_robot_end) + list(current_posx[3:])

        print(f"위로 원호 밀어 올리기 (목표 Z={pos2[2]:.1f})")
        movec(pos1, pos2, vel=VELOCITY, acc=ACC)
        if g_is_recovery_triggered: return 

class BicepCurlStrategy(ExerciseStrategy):
    def __init__(self):
        self.X_OFFSET = 30.0
        self.Z_APPROACH_OFFSET = 10.0  
        self.Y_APPROACH_OFFSET = -50.0   
        self.Y_SUPPORT_OFFSET = -50.0    

    def execute_assist(self, td_coord, td_coord_shoulder=None):
        global g_is_supporting, g_is_recovery_triggered 

        print("[이두컬] 정적 지지 시작")

        current_posx = get_current_posx()[0] 
        base_pos = list(td_coord[:3]) + list(current_posx[3:])

        approach_pos = list(base_pos)
        approach_pos[0] += self.X_OFFSET
        approach_pos[1] += self.Y_APPROACH_OFFSET
        approach_pos[2] = max(approach_pos[2] + self.Z_APPROACH_OFFSET, MIN_DEPTH)

        print(f"1차 접근 위치로 이동: X={approach_pos[0]:.1f}, Y={approach_pos[1]:.1f}, Z={approach_pos[2]:.1f}")
        movel(approach_pos, vel=VELOCITY, acc=ACC)
        if g_is_recovery_triggered: return 

        support_pos = list(approach_pos)
        support_pos[1] += self.Y_SUPPORT_OFFSET
        
        print(f"최종 지지 위치로 밀착: X={support_pos[0]:.1f}, Y={support_pos[1]:.1f}, Z={support_pos[2]:.1f}")
        movel(support_pos, vel=VELOCITY, acc=ACC)
        if g_is_recovery_triggered: return 

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

        self.package_path = get_package_share_directory("rehab_assist_robot") 
        self.gripper2cam_path = os.path.join(self.package_path, "resource", "T_gripper2camera_diff_braket.npy")
        self.gripper2cam = np.load(self.gripper2cam_path)

        self.strategies = {
            'bicep_curl': BicepCurlStrategy(),
            'shoulder_press': ShoulderPressStrategy(),
            'lateral_raise': LateralRaiseStrategy()
        }
        self.current_exercise = 'shoulder_press'

        self.cb_group = ReentrantCallbackGroup()

        self.mode_sub = self.create_subscription(
            String, '/set_exercise_mode', self.mode_callback, 10, callback_group=self.cb_group 
        )

        self.latest_elbow_cam_pos = None
        self.correction_sub = self.create_subscription(
            Point, '/right_elbow_3d', self.correction_target_callback, 10, callback_group=self.cb_group 
        )
        
        self.latest_shoulder_cam_pos = None
        self.shoulder_sub = self.create_subscription(
            Point, '/right_shoulder_3d', self.shoulder_target_callback, 10, callback_group=self.cb_group 
        )

        self.sys_cmd_sub = self.create_subscription(
            String, '/system_command', self.sys_cmd_callback, 10, callback_group=self.cb_group 
        )
        
        self.emg_sub = self.create_subscription(
            Bool, '/emergency_stop', self.emg_callback, 10, callback_group=self.cb_group 
        )

        self.correction_pub = self.create_publisher(String, '/end_correction', 10)

        self.get_logger().info(f"노드 시작! 현재 로봇 모드: [{self.current_exercise}]")

    def emg_callback(self, msg):
        global g_emergency_stop
        if msg.data:
            self.get_logger().warn("비상 정지 신호 수신. 플래그를 활성화합니다.")
            g_emergency_stop = True
        else:
            g_emergency_stop = False

    def shoulder_target_callback(self, msg):
        if msg.z > 0:
            self.latest_shoulder_cam_pos = [msg.x, msg.y, msg.z]
            self.try_execute_assist()

    def correction_target_callback(self, msg):
        if msg.z > 0:
            self.latest_elbow_cam_pos = [msg.x, msg.y, msg.z]
            self.try_execute_assist()

    def try_execute_assist(self):
        global g_is_supporting
        if self.is_moving or g_is_supporting:
            return

        if self.latest_elbow_cam_pos is None:
            return

        if self.current_exercise in ['lateral_raise', 'shoulder_press']:
            if self.latest_shoulder_cam_pos is None:
                return

        self.is_moving = True
        threading.Thread(target=self._execute_assist_thread, daemon=True).start()

    def _execute_assist_thread(self):
        global g_is_supporting, g_is_recovery_triggered, g_emergency_stop
        g_is_recovery_triggered = False
        try:
            robot_posx = get_current_posx()[0]
            
            td_coord_elbow = self.transform_to_base(self.latest_elbow_cam_pos, robot_posx)
            
            td_coord_shoulder = None
            if self.latest_shoulder_cam_pos is not None:
                td_coord_shoulder = self.transform_to_base(self.latest_shoulder_cam_pos, robot_posx)            
            
            strategy = self.strategies.get(self.current_exercise)
            if strategy:
                strategy.execute_assist(td_coord_elbow, td_coord_shoulder)

        except Exception as e:
            self.get_logger().error(f"교정 동작 중 에러: {e}")
            
            if str(e) == "EMERGENCY_STOP_TRIGGERED":
                self.get_logger().info("비상 정지 감지! 기존 모션을 덮어쓰기하여 즉각 제자리에 정지합니다.")

                try:
                    cur_pos = get_current_posx()[0]
                    movel(cur_pos, vel=500, acc=500)
                    # [수정] 절대 위치(과거 위치) 조회 후 덮어쓰기로 인한 역주행 현상을 방지하기 위해,
                    # 제어기 내부의 현재 상태 기준 상대 이동량(mod=1) 0을 하달하여 즉각 감속 정지 유도
                    # amovel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vel=300, acc=300, ref=0, mod=1)
                except Exception as e:
                    self.get_logger().error(f'비상 정지 모션 덮어쓰기 중 에러 발생: {e}')
        


        finally:
            self.latest_elbow_cam_pos = None
            self.latest_shoulder_cam_pos = None

            if g_emergency_stop:
                self.get_logger().info("3초 대기 후 초기 위치로 강제 복귀합니다.")
                g_emergency_stop = False
                time.sleep(3.0)
                self.move_to_init_pos(force=True)

            elif g_is_recovery_triggered:
                self.get_logger().info("복구 모드: 3초 대기 후 초기 위치로 복귀합니다.")
                time.sleep(3.0)
                self.move_to_init_pos()
                g_is_recovery_triggered = False
                
                msg = String()
                msg.data = "비상 정지 후 복구가 완료되었습니다."
                self.correction_pub.publish(msg)

            else:
                if not g_is_supporting:
                    msg = String()
                    msg.data = "자세 교정이 완료되었습니다. 원위치로 복귀합니다."
                    self.correction_pub.publish(msg)
                    
                    time.sleep(4.0) 
                    self.move_to_init_pos_slowly()

            self.is_moving = False
            self.latest_elbow_cam_pos = None
            self.latest_shoulder_cam_pos = None

    def sys_cmd_callback(self, msg):
        global g_is_supporting, g_is_recovery_triggered, g_emergency_stop 
        
        if msg.data == "START_EXERCISE":
            self.get_logger().info("새로운 운동 시작 명령 감지. 카메라 뷰 확보를 위해 로봇을 초기 위치로 복귀시킵니다.")
            if g_is_supporting:
                release_compliance_ctrl()
                g_is_supporting = False
            self.move_to_init_pos()

        elif msg.data == "END_EXERCISE":
            if g_is_supporting:
                self.get_logger().info("종료 신호 수신. 정적 지지를 해제합니다.")
                release_compliance_ctrl()
                g_is_supporting = False
                self.move_to_init_pos()
                
        elif msg.data == "RECOVERY": 
            self.get_logger().info("정지 명령 수신. 플래그를 활성화합니다.")
            g_is_recovery_triggered = True
            g_emergency_stop = True
            if g_is_supporting:
                release_compliance_ctrl()
                g_is_supporting = False

    def move_to_init_pos(self, force=False):
        if self.is_moving and not force:
            self.get_logger().warn("로봇이 현재 교정 동작 중입니다. 복귀 명령을 무시하거나 대기합니다.")
            return
            
        self.is_moving = True
        try:
            init_pos = [48.07, 29.12, 113.41, 131.73, -117.85, 62.66]
            self.get_logger().info("초기 위치(카메라 촬영 뷰)로 로봇 이동 시작...")
            movej(init_pos, vel=VELOCITY, acc=ACC)
            self.get_logger().info("초기 위치 도착 완료! 측면 뷰가 성공적으로 확보되었습니다.")
        except Exception as e:
            self.get_logger().error(f"초기 위치 이동 중 에러 발생: {e}")
        finally:
            self.is_moving = False
    
    def move_to_init_pos_slowly(self):
        try:
            init_pos = [48.07, 29.12, 113.41, 131.73, -117.85, 62.66]
            self.get_logger().info("카메라 뷰 확보를 위해 천천히 초기 위치로 복귀합니다.")
            movej(init_pos, vel=VELOCITY * 0.3, acc=ACC * 0.3)
        except Exception as e:
            self.get_logger().error(f"저속 초기 위치 이동 중 에러 발생: {e}")

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

    def transform_to_base(self, camera_coords, robot_pos):
        coord = np.append(np.array(camera_coords), 1) 
        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)
        base2cam = base2gripper @ self.gripper2cam
        return np.dot(base2cam, coord)[:3]

    def init_robot(self):
        JReady = [0, 0, 90, 0, 90, 0]
        movej(JReady, vel=VELOCITY, acc=ACC)
        
        self.move_to_init_pos()
        
        gripper.close_gripper()

def main(args=None):
    node = PostureCorrector()
    
    executor = MultiThreadedExecutor()
    executor.add_node(dsr_node)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
