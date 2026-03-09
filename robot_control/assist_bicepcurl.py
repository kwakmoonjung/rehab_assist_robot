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


# ==============================
# 로봇 / 그리퍼 기본 설정
# ==============================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

# 사람 보조 동작이므로 속도는 기존보다 낮게
MOVE_VEL = 30
MOVE_ACC = 30
APPROACH_VEL = 20
APPROACH_ACC = 20
RETREAT_VEL = 30
RETREAT_ACC = 30

# ==============================
# 보조 위치 / 힘 제어 튜닝값
# ==============================
# /target_correction_3d 는 이제 "팔꿈치 3D 좌표"라고 가정
# 팔꿈치 뒤쪽으로 살짝 들어가기 위한 카메라축 기준 오프셋(mm)
# 값이 크면 더 뒤쪽으로 들어감
ELBOW_BACK_OFFSET_CAM_Z = 60.0

# 팔꿈치로 바로 들이박지 않도록 먼저 위쪽에서 접근
APPROACH_OFFSET_BASE_Z = 80.0

# 너무 아래로 내려가지 않도록 최소 베이스 Z
MIN_BASE_Z = 80.0

# 좌/우 미세 보정이 필요하면 사용
BASE_Y_OFFSET = 0.0

# 컴플라이언스 강성
# 너무 낮으면 흔들리고, 너무 높으면 "버티는 느낌"이 강해짐
# 지금은 "살짝 받쳐주는" 쪽으로 중간 이하 수준
COMPLIANCE_STX = [1200, 1200, 800, 100, 100, 100]

# 보조 힘 제어
# Tool Z 방향으로 약하게 힘을 줌
# 방향이 반대로 나오면 부호만 바꾸면 됨 (+6.0 <-> -6.0)
ASSIST_FORCE_TOOL_Z = -6.0

# 보조 유지 시간
ASSIST_HOLD_SEC = 2.5

# 그리퍼 닫힘 힘
GRIP_FORCE = 40


DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class PostureCorrector(Node):
    def __init__(self):
        super().__init__("posture_corrector_node")

        try:
            global movej, movel, get_current_posj, get_current_posx, mwait
            global task_compliance_ctrl, set_desired_force, release_force
            global release_compliance_ctrl, set_external_force_reset, set_ref_coord
            from DSR_ROBOT2 import (
                movej,
                movel,
                get_current_posj,
                get_current_posx,
                mwait,
                task_compliance_ctrl,
                set_desired_force,
                release_force,
                release_compliance_ctrl,
                set_external_force_reset,
                set_ref_coord,
            )

            # 환경에 따라 상수 import가 안 될 수 있어 fallback도 둠
            try:
                global DR_TOOL, DR_FC_MOD_REL
                from DSR_ROBOT2 import DR_TOOL, DR_FC_MOD_REL
            except Exception:
                DR_TOOL = 2
                DR_FC_MOD_REL = 1

        except ImportError as e:
            self.get_logger().error(f"DSR_ROBOT2 라이브러리를 불러오지 못했습니다: {e}")
            sys.exit(1)

        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

        self.package_path = get_package_share_directory("rehab_assist_robot")
        self.is_moving = False

        self.init_robot()

        # 여기 들어오는 좌표는 이제 "팔꿈치 3D 좌표"
        self.correction_sub = self.create_subscription(
            Point,
            "/target_correction_3d",
            self.correction_target_callback,
            10,
        )

        self.get_logger().info("🤖 posture_corrector.py 시작")
        self.get_logger().info("📌 /target_correction_3d 를 '팔꿈치 3D 좌표'로 사용합니다.")
        self.get_logger().info("🦾 그리퍼 닫힘 상태로 팔꿈치 뒤쪽에 접근 후, 컴플라이언스로 약하게 보조합니다.")

    # ==========================================
    # 좌표 변환 유틸
    # ==========================================
    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        """
        camera_coords : 카메라 기준 3D 좌표 [x, y, z]
        robot_pos     : 현재 로봇 task pose [x, y, z, rx, ry, rz]
        """
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords, dtype=np.float64), 1.0)

        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)

        # 기존 코드 구조 유지
        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)

        return td_coord[:3]

    # ==========================================
    # 그리퍼 유틸
    # ==========================================
    def close_gripper_safely(self):
        self.get_logger().info("✊ 그리퍼 닫기")
        self.gripper.close_gripper(force_val=GRIP_FORCE)
        time.sleep(1.5)

        try:
            for _ in range(20):
                status = self.gripper.get_status()
                if not status[0]:
                    break
                time.sleep(0.1)
        except Exception:
            pass

    def open_gripper_safely(self):
        self.get_logger().info("🖐️ 그리퍼 열기")
        self.gripper.open_gripper()
        time.sleep(1.0)

        try:
            for _ in range(20):
                status = self.gripper.get_status()
                if not status[0]:
                    break
                time.sleep(0.1)
        except Exception:
            pass

    # ==========================================
    # 컴플라이언스 보조
    # ==========================================
    def start_soft_elbow_support(self):
        """
        팔꿈치 뒤쪽에서 완전 고정이 아니라 '살짝 받쳐주는' 정도의 보조.
        Tool 기준 Z축 방향으로 약한 힘을 걸어줌.
        힘 방향이 반대로 나오면 ASSIST_FORCE_TOOL_Z 부호만 바꾸면 됨.
        """
        self.get_logger().info("🧩 컴플라이언스 시작")
        set_ref_coord(DR_TOOL)

        # 접촉 직전 위치에서 외력 초기화
        try:
            set_external_force_reset()
        except Exception as e:
            self.get_logger().warn(f"외력 초기화 실패(무시 가능): {e}")

        task_compliance_ctrl(stx=COMPLIANCE_STX, time=0.2)

        fd = [0.0, 0.0, ASSIST_FORCE_TOOL_Z, 0.0, 0.0, 0.0]
        fdir = [0, 0, 1, 0, 0, 0]

        set_desired_force(fd, dir=fdir, mod=DR_FC_MOD_REL)
        self.get_logger().info(
            f"🤲 약한 보조 힘 적용 중... Tool Z force={ASSIST_FORCE_TOOL_Z}N, hold={ASSIST_HOLD_SEC}s"
        )
        time.sleep(ASSIST_HOLD_SEC)

    def stop_soft_elbow_support(self):
        self.get_logger().info("🛑 컴플라이언스 종료")
        try:
            release_force()
            time.sleep(0.2)
        except Exception as e:
            self.get_logger().warn(f"release_force 실패: {e}")

        try:
            release_compliance_ctrl()
            time.sleep(0.2)
        except Exception as e:
            self.get_logger().warn(f"release_compliance_ctrl 실패: {e}")

    # ==========================================
    # 메인 콜백
    # ==========================================
    def correction_target_callback(self, msg):
        """
        msg.x, msg.y, msg.z : 카메라 기준 팔꿈치 3D 좌표(mm 가정)
        """
        if self.is_moving:
            self.get_logger().warn("⚠️ 현재 로봇이 동작 중입니다. 새 명령은 무시합니다.")
            return

        if msg.z <= 0:
            self.get_logger().warn("⚠️ 유효하지 않은 팔꿈치 depth 값입니다.")
            return

        self.is_moving = True

        try:
            # 1) 먼저 그리퍼를 닫은 상태로 만든다.
            self.close_gripper_safely()

            # 2) 팔꿈치 '뒤쪽'으로 들어가기 위해 카메라 z 기준으로 오프셋
            elbow_cam_pos = [msg.x, msg.y, msg.z + ELBOW_BACK_OFFSET_CAM_Z]

            gripper2cam_path = os.path.join(
                self.package_path,
                "resource",
                "T_gripper2camera_diff_braket.npy",
            )

            # 3) 현재 pose 기준으로 카메라 -> 베이스 좌표 변환
            current_posx = get_current_posx()[0]
            target_base_xyz = self.transform_to_base(
                elbow_cam_pos,
                gripper2cam_path,
                current_posx,
            )

            target_base_xyz[1] += BASE_Y_OFFSET
            target_base_xyz[2] = max(float(target_base_xyz[2]), MIN_BASE_Z)

            # 4) 현재 툴 자세는 유지하고 XYZ만 팔꿈치 뒤쪽으로 보냄
            latest_posx = get_current_posx()[0]
            target_support_pos = list(target_base_xyz[:3]) + list(latest_posx[3:])

            # 5) 바로 들이대지 않고 위에서 먼저 접근
            pre_approach_pos = list(target_support_pos)
            pre_approach_pos[2] += APPROACH_OFFSET_BASE_Z

            self.get_logger().info(
                f"📍 pre-approach 이동: X={pre_approach_pos[0]:.1f}, "
                f"Y={pre_approach_pos[1]:.1f}, Z={pre_approach_pos[2]:.1f}"
            )
            movel(pre_approach_pos, vel=APPROACH_VEL, acc=APPROACH_ACC)
            mwait()

            self.get_logger().info(
                f"📍 팔꿈치 뒤 보조 위치 이동: X={target_support_pos[0]:.1f}, "
                f"Y={target_support_pos[1]:.1f}, Z={target_support_pos[2]:.1f}"
            )
            movel(target_support_pos, vel=APPROACH_VEL, acc=APPROACH_ACC)
            mwait()

            # 6) 컴플라이언스로 살짝 받쳐주는 보조
            self.start_soft_elbow_support()

            # 7) 보조 종료 후 살짝 뒤로 빠짐
            self.stop_soft_elbow_support()

            self.get_logger().info("↩️ 보조 종료 후 retreat")
            movel(pre_approach_pos, vel=RETREAT_VEL, acc=RETREAT_ACC)
            mwait()

            # 필요하면 마지막에 열어도 되고, 닫은 상태 유지해도 됨
            # 여기서는 안전하게 다시 연다.
            self.open_gripper_safely()

            self.get_logger().info("✅ 팔꿈치 보조 동작 완료")

        except Exception as e:
            self.get_logger().error(f"❌ 보조 동작 중 에러 발생: {e}")

            # 예외 시에도 컴플라이언스가 남아 있지 않게 정리
            try:
                self.stop_soft_elbow_support()
            except Exception:
                pass

            try:
                self.open_gripper_safely()
            except Exception:
                pass

        finally:
            self.is_moving = False

    # ==========================================
    # 초기 위치
    # ==========================================
    def init_robot(self):
        JReady = [0, 0, 90, 0, 90, 0]
        movej(JReady, vel=MOVE_VEL, acc=MOVE_ACC)
        mwait()

        init_pos = [73.03, 71.08, -33.55, 23.68, -123.47, -74.72]
        movej(init_pos, vel=MOVE_VEL, acc=MOVE_ACC)
        mwait()

        self.get_logger().info(f"초기 조인트 위치 세팅 완료: {init_pos}")
        self.open_gripper_safely()


def main(args=None):
    rclpy.init(args=args)

    dsr_node = rclpy.create_node("dsr_robot_core_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    node = PostureCorrector()
    rclpy.spin(node)

    node.destroy_node()
    dsr_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()