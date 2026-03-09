#!/usr/bin/env python3
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


# ==============================
# 로봇 기본 설정
# ==============================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

MOVE_VEL = 30
MOVE_ACC = 30
APPROACH_VEL = 20
APPROACH_ACC = 20
RETREAT_VEL = 30
RETREAT_ACC = 30

# ==============================
# 손목 추종 설정
# ==============================
# 오른손목 따라가기: "/right_wrist_3d"
# 왼손목 따라가기: "/left_wrist_3d"
TARGET_WRIST_TOPIC = "/right_wrist_3d"

# 목표 지점으로 바로 들이대지 않도록 위에서 먼저 접근
APPROACH_OFFSET_BASE_Z = 80.0

# 너무 낮은 위치로 내려가는 것 방지
MIN_BASE_Z = 80.0

# 필요 시 좌우 미세 보정
BASE_Y_OFFSET = 0.0

# 목표 위치에서 잠깐 정지
TARGET_WAIT_SEC = 0.5


DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class PostureCorrector(Node):
    def __init__(self):
        super().__init__("posture_corrector_node")

        try:
            global movej, movel, get_current_posx, mwait
            from DSR_ROBOT2 import (
                movej,
                movel,
                get_current_posx,
                mwait,
            )

        except ImportError as e:
            self.get_logger().error(f"DSR_ROBOT2 라이브러리를 불러오지 못했습니다: {e}")
            sys.exit(1)

        self.package_path = get_package_share_directory("rehab_assist_robot")
        self.is_moving = False

        self.init_robot()

        # 스페이스바 눌렀을 때 pose_tracking_node 가 발행하는 손목 3D 좌표 구독
        self.wrist_sub = self.create_subscription(
            Point,
            TARGET_WRIST_TOPIC,
            self.wrist_target_callback,
            10,
        )

        self.get_logger().info("🤖 posture_corrector.py 시작")
        self.get_logger().info(f"📡 구독 토픽: {TARGET_WRIST_TOPIC}")
        self.get_logger().info("📌 스페이스바로 발행된 손목 3D 좌표를 받아 로봇이 해당 위치로 이동합니다.")

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

        base2cam = base2gripper @ gripper2cam
        target_coord = np.dot(base2cam, coord)

        return target_coord[:3]

    # ==========================================
    # 메인 콜백
    # ==========================================
    def wrist_target_callback(self, msg):
        """
        msg.x, msg.y, msg.z : 로봇팔 카메라 기준 손목 3D 좌표(mm 가정)
        """
        if self.is_moving:
            self.get_logger().warn("⚠️ 현재 로봇이 동작 중이라 새 손목 좌표는 무시합니다.")
            return

        if msg.z <= 0:
            self.get_logger().warn("⚠️ 유효하지 않은 손목 depth 값입니다.")
            return

        self.is_moving = True

        try:
            wrist_cam_pos = [msg.x, msg.y, msg.z]

            gripper2cam_path = os.path.join(
                self.package_path,
                "resource",
                "T_gripper2camera_diff_braket.npy",
            )

            if not os.path.exists(gripper2cam_path):
                raise FileNotFoundError(f"캘리브레이션 파일이 없습니다: {gripper2cam_path}")

            # 현재 로봇 pose 기준 카메라 -> 베이스 변환
            current_posx = get_current_posx()[0]
            target_base_xyz = self.transform_to_base(
                wrist_cam_pos,
                gripper2cam_path,
                current_posx,
            )

            target_base_xyz[1] += BASE_Y_OFFSET
            target_base_xyz[2] = max(float(target_base_xyz[2]), MIN_BASE_Z)

            # 현재 툴 자세(rx, ry, rz)는 유지하고 XYZ만 변경
            latest_posx = get_current_posx()[0]
            target_pos = list(target_base_xyz[:3]) + list(latest_posx[3:])

            # 위에서 먼저 접근
            pre_approach_pos = list(target_pos)
            pre_approach_pos[2] += APPROACH_OFFSET_BASE_Z

            self.get_logger().info(
                f"📷 wrist camera coord: X={msg.x:.1f}, Y={msg.y:.1f}, Z={msg.z:.1f}"
            )
            self.get_logger().info(
                f"📍 pre-approach 이동: X={pre_approach_pos[0]:.1f}, "
                f"Y={pre_approach_pos[1]:.1f}, Z={pre_approach_pos[2]:.1f}"
            )
            movel(pre_approach_pos, vel=APPROACH_VEL, acc=APPROACH_ACC)
            mwait()

            self.get_logger().info(
                f"📍 손목 목표 위치 이동: X={target_pos[0]:.1f}, "
                f"Y={target_pos[1]:.1f}, Z={target_pos[2]:.1f}"
            )
            movel(target_pos, vel=APPROACH_VEL, acc=APPROACH_ACC)
            mwait()

            time.sleep(TARGET_WAIT_SEC)

            self.get_logger().info("↩️ 원위치 상단으로 retreat")
            movel(pre_approach_pos, vel=RETREAT_VEL, acc=RETREAT_ACC)
            mwait()

            self.get_logger().info("✅ 손목 좌표 이동 완료")

        except Exception as e:
            self.get_logger().error(f"❌ 손목 이동 중 에러 발생: {e}")

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