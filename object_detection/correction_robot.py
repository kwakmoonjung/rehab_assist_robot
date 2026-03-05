#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import DR_init
import numpy as np

from od_msg.srv import SrvDepthPosition

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

VEL = 40
ACC = 40

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

from DSR_ROBOT2 import movej, movel, get_current_posx, mwait


class CorrectionRobot(Node):

    def __init__(self):

        super().__init__("correction_robot")

        self.client = self.create_client(
            SrvDepthPosition,
            "/get_3d_position"
        )

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waiting detection service")

        self.req = SrvDepthPosition.Request()

        self.init_robot()

    def init_robot(self):

        ready = [0,0,90,0,90,0]

        movej(ready,vel=VEL,acc=ACC)

        mwait()

    def get_person_position(self):

        self.req.target = "person"

        future = self.client.call_async(self.req)

        rclpy.spin_until_future_complete(self,future)

        result = future.result()

        if result is None:
            return None

        return np.array(result.depth_position)

    def correct_z_up(self):

        pos = self.get_person_position()

        if pos is None:
            self.get_logger().warn("no person detected")
            return

        target = [
            pos[0],
            pos[1],
            pos[2] + 100
        ]

        robot_pose = get_current_posx()[0]

        move_pose = [
            target[0],
            target[1],
            target[2],
            robot_pose[3],
            robot_pose[4],
            robot_pose[5]
        ]

        movel(move_pose,vel=VEL,acc=ACC)

        mwait()


def main():

    rclpy.init()

    node = CorrectionRobot()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()