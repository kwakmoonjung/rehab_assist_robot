#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_srvs.srv import Trigger

import DR_init

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

VEL = 30
ACC = 30

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

from DSR_ROBOT2 import movel, get_current_posx


class CorrectionRobot(Node):

    def __init__(self):

        super().__init__("correction_robot")

        self.wrist_pos = None

        self.create_subscription(
            Point,
            "/rehab/wrist_position",
            self.wrist_callback,
            10
        )

        self.create_service(
            Trigger,
            "/correction/start",
            self.start_correction
        )

        self.get_logger().info("Correction robot ready")

    def wrist_callback(self,msg):

        self.wrist_pos = msg

    def start_correction(self,request,response):

        if self.wrist_pos is None:

            response.success=False
            return response

        robot_pose = get_current_posx()[0]

        start_pose = [

            self.wrist_pos.x,
            self.wrist_pos.y,
            self.wrist_pos.z,
            robot_pose[3],
            robot_pose[4],
            robot_pose[5]

        ]

        target_pose = [

            self.wrist_pos.x,
            self.wrist_pos.y,
            self.wrist_pos.z + 100,
            robot_pose[3],
            robot_pose[4],
            robot_pose[5]

        ]

        movel(start_pose,vel=VEL,acc=ACC)
        movel(target_pose,vel=VEL,acc=ACC)

        response.success=True
        return response


def main():

    rclpy.init()

    node = CorrectionRobot()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()