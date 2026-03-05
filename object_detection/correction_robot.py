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

        self.wrist = None

        self.create_subscription(
            Point,
            "/rehab/wrist_position",
            self.wrist_callback,
            10
        )

        self.create_service(
            Trigger,
            "/correction/start",
            self.correction_service
        )

    def wrist_callback(self,msg):

        self.wrist = msg

    def correction_service(self,req,res):

        if self.wrist is None:

            res.success=False
            return res

        robot_pose = get_current_posx()[0]

        start_pose = [

            self.wrist.x,
            self.wrist.y,
            self.wrist.z,
            robot_pose[3],
            robot_pose[4],
            robot_pose[5]

        ]

        target_pose = [

            self.wrist.x,
            self.wrist.y,
            self.wrist.z + 100,
            robot_pose[3],
            robot_pose[4],
            robot_pose[5]

        ]

        movel(start_pose,vel=VEL,acc=ACC)
        movel(target_pose,vel=VEL,acc=ACC)

        res.success=True

        return res


def main():

    rclpy.init()

    node = CorrectionRobot()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()