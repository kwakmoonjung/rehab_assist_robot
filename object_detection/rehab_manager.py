#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import numpy as np
import time

from correction_robot import CorrectionRobot


class RehabManager(Node):

    def __init__(self):

        super().__init__("rehab_manager")

        self.angles = []

        self.subscription = self.create_subscription(
            Float32,
            "/rehab/elbow_angle",
            self.angle_callback,
            10
        )

        self.robot = CorrectionRobot()

        self.get_logger().info("rehab manager started")

    def angle_callback(self,msg):

        self.angles.append(msg.data)

    def analyze_session(self):

        if len(self.angles) < 5:

            print("not enough data")

            return False

        min_angle = np.min(self.angles)
        max_angle = np.max(self.angles)

        print("analysis result")
        print("min:",min_angle)
        print("max:",max_angle)

        if max_angle < 110:

            print("팔이 충분히 올라가지 않았습니다")

            return True

        return False

    def run_session(self):

        print("운동 시작 (10초 기록)")

        start = time.time()

        while time.time() - start < 10:

            rclpy.spin_once(self)

        print("운동 종료")

        need_correction = self.analyze_session()

        if need_correction:

            print("교정 시작")

            self.robot.correct_z_up()

        else:

            print("정상 자세")


def main():

    rclpy.init()

    node = RehabManager()

    node.run_session()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()