#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
import numpy as np
import time


class RehabManager(Node):

    def __init__(self):

        super().__init__("rehab_manager")

        self.angles = []

        self.create_subscription(
            Float32,
            "/rehab/elbow_angle",
            self.angle_callback,
            10
        )

        self.client = self.create_client(
            Trigger,
            "/correction/start"
        )

    def angle_callback(self,msg):

        self.angles.append(msg.data)

    def analyze_session(self):

        if len(self.angles) < 10:
            return False

        angles = np.array(self.angles)

        max_angle = np.max(angles)
        min_angle = np.min(angles)

        ROM = max_angle - min_angle

        print("max:",max_angle)
        print("min:",min_angle)
        print("ROM:",ROM)

        if max_angle < 110:

            print("교정 필요")

            return True

        return False

    def run_session(self):

        start = time.time()

        while time.time() - start < 10:

            rclpy.spin_once(self)

        if self.analyze_session():

            req = Trigger.Request()

            future = self.client.call_async(req)

            rclpy.spin_until_future_complete(self,future)


def main():

    rclpy.init()

    node = RehabManager()

    node.run_session()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()