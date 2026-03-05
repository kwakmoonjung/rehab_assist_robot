#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import Float32
from geometry_msgs.msg import Point

from object_detection.realsense import ImgNode
from ultralytics import YOLO


class ObjectDetectionNode(Node):

    def __init__(self):

        super().__init__("object_detection_node")

        self.img_node = ImgNode()

        self.model = YOLO("yolo11n-pose.pt")

        self.angle_pub = self.create_publisher(
            Float32,
            "/rehab/elbow_angle",
            10
        )

        self.wrist_pub = self.create_publisher(
            Point,
            "/rehab/wrist_position",
            10
        )

        self.timer = self.create_timer(
            0.03,
            self.loop
        )

        self.get_logger().info("YOLO pose detection started")

    def loop(self):

        rclpy.spin_once(self.img_node)

        frame = self.img_node.color_frame

        if frame is None:
            return

        results = self.model(frame, verbose=False)

        if len(results) == 0:
            return

        kp = results[0].keypoints

        if kp is None:
            return

        keypoints = kp.xy[0].cpu().numpy()

        shoulder = keypoints[6]
        elbow = keypoints[8]
        wrist = keypoints[10]

        angle = self.calculate_angle(
            shoulder,
            elbow,
            wrist
        )

        angle_msg = Float32()
        angle_msg.data = float(angle)

        self.angle_pub.publish(angle_msg)

        wrist_msg = Point()

        wrist_msg.x = float(wrist[0])
        wrist_msg.y = float(wrist[1])
        wrist_msg.z = 0.0

        self.wrist_pub.publish(wrist_msg)

    def calculate_angle(self,a,b,c):

        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)

        cosine = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))

        angle = np.degrees(
            np.arccos(
                np.clip(cosine,-1,1)
            )
        )

        return angle


def main():

    rclpy.init()

    node = ObjectDetectionNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()