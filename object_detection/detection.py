#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import Float32
from geometry_msgs.msg import Point

from ultralytics import YOLO
from object_detection.realsense import ImgNode


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

        # camera intrinsics (캘리브레이션 값 입력)
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0

        # camera → robot transform (예시)
        self.T_cam_robot = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ])

        self.timer = self.create_timer(0.03, self.loop)

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

        # depth
        depth = self.img_node.get_depth(
            int(wrist[0]),
            int(wrist[1])
        )

        if depth is None:
            return

        p_cam = self.pixel_to_camera(
            wrist[0],
            wrist[1],
            depth
        )

        p_robot = self.camera_to_robot(
            p_cam
        )

        wrist_msg = Point()
        wrist_msg.x = float(p_robot[0])
        wrist_msg.y = float(p_robot[1])
        wrist_msg.z = float(p_robot[2])

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

    def pixel_to_camera(self,u,v,z):

        X = (u - self.cx) * z / self.fx
        Y = (v - self.cy) * z / self.fy

        return np.array([X,Y,z])

    def camera_to_robot(self,p):

        p4 = np.append(p,1)

        pr = self.T_cam_robot @ p4

        return pr[:3]


def main():

    rclpy.init()

    node = ObjectDetectionNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()