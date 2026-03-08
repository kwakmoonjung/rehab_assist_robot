import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    rs_launch_file = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'examples',
        'align_depth',
        'rs_align_depth_launch.py'
    )

    common_args = {
        'depth_module.depth_profile': '848x480x30',
        'rgb_camera.color_profile': '1280x720x30',
        'initial_reset': 'true',
        'align_depth.enable': 'true',
        'enable_rgbd': 'true',
        'pointcloud.enable': 'true'
    }

    # 1. 정면 로봇 카메라 (Robot Cam)
    robot_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch_file),
        launch_arguments={
            **common_args,
            'serial_no': '_141722076528',  # ⭐️ 문자열 인식을 위해 앞에 _ 추가
            'camera_namespace': 'robot',
            'camera_name': 'camera'
        }.items()
    )

    # 2. 측면 고정 카메라 (Fixed Cam)
    fixed_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch_file),
        launch_arguments={
            **common_args,
            'serial_no': '_215322078366',  # ⭐️ 문자열 인식을 위해 앞에 _ 추가
            'camera_namespace': 'fixed',
            'camera_name': 'camera'
        }.items()
    )

    return LaunchDescription([
        robot_cam,
        fixed_cam
    ])