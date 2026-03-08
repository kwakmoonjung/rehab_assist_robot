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

    # [추가] USB 2.1 연결 카메라용 해상도 설정 변수
    fixed_override_args = {
        'depth_module.depth_profile': '640x480x15',
        'rgb_camera.color_profile': '640x480x15',
        'enable_depth': 'false',         # [추가]
        'align_depth.enable': 'false',   # [추가]
        'enable_rgbd': 'false',          # [추가]
        'pointcloud.enable': 'false'     # [추가]
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
            **fixed_override_args,  # [추가] 고정 카메라 해상도 덮어쓰기
            'serial_no': '_215322078366',  # ⭐️ 문자열 인식을 위해 앞에 _ 추가
            'camera_namespace': 'fixed',
            'camera_name': 'camera'
        }.items()
    )

    return LaunchDescription([
        robot_cam,
        fixed_cam
    ])